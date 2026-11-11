"""Arquitectura agéntica Plan-and-Execute para la auditoría factual del discurso.

Flujo:
    Planner  -> descompone el discurso en pasos de verificación (una promesa/tema por paso)
    Executor -> ejecuta cada paso contra las fuentes documentales (benchmarks MEF/BCRP/JNE)
    Replanner-> si un paso no encuentra evidencia con la consulta estricta, la reformula
                con un léxico temático más amplio (segunda pasada)
    Salida   -> ``factual_rag_metrics`` con el mismo esquema que el agente factual, para que
                el juez y el reportero existentes funcionen sin cambios.

A diferencia del pipeline (agentes fijos en paralelo), aquí el *plan* es explícito y
auditable: queda en ``state["plan"]`` y sus métricas en ``state["plan_metrics"]``.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from ..agentes.factual_agent import (
    FACTUAL_BENCHMARKS,
    REITERATION_FACTOR,
    _find_matching_benchmarks,
    _normalize_for_matching,
    _slugify_topic,
)

AGENT_NAME = "PlanAndExecute"

# Léxico temático para la replanificación: la primera pasada usa los keywords estrictos
# del benchmark; si no hay coincidencia, el replanner amplía la consulta con este léxico.
THEME_LEXICON: Dict[str, List[str]] = {
    "Cronograma Electoral y Mandato Presidencial": [
        "elecciones", "eleccion", "comicios", "sufragio", "votacion", "votar",
        "urnas", "cronograma electoral", "adelanto", "mandato",
    ],
    "Reforma Constitucional": [
        "constitucion", "constituyente", "reforma", "carta magna", "referendum",
        "asamblea",
    ],
    "Consistencia Macroeconómica y Presupuestal MEF/BCRP": [
        "economia", "pbi", "presupuesto", "gasto", "inversion", "fiscal",
        "deficit", "crecimiento", "millones", "obras",
    ],
    "Seguridad y Paz Social": [
        "seguridad", "violencia", "policia", "paz", "orden", "bloqueos",
        "fallecidos", "muertos", "conflicto",
    ],
}

BENCHMARK_BY_TOPIC = {benchmark["topic"]: benchmark for benchmark in FACTUAL_BENCHMARKS}
NORMALIZED_LEXICON = {
    topic: [_normalize_for_matching(word).strip() for word in words]
    for topic, words in THEME_LEXICON.items()
}

MAX_PLAN_STEPS = 60
MAX_REPLANS_PER_STEP = 1


def _claim_candidates(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Selecciona los segmentos que ameritan un paso de verificación.

    Criterio: segmentos con una promesa extraída o con una mención directa a un
    benchmark regulatorio. Los saludos y texto protocolar quedan fuera del plan.
    """
    segments = state.get("segments", [])
    promise_segment_ids = {
        promise.get("segment_id") for promise in state.get("extracted_promises", [])
    }
    candidates = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        has_promise = segment.get("id") in promise_segment_ids
        has_regulatory_topic = bool(_find_matching_benchmarks(text))
        if has_promise or has_regulatory_topic:
            candidates.append(segment)
    if not candidates:
        candidates = [segment for segment in segments if str(segment.get("text", "")).strip()]
    return candidates[:MAX_PLAN_STEPS]


def build_plan(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Planner: convierte el discurso en una lista ordenada de pasos verificables."""
    plan: List[Dict[str, Any]] = []
    for index, segment in enumerate(_claim_candidates(state), start=1):
        text = str(segment.get("text", "")).strip()
        strict_matches = _find_matching_benchmarks(text)
        expected = max(strict_matches, key=lambda benchmark: benchmark["fact_target"]) if strict_matches else None
        plan.append({
            "step": index,
            "segment_id": segment["id"],
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "claim": text,
            "query": text,
            "expected_topic": expected["topic"] if expected else None,
            "attempts": 0,
            "status": "pendiente",
            "replanned": False,
            "source": None,
            "score": 0.0,
        })
    return plan


def _theme_lookup(claim: str) -> Dict[str, Any] | None:
    """Reformulación temática: elige el benchmark cuyo léxico aparece más en la promesa."""
    normalized = _normalize_for_matching(claim)
    best_topic, best_hits = None, 0
    for topic, words in NORMALIZED_LEXICON.items():
        hits = sum(1 for word in words if word and f" {word} " in normalized)
        if hits > best_hits:
            best_topic, best_hits = topic, hits
    return BENCHMARK_BY_TOPIC.get(best_topic) if best_topic else None


def _language_modifier(state: Dict[str, Any], segment_id: int) -> float:
    """Ajuste por sobre-afirmaciones/atenuadores, igual que en el agente factual."""
    linguistic_map = {
        row["id"]: row
        for row in state.get("linguistic_metrics", {}).get("by_segment", [])
    }
    features = linguistic_map.get(segment_id, {})
    hedges_ratio = features.get("hedges", 0.0) / 100.0
    assertives_ratio = features.get("truth_assertives", 0.0) / 100.0
    return (assertives_ratio * 0.4) + (hedges_ratio * 0.3)


def _resolve_step(step: Dict[str, Any], benchmark: Dict[str, Any], state: Dict[str, Any], mode: str) -> None:
    """Marca un paso como resuelto con la evidencia documental encontrada."""
    modifier = _language_modifier(state, step["segment_id"])
    step["status"] = "resuelto"
    step["source"] = benchmark["source"]
    step["topic"] = benchmark["topic"]
    step["score"] = float(np.clip(benchmark["fact_target"] + modifier, 0.0, 0.98))
    step["resolution"] = mode


def execute_plan(plan: List[Dict[str, Any]], state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Executor + Replanner: primera pasada estricta y segunda pasada temática."""
    for step in plan:
        step["attempts"] = 1
        strict_matches = _find_matching_benchmarks(step["claim"])
        if strict_matches:
            benchmark = max(strict_matches, key=lambda item: item["fact_target"])
            _resolve_step(step, benchmark, state, mode="keyword_estricto")
        else:
            step["status"] = "sin_evidencia"

    for step in plan:
        if step["status"] == "resuelto" or step["attempts"] > MAX_REPLANS_PER_STEP:
            continue
        step["attempts"] += 1
        benchmark = _theme_lookup(step["claim"])
        if benchmark:
            step["replanned"] = True
            _resolve_step(step, benchmark, state, mode="replanificacion_tematica")
        else:
            step["status"] = "sin_evidencia"

    return plan


def _build_rows(state: Dict[str, Any], plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Traduce el plan ejecutado al esquema ``by_segment`` que consume el juez."""
    resolved = [step for step in plan if step["status"] == "resuelto"]
    topics_in_order: List[str] = []
    for step in resolved:
        if step["topic"] not in topics_in_order:
            topics_in_order.append(step["topic"])
    anchor_by_topic = {
        topic: min((step for step in resolved if step["topic"] == topic), key=lambda item: item["start"])["segment_id"]
        for topic in topics_in_order
    }
    mentions = Counter(step["topic"] for step in resolved)
    step_by_segment = {step["segment_id"]: step for step in resolved}

    rows: List[Dict[str, Any]] = []
    for segment in state.get("segments", []):
        segment_id = segment["id"]
        row: Dict[str, Any] = {
            "id": segment_id,
            "text": segment.get("text", ""),
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
        }
        step = step_by_segment.get(segment_id)
        if step:
            is_anchor = anchor_by_topic[step["topic"]] == segment_id
            discrepancy = step["score"] if is_anchor else float(
                np.clip(step["score"] * REITERATION_FACTOR, 0.05, 0.60)
            )
            benchmark = BENCHMARK_BY_TOPIC[step["topic"]]
            row.update({
                "topic": step["topic"],
                "topic_key": _slugify_topic(step["topic"]),
                "source": step["source"],
                "audit_note": benchmark["audit_note"],
                "is_primary": is_anchor,
                "mention_count": int(mentions[step["topic"]]),
                "plan_step": step["step"],
                "resolution": step["resolution"],
            })
            if not is_anchor:
                row["reiteration_of"] = anchor_by_topic[step["topic"]]
        else:
            discrepancy = float(np.clip(0.12 + (_language_modifier(state, segment_id) * 0.35), 0.05, 0.35))
            row.update({
                "topic": "Sin evidencia documental",
                "topic_key": "sin_evidencia",
                "source": "Plan-and-Execute",
                "audit_note": "Segmento sin promesa verificable o sin fuente documental asociada.",
                "is_primary": True,
                "mention_count": 1,
            })

        row["factual_discrepancy"] = discrepancy
        row["index"] = float(discrepancy * 100.0)
        row["families"] = {"factual_rag": row["index"]}
        rows.append(row)
    return rows


def _public_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Versión serializable y acotada del plan para el JSON y el reporte."""
    return [
        {
            "step": step["step"],
            "segment_id": step["segment_id"],
            "start": round(float(step["start"]), 2),
            "claim": step["claim"][:280],
            "query": step["query"][:280],
            "expected_topic": step.get("expected_topic"),
            "status": step["status"],
            "resolution": step.get("resolution"),
            "source": step.get("source"),
            "replanned": step["replanned"],
            "attempts": step["attempts"],
            "score": round(float(step["score"]), 3),
        }
        for step in plan
    ]


def plan_and_execute_agent(state: Dict[str, Any]) -> dict:
    """Nodo que reemplaza al agente factual cuando se usa Plan-and-Execute."""
    started_at = time.time()
    plan = build_plan(state)
    plan = execute_plan(plan, state)
    rows = _build_rows(state, plan)

    resolved = [step for step in plan if step["status"] == "resuelto"]
    replanned = [step for step in plan if step["replanned"]]
    unresolved = [step for step in plan if step["status"] != "resuelto"]
    topics = sorted({step["topic"] for step in resolved})
    topic_counts = Counter(step["topic"] for step in resolved)
    critical_topics = sorted({
        row["topic"] for row in rows if row.get("is_primary") and row["index"] >= 65.0
    })

    metrics = {
        "architecture": "plan-and-execute",
        "planned_steps": len(plan),
        "resolved_steps": len(resolved),
        "strict_resolutions": sum(1 for step in resolved if step.get("resolution") == "keyword_estricto"),
        "replanned_steps": len(replanned),
        "unresolved_steps": len(unresolved),
        "coverage": round(len(resolved) / len(plan), 3) if plan else 0.0,
        "topics_detected": len(topics),
        "critical_topics": len(critical_topics),
        "tokens": 0,          # planner/executor deterministas: no consumen LLM
        "temperature": None,  # se informa None para dejar explícito que no hay muestreo
        "duration_s": round(time.time() - started_at, 3),
        "planner": "rule-based (keywords estrictos + replanificación temática)",
    }

    print(
        f"  → [{AGENT_NAME}] Plan ejecutado: {metrics['planned_steps']} pasos, "
        f"{metrics['resolved_steps']} con evidencia ({metrics['replanned_steps']} por replanificación), "
        f"cobertura {metrics['coverage'] * 100:.0f}%."
    )

    return {
        "factual_rag_metrics": {
            "by_segment": rows,
            "summary": {
                "mean_factual_discrepancy": float(np.mean([row["index"] for row in rows])) / 100.0 if rows else 0.0,
                "evaluated_segments_count": len(rows),
                "critical_segments_count": sum(1 for row in rows if row["index"] >= 65.0),
                "unique_topics_count": len(topics),
                "critical_topics_count": len(critical_topics),
                "repeated_mentions_count": sum(count - 1 for count in topic_counts.values() if count > 1),
            },
            "markers_direction": {"factual_discrepancy": +1},
        },
        "plan": _public_plan(plan),
        "plan_metrics": metrics,
        "logs": [
            f"[{AGENT_NAME}] {metrics['planned_steps']} pasos planificados, "
            f"{metrics['resolved_steps']} resueltos, {metrics['replanned_steps']} replanificados, "
            f"cobertura {metrics['coverage'] * 100:.0f}%."
        ],
    }
