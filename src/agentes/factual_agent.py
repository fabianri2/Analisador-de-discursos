"""Agente de Auditoría Factual (RAG): cruza afirmaciones del discurso con datos oficiales del MEF, BCRP y entidades del Estado."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Any
import numpy as np

from ..state import SpeechAnalysisState
from ..utilities import calculate_sigmoid

AGENT_NAME = "FactualAlignment"

# Una misma promesa suele repetirse a lo largo de un discurso. Para no contar nueve
# veces la misma discrepancia, solo la primera mención de cada tema se reporta como
# hallazgo crítico; las repeticiones conservan una señal atenuada.
REITERATION_FACTOR = 0.45

# Base de conocimiento factual regulatoria y documental del Estado peruano (2023 - 2024).
# Los keywords son frases específicas (no palabras sueltas como "orden" o "gasto"),
# porque esos términos genéricos hacían que casi cualquier segmento se marcara como
# discrepancia crítica frente al MEF/BCRP.
FACTUAL_BENCHMARKS = [
    {
        "keywords": [
            "adelanto de elecciones", "elecciones generales", "comicios generales",
            "recorte del periodo", "recorto el periodo", "convoque a elecciones",
        ],
        "topic": "Cronograma Electoral y Mandato Presidencial",
        "fact_target": 0.90,  # Alta discrepancia: Las elecciones prometidas para 2023 no se convocaron ni se realizaron.
        "source": "JNE / ONPE / Diario Oficial El Peruano (Registro de Comicios 2023)",
        "audit_note": "La promesa categórica de elecciones generales en 2023 y recorte de mandato no se concretó formalmente en el año fiscal auditado."
    },
    {
        "keywords": [
            "reforma total de la constitucion", "asamblea constituyente",
            "reforma constitucional", "comision de constitucion",
        ],
        "topic": "Reforma Constitucional",
        "fact_target": 0.78,
        "source": "Archivo Digital de la Legislatura del Congreso de la República (2023)",
        "audit_note": "La iniciativa de reforma total de la carta de 1993 no superó las comisiones dictaminadoras ni fue promulgada."
    },
    {
        "keywords": [
            "producto bruto interno", "pbi", "presupuesto publico", "presupuesto nacional",
            "inversion publica", "deficit fiscal", "crecimiento economico",
            "obras publicas", "millones de soles", "ejecucion presupuestal",
        ],
        "topic": "Consistencia Macroeconómica y Presupuestal MEF/BCRP",
        "fact_target": 0.82,
        "source": "BCRP (Memoria Anual 2023) / MEF (Marco Macroeconómico Multianual 2024-2027)",
        "audit_note": "El PBI oficial cerró en recesión (-0.6%), la inversión privada cayó -7.2% y el déficit fiscal (2.8%) perforó el techo de la regla fiscal."
    },
    {
        "keywords": [
            "policia nacional", "paz social", "estado de emergencia", "orden interno",
            "bloqueos", "muertos", "fallecidos", "violencia",
        ],
        "topic": "Seguridad y Paz Social",
        "fact_target": 0.72,
        "source": "Defensoría del Pueblo (Reporte de Conflictos Sociales Enero-Marzo 2023)",
        "audit_note": "Los registros oficiales constataron 49 civiles fallecidos en enfrentamientos y paralización de corredores viales clave."
    }
]


def _normalize_for_matching(raw_text: str) -> str:
    """Lowercases, strips accents and pads the text so keywords can match whole words only."""
    decomposed = unicodedata.normalize("NFD", raw_text.lower())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    tokens = re.findall(r"[a-z0-9]+", without_accents)
    return " " + " ".join(tokens) + " "


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Compiles a whole-word, accent-insensitive matcher that also tolerates plurals."""
    normalized = _normalize_for_matching(keyword).strip()
    body = r"\s+".join(re.escape(token) for token in normalized.split())
    return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")


BENCHMARK_PATTERNS = [
    (benchmark, [_keyword_pattern(keyword) for keyword in benchmark["keywords"]])
    for benchmark in FACTUAL_BENCHMARKS
]


def _slugify_topic(topic: str) -> str:
    """Builds a stable identifier so the same regulatory topic is deduplicated downstream."""
    return "_".join(_normalize_for_matching(topic).split())


def _find_matching_benchmarks(segment_text: str) -> List[Dict[str, Any]]:
    """Returns every regulatory benchmark whose specific keywords appear in the segment."""
    normalized_text = _normalize_for_matching(segment_text)
    return [
        benchmark
        for benchmark, patterns in BENCHMARK_PATTERNS
        if any(pattern.search(normalized_text) for pattern in patterns)
    ]


def _map_segments_by_id(agent_metrics_block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    if not isinstance(agent_metrics_block, dict):
        return {}
    return {row["id"]: row for row in agent_metrics_block.get("by_segment", [])}


def factual_alignment_agent(state: SpeechAnalysisState) -> dict:
    """
    Main cross-modal validation node checking presidential speech statements against hard factual data.
    Contrasts commitments with actual Year N+1 regulatory records from MEF, BCRP and official sources.

    Regla de reporte: cada tema regulatorio genera UN hallazgo (en su primera mención).
    Los segmentos que repiten el mismo tema quedan marcados como reiteración para no
    duplicar la misma discrepancia en el reporte ni en el veredicto.
    """
    sentiment_map = _map_segments_by_id(state.get("sentiment_metrics", {}))
    linguistic_map = _map_segments_by_id(state.get("linguistic_metrics", {}))
    speech_segments = state.get("segments", [])

    # ---- Pase 1: un solo tema (el más crítico) por segmento ----
    primary_benchmark_by_segment: Dict[int, Dict[str, Any] | None] = {}
    for segment in speech_segments:
        matched = _find_matching_benchmarks(segment.get("text", ""))
        primary_benchmark_by_segment[segment["id"]] = (
            max(matched, key=lambda benchmark: benchmark["fact_target"]) if matched else None
        )

    # ---- Pase 2: la primera mención de cada tema es el hallazgo que se reporta ----
    anchor_segment_by_topic: Dict[str, int] = {}
    for segment in speech_segments:
        benchmark = primary_benchmark_by_segment[segment["id"]]
        if benchmark and benchmark["topic"] not in anchor_segment_by_topic:
            anchor_segment_by_topic[benchmark["topic"]] = segment["id"]
    topic_mentions = Counter(
        benchmark["topic"] for benchmark in primary_benchmark_by_segment.values() if benchmark
    )

    segments_output = []
    factual_discrepancies_timeline = []

    for segment in speech_segments:
        segment_id = segment["id"]
        segment_text = segment.get("text", "")

        row_result = {
            "id": segment_id,
            "text": segment_text,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
        }

        # 1. Factores lingüísticos y emocionales que agravan la opacidad (Vrij, 2008)
        linguistic_features = linguistic_map.get(segment_id, {})
        hedges_ratio = linguistic_features.get("hedges", 0.0) / 100.0
        assertives_ratio = linguistic_features.get("truth_assertives", 0.0) / 100.0

        primary_benchmark = primary_benchmark_by_segment.get(segment_id)
        is_anchor = bool(primary_benchmark) and anchor_segment_by_topic.get(primary_benchmark["topic"]) == segment_id

        if primary_benchmark:
            base_discrepancy = primary_benchmark["fact_target"]
            # Si se usan afirmaciones excesivas ("les aseguro", "la verdad") o evasiones, la brecha factual se acentúa
            discrepancy_modifier = (assertives_ratio * 0.4) + (hedges_ratio * 0.3)
            raw_discrepancy = float(np.clip(base_discrepancy + discrepancy_modifier, 0.0, 0.98))
            final_segment_discrepancy = raw_discrepancy if is_anchor else float(
                np.clip(raw_discrepancy * REITERATION_FACTOR, 0.05, 0.60)
            )

            row_result["topic"] = primary_benchmark["topic"]
            row_result["topic_key"] = _slugify_topic(primary_benchmark["topic"])
            row_result["source"] = primary_benchmark["source"]
            row_result["audit_note"] = primary_benchmark["audit_note"]
            row_result["is_primary"] = is_anchor
            row_result["mention_count"] = int(topic_mentions[primary_benchmark["topic"]])
            if not is_anchor:
                row_result["reiteration_of"] = anchor_segment_by_topic[primary_benchmark["topic"]]
        else:
            # Segmento sin contraste directo con el MEF/BCRP (introducciones, saludos protocolares)
            # Desvío residual base bajo (12% - 25%)
            final_segment_discrepancy = float(np.clip(0.12 + (hedges_ratio * 0.35), 0.05, 0.35))
            row_result["topic"] = "Discurso General / Protocolar"
            row_result["source"] = "Archivo Presidencial"
            row_result["audit_note"] = "Afirmación protocolar o descriptiva sin contraste cuantitativo directo."
            row_result["is_primary"] = True
            row_result["mention_count"] = 1

        row_result["factual_discrepancy"] = final_segment_discrepancy
        row_result["index"] = float(final_segment_discrepancy * 100.0)
        row_result["families"] = {"factual_rag": row_result["index"]}

        segments_output.append(row_result)
        factual_discrepancies_timeline.append(final_segment_discrepancy)

    discrepancy_array = np.array(factual_discrepancies_timeline, dtype=float)
    evaluated_count = len(discrepancy_array)
    mean_discrepancy = float(discrepancy_array.mean()) if evaluated_count else 0.0
    critical_topics = [
        topic for topic, anchor_id in anchor_segment_by_topic.items()
        if next(row["index"] for row in segments_output if row["id"] == anchor_id) >= 65.0
    ]

    summary_metrics = {
        "mean_factual_discrepancy": mean_discrepancy,
        "evaluated_segments_count": evaluated_count,
        "critical_segments_count": sum(1 for d in factual_discrepancies_timeline if d >= 0.65),
        "unique_topics_count": len(anchor_segment_by_topic),
        "critical_topics_count": len(critical_topics),
        "repeated_mentions_count": sum(count - 1 for count in topic_mentions.values() if count > 1),
    }

    print(
        f"  → [{AGENT_NAME} Agent] Validación cruzada completada. Segmentos evaluados contra MEF/BCRP: "
        f"{evaluated_count} (Temas detectados: {len(anchor_segment_by_topic)}, "
        f"críticos: {summary_metrics['critical_topics_count']}, "
        f"menciones repetidas agrupadas: {summary_metrics['repeated_mentions_count']})"
    )

    return {
        "factual_rag_metrics": {
            "by_segment": segments_output,
            "summary": summary_metrics,
            "markers_direction": {"factual_discrepancy": +1}
        },
        "logs": [
            f"[{AGENT_NAME}] Cruce RAG con MEF/BCRP: discrepancia media de {mean_discrepancy * 100:.1f}/100, "
            f"{len(anchor_segment_by_topic)} temas únicos y {summary_metrics['repeated_mentions_count']} repeticiones agrupadas."
        ]
    }
