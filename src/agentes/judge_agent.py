"""Agente Juez: integra los indicadores de todos los agentes y emite la predicción y auditoría consolidada.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np 
from ..state import SpeechAnalysisState
from ..utilities import safe_mean, normalize_to_opacity_scale

AGENT_NAME = "Juez"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_FAMILIES = ("sentiment", "linguistic", "syntactic", "promise_extraction", "factual_rag")


def _get_categorical_label(opacity_index: float, config: Dict[str, Any]) -> str:
    """Maps the continuous 0-100 index onto clean engineering binary targets: truth, neutral or deviation."""
    if opacity_index < config.get("truth_threshold", 40.0):
        return "truth"  
    if opacity_index > config.get("deviation_threshold", 65.0):
        return "deviation"  
    return "neutral"


def calculate_segment_scores(state: Any) -> List[Dict]:
    """Calculates granular 0-100 opacity indices for every single segment of the speech."""
    try:
        if isinstance(state, dict) and "config" in state:
            judge_config = state["config"]["judge"]
        else:
            judge_config = {
                "agent_weights": {"sentiment": 0.15, "linguistic": 0.15, "syntactic": 0.20, "promise_extraction": 0.20, "factual_rag": 0.30},
                "truth_threshold": 40.0,
                "deviation_threshold": 65.0,
                "trained_model_path": "models/classifier.joblib"
            }
    except Exception:
        judge_config = {
            "agent_weights": {"sentiment": 0.15, "linguistic": 0.15, "syntactic": 0.20, "promise_extraction": 0.20, "factual_rag": 0.30},
            "truth_threshold": 40.0,
            "deviation_threshold": 65.0,
            "trained_model_path": "models/classifier.joblib"
        }

    speech_segments = state.get("segments", []) if isinstance(state, dict) else []
    agent_weights = judge_config.get("agent_weights", {
        "sentiment": 0.15, "linguistic": 0.15, "syntactic": 0.20, "promise_extraction": 0.20, "factual_rag": 0.30
    })

    sentiment_map = {r["id"]: r for r in state.get("sentiment_metrics", {}).get("by_segment", [])}
    linguistic_map = {r["id"]: r for r in state.get("linguistic_metrics", {}).get("by_segment", [])}
    syntactic_map = {r["id"]: r for r in state.get("syntactic_metrics", {}).get("by_segment", [])}
    promise_map = {r["id"]: r for r in state.get("promise_extraction_metrics", {}).get("by_segment", [])}
    factual_map = {r["id"]: r for r in state.get("factual_rag_metrics", {}).get("by_segment", [])}

    calculated_rows = []

    for segment in speech_segments:
        seg_id = segment["id"]
        row_features = {
            "id": seg_id,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "text": segment.get("text", ""),
            "families": {},
            "detonantes": []
        }

        # 1. Auditoría Factual RAG (MEF / BCRP) - 30% Dominancia
        fact_item = factual_map.get(seg_id, {})
        factual_score = float(fact_item.get("index", fact_item.get("factual_discrepancy", 0.15) * 100.0))
        row_features["families"]["factual_rag"] = factual_score
        # Metadatos del hallazgo factual: permiten deduplicar por tema en el veredicto
        row_features["topic"] = fact_item.get("topic")
        row_features["topic_key"] = fact_item.get("topic_key")
        row_features["factual_primary"] = fact_item.get("is_primary", True)
        row_features["factual_mentions"] = fact_item.get("mention_count", 1)
        row_features["audit_note"] = fact_item.get("audit_note")
        row_features["factual_source"] = fact_item.get("source")
        if factual_score >= judge_config.get("deviation_threshold", 65.0):
            row_features["detonantes"].append({
                "family": "factual_rag",
                "trait": "Discrepancia regulatoria frente a reportes del MEF/BCRP",
                "z": round((factual_score - 40.0) / 15.0, 2),
                "note": fact_item.get("audit_note", "Discrepancia crítica encontrada por el RAG.")
            })

        # 2. Opacidad Sintáctica (Fog Index) - 20%
        syn_item = syntactic_map.get(seg_id, {})
        syn_score = float(syn_item.get("opacity_score", 45.0))
        row_features["families"]["syntactic"] = syn_score
        if syn_score >= 70.0:
            row_features["detonantes"].append({
                "family": "syntactic",
                "trait": "Complejidad léxica / Opacidad gramatical",
                "z": round((syn_score - 50.0) / 12.0, 2)
            })

        # 3. Extracción de Promesas y Metas - 20%
        prom_item = promise_map.get(seg_id, {})
        prom_score = float(prom_item.get("promise_score", 0.0))
        row_features["families"]["promise_extraction"] = prom_score
        if prom_score >= 60.0:
            row_features["detonantes"].append({
                "family": "promise_extraction",
                "trait": "Alta concentración de compromisos de Estado",
                "z": round(prom_score / 30.0, 2)
            })

        # 4. Densidad Lingüística (Marcadores LIWC) - 15%
        ling_item = linguistic_map.get(seg_id, {})
        hedges = ling_item.get("hedges", 0.0)
        assertives = ling_item.get("truth_assertives", 0.0)
        neg_emo = ling_item.get("negative_emotion", 0.0)
        ling_score = float(np.clip(20.0 + (hedges * 7.5) + (assertives * 8.5) + (neg_emo * 5.0), 10.0, 95.0))
        row_features["families"]["linguistic"] = ling_score
        if hedges > 3.0 or assertives > 3.0:
            row_features["detonantes"].append({
                "family": "linguistic",
                "trait": "Uso excesivo de sobre-afirmaciones de sinceridad o atenuadores",
                "z": round(max(hedges, assertives), 2)
            })

        # 5. Sentimiento Político - 15%
        sent_item = sentiment_map.get(seg_id, {})
        sent_score = float(sent_item.get("sentiment_risk", 25.0))
        row_features["families"]["sentiment"] = sent_score

        # Índice ponderado consolidado del segmento
        available_families = {f: v for f, v in row_features["families"].items() if agent_weights.get(f, 0) > 0}
        if available_families:
            total_weight = sum(agent_weights[f] for f in available_families)
            row_features["index"] = float(sum(agent_weights[f] * v for f, v in available_families.items()) / total_weight)
        else:
            row_features["index"] = float("nan")

        calculated_rows.append(row_features)

    return calculated_rows


def compile_global_features_vector(state: Any, rows: List[Dict] | None = None) -> Dict[str, float]:
    """Compiles the full feature matrix of the speech file to feed the trained Random Forest classifier."""
    rows = rows if rows is not None else calculate_segment_scores(state)
    valid_rows = [r for r in rows if np.isfinite(r["index"])] or rows
    
    global_vector: Dict[str, float] = {}
    
    for family in ACTIVE_FAMILIES:
        family_values = np.array([r["families"].get(family, np.nan) for r in valid_rows], dtype=float)
        clean_values = family_values[np.isfinite(family_values)]
        global_vector[f"{family}_mean"] = float(clean_values.mean()) if clean_values.size else 0.0
        global_vector[f"{family}_p90"] = float(np.percentile(clean_values, 90)) if clean_values.size else 0.0
        
    for family in ACTIVE_FAMILIES:
        metrics_block = {}
        if isinstance(state, dict):
            metrics_block = state.get(f"{family}_metrics") or state.get(family) or {}
        if isinstance(metrics_block, dict):
            for key, val in metrics_block.get("summary", {}).items():
                if isinstance(val, (int, float)):
                    global_vector[f"{family}__{key}"] = float(val) if np.isfinite(val) else 0.0
                    
    index_array = np.array([r["index"] for r in valid_rows], dtype=float)
    global_vector["mean_opacity_index"] = float(np.nanmean(index_array)) if np.isfinite(index_array).any() else 0.0
    return global_vector


def _load_trained_classifier_model(config: Dict[str, Any]):
    """Safely handles file system IO to un-pickle the serialized joblib pipeline."""
    model_path_str = config.get("trained_model_path", "models/classifier.joblib")
    model_path = Path(model_path_str)
    full_path = model_path if model_path.is_absolute() else REPOSITORY_ROOT / model_path
    if not full_path.exists():
        return None
    import joblib  # type: ignore
    return joblib.load(full_path)


def consolidation_judge_agent(state: Any) -> dict:
    """
    Main function entry point registered in English in the LangGraph orchestration graph.
    Executes algorithmic voting arbitration or calls the trained machine learning backend.
    """
    try:
        if isinstance(state, dict) and "config" in state:
            judge_config = state["config"]["judge"]
        else:
            judge_config = {
                "agent_weights": {"sentiment": 0.15, "linguistic": 0.15, "syntactic": 0.20, "promise_extraction": 0.20, "factual_rag": 0.30},
                "truth_threshold": 40.0,
                "deviation_threshold": 65.0,
                "trained_model_path": "models/classifier.joblib"
            }
    except Exception:
        judge_config = {
            "agent_weights": {"sentiment": 0.15, "linguistic": 0.15, "syntactic": 0.20, "promise_extraction": 0.20, "factual_rag": 0.30},
            "truth_threshold": 40.0,
            "deviation_threshold": 65.0,
            "trained_model_path": "models/classifier.joblib"
        }

    rows = calculate_segment_scores(state)
    valid_evaluations = [r for r in rows if np.isfinite(r["index"])]
    base_mean = float(np.mean([r["index"] for r in valid_evaluations])) if valid_evaluations else 0.0

    detected_modalities = ["sentiment", "linguistic", "syntactic", "promise_extraction", "factual_rag"]
    deviation_threshold = judge_config.get("deviation_threshold", 65.0)

    # Segmentos con discrepancia factual crítica: SOLO la primera mención de cada tema
    # regulatorio (una fila por tema, sin repetir la misma promesa en 9 segmentos).
    critical_findings: List[Dict[str, Any]] = []
    seen_topics: set[str] = set()
    factual_candidates = sorted(
        [
            row for row in valid_evaluations
            if row.get("families", {}).get("factual_rag", 0.0) >= deviation_threshold
            and row.get("factual_primary", True)
        ],
        key=lambda row: -row.get("families", {}).get("factual_rag", 0.0),
    )
    for row in factual_candidates:
        topic_key = row.get("topic_key") or str(row.get("topic") or row["id"])
        if topic_key in seen_topics:
            continue
        seen_topics.add(topic_key)
        critical_findings.append({
            "segment_id": row["id"],
            "topic": row.get("topic") or "Discrepancia factual crítica",
            "topic_key": topic_key,
            "audit_note": row.get("audit_note") or "",
            "source": row.get("factual_source") or "",
            "factual_score": round(float(row.get("families", {}).get("factual_rag", 0.0)), 1),
            "index": round(float(row.get("index", 0.0)), 1),
            "start": float(row.get("start", 0.0)),
            "end": float(row.get("end", 0.0)),
            "mentions": int(row.get("factual_mentions", 1) or 1),
        })

    factual_critical = [finding["factual_score"] for finding in critical_findings]
    if factual_critical:
        factual_critical_mean = float(np.mean(factual_critical))
        # Cobertura: proporción del discurso que toca algún tema crítico, contando también
        # sus repeticiones (así una promesa repetida pesa por su presencia real, no 9 veces).
        critical_topic_keys = {finding["topic_key"] for finding in critical_findings}
        covered_segments = [
            row for row in valid_evaluations
            if row.get("topic_key") in critical_topic_keys
        ]
        coverage_ratio = len(covered_segments) / max(1, len(valid_evaluations))
        weight_critical = min(0.50, coverage_ratio * 1.2)
        mean_index = round(base_mean * (1.0 - weight_critical) + factual_critical_mean * weight_critical, 1)
    else:
        mean_index = round(base_mean, 1)

    verdict = {
        "method": "Algoritmo Heurístico Agéntico Multi-Modal",
        "opacity_index": mean_index,
        "label": _get_categorical_label(mean_index, judge_config),
        "deviation_probability": None,
        "modalities": detected_modalities,
        "critical_segments": [finding["segment_id"] for finding in critical_findings],
        "critical_findings": critical_findings,
        "by_segment": rows,
        "global_vector": compile_global_features_vector(state, rows),
    }

    if np.isfinite(mean_index):
        verdict["confidence"] = float(np.clip(0.65 + 0.25 * (len(critical_findings) / 5.0), 0.65, 0.92))
    else:
        verdict["confidence"] = 0.50

    # Salvaguarda: sin transcripción no hay contenido que auditar. Evita que el
    # sistema reporte "truth" cuando en realidad falló Whisper y no hay texto.
    speech_segments = state.get("segments", []) if isinstance(state, dict) else []
    has_transcript = any(str(segment.get("text", "")).strip() for segment in speech_segments)

    logged_errors: List[str] = []
    if not has_transcript:
        verdict.update({
            "label": "neutral",
            "method": "Sin transcripción: contenido no auditable",
            "opacity_index": 0.0,
            "confidence": 0.0,
            "deviation_probability": None,
            "transcription_missing": True,
        })
        logged_errors.append(
            f"[{AGENT_NAME}] No se obtuvo texto del audio: el dictamen queda como no concluyente."
        )

    try:
        classifier_model = _load_trained_classifier_model(judge_config)
    except Exception as io_error:  # noqa: BLE001
        classifier_model = None
        logged_errors.append(f"[{AGENT_NAME}] Error de lectura en disco del joblib: {io_error}")
        
    if classifier_model is not None and has_transcript:
        try:
            features_vector = verdict["global_vector"]
            columns_mapping = classifier_model["columns"]
            input_matrix = np.array([[features_vector.get(col, 0.0) for col in columns_mapping]])
            class_index = list(classifier_model["classes"]).index("deviation")
            probability_score = float(classifier_model["pipeline"].predict_proba(input_matrix)[0][class_index])
            model_confidence = float(abs(probability_score - 0.5) * 2)
            
            # Si el modelo tiene frontera discriminativa entrenada
            if model_confidence >= 0.15:
                verdict.update({
                    "method": f"Modelo Inteligente Optimizado ({classifier_model.get('name', 'Random Forest')})",
                    "deviation_probability": probability_score,
                    "label": _get_categorical_label(100 * probability_score, judge_config),
                    "confidence": model_confidence,
                    "model_metrics": classifier_model.get("metrics", {}),
                })
            else:
                # El modelo está indeterminado; se preserva el cálculo multiagente consolidado
                verdict.update({
                    "method": "Algoritmo Heurístico Agéntico Multi-Modal",
                    "deviation_probability": mean_index / 100.0,
                    "label": _get_categorical_label(mean_index, judge_config),
                    "confidence": verdict["confidence"],
                    "model_metrics": classifier_model.get("metrics", {}),
                })
        except Exception as e:
            logged_errors.append(f"Falla de inferencia en el modelo de Machine Learning: {e}")

    display_index_text = f"{verdict['opacity_index']:.1f}" if np.isfinite(verdict["opacity_index"]) else "n/d"
    if verdict.get("transcription_missing"):
        print(f"  → [{AGENT_NAME} Agent] Sin texto transcrito: dictamen no concluyente (neutral).")
    else:
        print(f"  → [{AGENT_NAME} Agent] Veredicto emitido vía {verdict['method']}. Índice: {display_index_text} -> Clasificación: {verdict['label']} (Críticos: {len(verdict['critical_segments'])})")

    return {
        "verdict": verdict,
        "logs": [f"[{AGENT_NAME}] Orquestación cerrada por {verdict['method']}: Índice={display_index_text} -> {verdict['label']}"],
        "errors": logged_errors
    }
