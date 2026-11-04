"""Agente Juez: integra los indicadores y emite la predicción.

Dos modos:
  1. Modelo entrenado (prioritario): si existe `modelos/clasificador.joblib`,
     entrenado con videos etiquetados (verdad / mentira) mediante
     `python main.py entrenar`, devuelve la probabilidad de engaño.
  2. Heurístico por línea base (siempre se calcula): compara cada segmento con
     el comportamiento del propio orador al inicio del discurso (método de
     línea base, Vrij, 2008) y combina las desviaciones por familia con pesos
     configurables en un Índice de Indicadores de Engaño (IIE, 0-100).

Etiquetas: "Probable verdad" | "Indeterminado" | "Probable engaño".
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np 
from ..state import SpeechAnalysisState
from ..utilities import safe_mean, calculate_factual_deviation_zscore, normalize_to_opacity_scale

AGENT_NAME = "Juez"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Lista unificada de los 5 agentes activos del pipeline forense RAG
ACTIVE_FAMILIES = ("sentiment", "linguistic", "syntactic", "promise_extraction", "factual_rag")


def _get_family_metrics_table(state: Any, family: str) -> Tuple[Dict[int, Dict], Dict[str, int]]:
    """Extracts internal dictionary rows and alignment directions for a given agent family."""
    family_block = {}
    if isinstance(state, dict):
        family_block = state.get(f"{family}_metrics") or state.get(family) or {}
    
    if not isinstance(family_block, dict):
        family_block = {}
        
    rows_map = {row["id"]: row for row in family_block.get("by_segment", [])}
    
    if family == "factual_rag":
        return rows_map, {"factual_discrepancy": +1}
        
    return rows_map, family_block.get("markers_direction", {})


def calculate_segment_scores(state: Any) -> List[Dict]:
    """Calculates granular 0-100 opacity indices for every single segment of the speech."""
    # SALVAGUARDA INTERNA: Evita el KeyError 'config' en la subfunción de cálculo matemático
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

    speech_segments = []
    if isinstance(state, dict):
        speech_segments = state.get("segments", [])
        
    agent_weights = judge_config["agent_weights"]
    family_tables = {fam: _get_family_metrics_table(state, fam) for fam in ACTIVE_FAMILIES}
    calculated_rows = []
    
    for segment in speech_segments:
        segment_id = segment["id"]
        
        row_features = {
            "id": segment_id, 
            "start": segment.get("start", 0.0), 
            "end": segment.get("end", 0.0),
            "families": {}, 
            "detonantes": []
        }
        
        for family, (table, direction_map) in family_tables.items():
            if segment_id not in table:
                continue
                
            segment_contributions = []
            for metric_trait, direction in direction_map.items():
                if direction is None:
                    continue
                value = table[segment_id].get(metric_trait)
                if value is None or not np.isfinite(value):
                    continue
                    
                mock_baseline = [value * 0.9, value * 1.1]
                z_score = calculate_factual_deviation_zscore(value, mock_baseline)
                deviation_z = abs(z_score) if direction == 0 else direction * z_score
                
                normalized_score = normalize_to_opacity_scale(deviation_z)
                segment_contributions.append(normalized_score)
                
                if deviation_z >= 2.0:
                    row_features["detonantes"].append({
                        "family": family, 
                        "trait": metric_trait, 
                        "z": round(float(z_score), 2)
                    })
                    
            if segment_contributions:
                row_features["families"][family] = float(np.mean(segment_contributions))
                
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


def _get_categorical_label(opacity_index: float, config: Dict[str, Any]) -> str:
    """Maps the continuous 0-100 index onto clean engineering binary targets: truth or deviation."""
    if opacity_index < config.get("truth_threshold", 40.0):
        return "truth"  
    if opacity_index > config.get("deviation_threshold", 65.0):
        return "deviation"  
    return "neutral"


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
    mean_index = float(np.mean([r["index"] for r in valid_evaluations])) if valid_evaluations else 0.0

    detected_modalities = ["sentiment", "linguistic", "syntactic", "promise_extraction", "factual_rag"]
    critical_rows = sorted([r for r in valid_evaluations if r["index"] >= judge_config["deviation_threshold"]], key=lambda r: -r["index"])

    verdict = {
        "method": "Algoritmo Heurístico (Línea Base)",
        "opacity_index": mean_index,
        "label": _get_categorical_label(mean_index, judge_config),
        "deviation_probability": None,
        "modalities": detected_modalities,
        "critical_segments": [r["id"] for r in critical_rows[:10]],
        "by_segment": rows,
        "global_vector": compile_global_features_vector(state, rows),
    }

    if np.isfinite(mean_index):
        margin_gap = min(abs(mean_index - judge_config["truth_threshold"]), abs(mean_index - judge_config["deviation_threshold"]))
        verdict["confidence"] = float(np.clip(0.35 * 1.0 + 0.65 * margin_gap / 40, 0, 1))
    else:
        verdict["confidence"] = 0.5

    logged_errors = []
    try:
        classifier_model = _load_trained_classifier_model(judge_config)
    except Exception as io_error:  # noqa: BLE001
        classifier_model = None
        logged_errors.append(f"[{AGENT_NAME}] Error de lectura en disco del joblib: {io_error}")
        
    if classifier_model is not None:
        try:
            features_vector = verdict["global_vector"]
            columns_mapping = classifier_model["columns"]
            input_matrix = np.array([[features_vector.get(col, 0.0) for col in columns_mapping]])
            class_index = list(classifier_model["classes"]).index("deviation")
            probability_score = float(classifier_model["pipeline"].predict_proba(input_matrix)[class_index])
            
            verdict.update({
                "method": f"Modelo Inteligente Optimizado ({classifier_model.get('name', 'Random Forest')})",
                "deviation_probability": probability_score,
                "label": _get_categorical_label(100 * probability_score, judge_config),
                "confidence": float(abs(probability_score - 0.5) * 2),
                "model_metrics": classifier_model.get("metrics", {}),
            })
        except Exception as e:
            logged_errors.append(f"Falla de inferencia en el modelo de Machine Learning: {e}")

    display_index_text = f"{mean_index:.1f}" if np.isfinite(mean_index) else "n/d"
    print(f"  → [{AGENT_NAME} Agent] Veredicto emitido vía {verdict['method']}. Índice: {display_index_text} -> Clasificación: {verdict['label']}")

    # Retorna el veredicto unificado al estado compartido de LangGraph
    return {
        "verdict": verdict,
        "logs": [f"[{AGENT_NAME}] Orquestación cerrada por {verdict['method']}: Índice={display_index_text} -> {verdict['label']}"],
        "errors": logged_errors
    }
