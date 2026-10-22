"""Configuración por defecto del sistema. Se puede sobrescribir con un JSON (--config)."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIGURATION: Dict[str, Any] = {
    "audio": {
        "sampling_rate": 16000
    },
    "transcription": {
        # Model sizing configuration for faster-whisper (tiny, base, small, medium, large-v3)
        "whisper_model": "base",
        "language": "es",
        "segment_window_seconds": 8.0,
    },
    "sentiment": {
        # Soft-switch for local text classification engines (textblob, pipeline-vader)
        "engine": "textblob",
    },
    "rag_pipeline": {
        # Vector database chunking and retrieval parameters for auditing MEF/BCRP PDFs
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k_documents": 4,
        "vector_database_path": "data_storage/vector_chroma",
    },
    "judge": {
        # Normalized multi-agent evaluation weights for the consolidated consistency index
        "agent_weights": {
            "sentiment": 0.15,
            "linguistic": 0.15,
            "syntactic": 0.20,
            "promise_extraction": 0.20,
            "factual_rag": 0.30,  # Dominancia metodológica: Auditoría regulatoria dura
        },
        # Operational metric boundaries mapped over the structural 0-100 opacity scale
        "truth_threshold": 40.0,
        "deviation_threshold": 65.0,
        # Path binary target location for caching optimized random forest weights
        "trained_model_path": "models/classifier.joblib",
    },
}


def _merge_dictionaries(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges a custom configuration JSON onto the system default matrix structure."""
    for key, value in override_dict.items():
        if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
            _merge_dictionaries(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def cargar_config(custom_path: str | None = None) -> Dict[str, Any]:
    """
    Loads, duplicates, and returns the active execution parameters configuration matrix.
    Mantiene el nombre formal de importacion del proyecto ('cargar_config') para no romper main.py.
    """
    active_config = deepcopy(DEFAULT_CONFIGURATION)
    if custom_path:
        with open(Path(custom_path), encoding="utf-8") as file_stream:
            _merge_dictionaries(active_config, json.load(file_stream))
    return active_config