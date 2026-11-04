"""Funciones auxiliares compartidas por los agentes."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np


def safe_mean(values: Iterable[float]) -> float:
    """Calculates a clean mathematical mean skipping uninitialized or NaN entries."""
    clean_values = [x for x in values if x is not None and not math.isnan(x)]
    return float(np.mean(clean_values)) if clean_values else float("nan")


def calculate_factual_deviation_zscore(current_value: float, historical_baseline: Sequence[float]) -> float:
    """
    Measures the standard deviation of a presidential macroeconomic claim 
    against the actual historical baseline verified by MEF/BCRP regulatory data.
    """
    baseline_array = np.array([x for x in historical_baseline if x is not None and not math.isnan(x)], dtype=float)
    if current_value is None or math.isnan(current_value) or baseline_array.size < 2:
        return 0.0
    
    standard_deviation = float(baseline_array.std(ddof=1))
    if standard_deviation < 1e-9:
        return 0.0
        
    return float((current_value - baseline_array.mean()) / standard_deviation)


def calculate_sigmoid(x: float) -> float:
    """Standard activation function used to normalize confidence intervals between 0 and 1."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def normalize_to_opacity_scale(z_score: float) -> float:
    """
    Maps an extraction or budget deviation z-score into a readable 0-100 index.
    z=0 -> 0 (Perfect Alignment); z=3 -> ~84 (Severe Socioeconomic Deviation).
    """
    z_score = max(0.0, z_score)
    return float(100.0 * (1.0 - math.exp(-0.62 * z_score)))


def find_speech_segment_id(timestamp: float, segments: List[Dict]) -> int | None:
    """
    Identifies the unique segment ID for a given timestamp within the speech timeline.
    Sincronizado con claves estrictas en ingles ('start', 'end', 'id').
    """
    for segment in segments:
        if segment["start"] <= timestamp < segment["end"]:
            return segment["id"]
    return None


def round_nested_data(data: any, decimal_places: int = 3) -> any:
    """
    Recursively traverses dictionaries, lists, and matrices to sanitize float numeric precision 
    and format JSON outputs for Streamlit rendering.
    """
    if isinstance(data, dict):
        return {key: round_nested_data(val, decimal_places) for key, val in data.items()}
    if isinstance(data, list):
        return [round_nested_data(val, decimal_places) for val in data]
    if isinstance(data, (float, np.floating)):
        if math.isnan(float(data)):
            return None
        return round(float(data), decimal_places)
    if isinstance(data, np.integer):
        return int(data)
    return data