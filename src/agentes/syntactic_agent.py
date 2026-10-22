"""Agente de Opacidad Sintáctica: mide la complejidad gramatical y el Índice de Niebla (Gunning Fog)."""
from __future__ import annotations

import re
import unicodedata
import numpy as np

AGENT_NAME = "Syntactic"


def _estimate_syllables_es(word: str) -> int:
    """Estimates Spanish syllables based on vowel groups and diphthongs."""
    cleaned = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
    if not cleaned:
        return 0
    # Group vowels
    vowels = re.findall(r"[aeiouáéíóúü]+", cleaned)
    return max(1, len(vowels))


def _calculate_segment_opacity(text: str) -> float:
    """Calculates adapted Gunning-Fog / Fernandez-Huerta syntactic opacity score (0-100)."""
    words = re.findall(r"[a-záéíóúüñ]+", unicodedata.normalize("NFC", text.lower()))
    if not words:
        return 0.0
        
    total_words = len(words)
    complex_words = sum(1 for w in words if _estimate_syllables_es(w) >= 3)
    complex_ratio = (complex_words / total_words) * 100.0
    
    # Fog formula adapted for segment-level oral delivery
    raw_fog = (total_words * 0.5) + (complex_ratio * 0.7)
    # Normalized onto standard opacity scale 0 - 100
    opacity = float(np.clip(raw_fog * 1.8, 12.0, 95.0))
    return opacity


def text_opacity_analyst_agent(state: dict) -> dict:
    """
    Analyzes the syntax complexity, lexical density, and opacity of speech segments.
    """
    segments = state.get("segments", [])
    
    if not segments:
        speech_text = state.get("transcribed_text", "")
        opacity_score = _calculate_segment_opacity(speech_text) if speech_text else 0.0
        return {
            "syntactic_metrics": {
                "by_segment": [],
                "summary": {"mean_opacity": opacity_score},
                "markers_direction": {"opacity_score": +1}
            }
        }

    segments_output = []
    opacity_values = []
    
    for segment in segments:
        text = segment.get("text", "")
        score = _calculate_segment_opacity(text)
        opacity_values.append(score)
        
        segments_output.append({
            "id": segment["id"],
            "opacity_score": score,
            "text": text
        })

    mean_opacity = float(np.mean(opacity_values)) if opacity_values else 0.0
    print(f"  → [{AGENT_NAME} Agent] Opacidad sintáctica media calculada: {mean_opacity:.1f}%")

    return {
        "syntactic_metrics": {
            "by_segment": segments_output,
            "summary": {
                "mean_opacity": mean_opacity,
                "max_opacity": float(max(opacity_values)) if opacity_values else 0.0
            },
            "markers_direction": {"opacity_score": +1}
        },
        "logs": [f"[{AGENT_NAME}] Opacidad media Fog Index: {mean_opacity:.1f}/100"]
    }