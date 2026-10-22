"""Agente Lingüístico: marcadores verbales asociados en la literatura al engaño.

Basado en el enfoque de conteo de categorías tipo LIWC (Newman, Pennebaker,
Berry y Richards, 2003; Hauch et al., 2015), adaptado al español:

  - primera_persona   : yo, me, mi... (los mentirosos tienden a distanciarse: MENOS)
  - exclusivas        : pero, excepto, sin embargo... (complejidad cognitiva: MENOS)
  - negaciones        : no, nunca, jamás... (MÁS)
  - atenuadores       : quizás, tal vez, creo que... (evasión: MÁS)
  - enfaticos_verdad  : sinceramente, les aseguro, créanme... (sobre-afirmación: MÁS)
  - emocion_negativa  : vocabulario negativo (MÁS)
  - palabras_por_seg  : ritmo verbal (cambios respecto a la línea base)

Todas las tasas se expresan por cada 100 palabras del segmento.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Any
import numpy as np

from ..state import SpeechAnalysisState

AGENT_NAME = "Linguistic"

# Vocabulario negativo local unificado para evitar dependencias rotas con sentiment_agent
NEGATIVE_VOCABULARY = (
    "mal malo mala malos peor peores crisis problema problemas corrupción corrupto pobreza "
    "miedo temor amenaza violencia delito delincuencia crimen inseguridad muerte muertos fracaso "
    "fracasar deuda desempleo hambre injusticia injusto mentira mentiras culpa culpable odio "
    "conflicto guerra caos ataque atacar destruir destrucción daño dañar triste tristeza dolor "
    "grave peligro riesgo abandono abandonado negativo negativa error errores falso escándalo "
    "robo robar pérdida pérdidas inflación recesión enemigo"
).split()

CATEGORIES: Dict[str, List[str]] = {
    "first_person": ["yo", "me", "mi", "mis", "mío", "mía", "míos", "mías", "conmigo"],
    "exclusive_words": ["pero", "excepto", "salvo", "aunque", "sino", "sin embargo",
                        "no obstante", "a pesar", "mientras que", "en cambio"],
    "negations": ["no", "nunca", "jamás", "jamas", "tampoco", "ni", "nadie", "nada", "ninguno",
                  "ninguna"],
    "hedges": ["quizás", "quizá", "tal vez", "posiblemente", "probablemente", "creo que",
               "me parece", "de alguna manera", "más o menos", "en cierto modo",
               "podría", "podríamos", "supuestamente", "aparentemente", "digamos",
               "por así decirlo", "en principio"],
    "truth_assertives": ["sinceramente", "honestamente", "francamente", "le juro", "les juro",
                         "créanme", "créame", "les aseguro", "le aseguro", "de verdad",
                         "en verdad", "para ser honesto", "para ser sincero",
                         "con toda sinceridad", "absolutamente", "definitivamente",
                         "categóricamente", "les digo la verdad", "la verdad es que"],
    "negative_emotion": NEGATIVE_VOCABULARY,
}

# Expected deviation markers direction used by the judge node consolidation matrix
MARKERS_DIRECTION = {
    "first_person": -1, "exclusive_words": -1, "negations": +1, "hedges": +1,
    "truth_assertives": +1, "negative_emotion": +1, "words_per_second": 0
}


def _normalize_text(input_text: str) -> str:
    """Normalizes characters to unicode standards and formats lowercase spacing boundaries."""
    return " " + " ".join(re.findall(r"[a-záéíóúüñ]+", unicodedata.normalize("NFC", input_text.lower()))) + " "


def _count_token_occurrences(normalized_text: str, token_terms: List[str]) -> int:
    """Counts absolute matches of bounded semantic categories inside the normalized array."""
    return sum(normalized_text.count(f" {term} ") for term in token_terms)


def analyze_text_markers(raw_text: str, duration_seconds: float) -> Dict[str, float]:
    """Calculates standardized occurrence percentages per 100 words within a speech segment."""
    normalized_text = _normalize_text(raw_text)
    total_tokens = max(1, len(normalized_text.split()))
    
    features = {category: 100.0 * _count_token_occurrences(normalized_text, terms_list) / total_tokens 
                for category, terms_list in CATEGORIES.items()}
                
    features["total_words"] = float(len(normalized_text.split()))
    features["words_per_second"] = len(normalized_text.split()) / duration_seconds if duration_seconds > 0 else 0.0
    return features


def linguistic_agent(state: SpeechAnalysisState) -> dict:
    """
    Main linguistic agent tracking text complexity benchmarks, evasive hedges, 
    and excessive truth-assertive patterns across the segment pipeline.
    """
    # Sincronizado en ingles 'segments'
    segments_list = state.get("segments", [])
    
    # Sistema de degradación controlada: Si no hay texto, se bypasséa el nodo de forma segura
    if not segments_list or not any(s.get("text") for s in segments_list):
        return {
            "linguistic_metrics": {"by_segment": [], "summary": {}},
            "logs": [f"[{AGENT_NAME}] Ejecución omitida: No se detectaron segmentos de texto."]
        }

    segments_output = []
    for segment in segments_list:
        segment_text = segment.get("text", "")
        
        # Corrección crítica: Sincronizado en ingles 'end' y 'start' de Whisper
        segment_duration = max(0.1, float(segment["end"] - segment["start"]))
        
        features = analyze_text_markers(segment_text, segment_duration)
        features["id"] = segment["id"]
        
        # Extracción e identificación de frases específicas para el reporte forense final
        normalized_text = _normalize_text(segment_text)
        features["detected_hedges"] = [t for t in CATEGORIES["hedges"] if f" {t} " in normalized_text]
        features["detected_assertives"] = [t for t in CATEGORIES["truth_assertives"] if f" {t} " in normalized_text]
        
        segments_output.append(features)

    # Compilación estadística consolidada de KPIs lingüísticos del discurso
    summary_metrics = {category: float(np.mean([p[category] for p in segments_output]))
                       for category in list(CATEGORIES) + ["words_per_second"]}
                       
    summary_metrics["total_speech_words"] = int(sum(p["total_words"] for p in segments_output))

    print(f"  → [{AGENT_NAME} Agent] Conteo completado. Palabras totales auditadas: {summary_metrics['total_speech_words']}")

    # Retorna las métricas estructuradas al Grafo de Estado de LangGraph
    return {
        "linguistic_metrics": {
            "by_segment": segments_output,
            "summary": summary_metrics,
            "markers_direction": MARKERS_DIRECTION
        },
        "logs": [f"[{AGENT_NAME}] {summary_metrics['total_speech_words']} palabras analizadas en categorías LIWC."]
    }