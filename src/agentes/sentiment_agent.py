"""Agente de Sentimiento: polaridad y emociones del texto de cada segmento en español.
"""
from __future__ import annotations

import re
import unicodedata
import numpy as np
from textblob import TextBlob
from ..state import SpeechAnalysisState

AGENT_NAME = "Sentiment"

SPANISH_POSITIVE = set(
    "paz tranquilidad progreso desarrollo solucion soluciones estabilidad confianza esperanza "
    "union justicia dialogo libertad democracia victoria exito bienestar transparente verdad "
    "trabajo consensos consenso acuerdo acuerdos seguridad legitimo legitimidad mejorar "
    "salud educacion crecimiento alivio recuperacion defender proteger proteccion querido queridos".split()
)

SPANISH_NEGATIVE = set(
    "violencia caos crisis conflicto muerte muertos destruccion dano odio culpa culpable "
    "amenaza peligro riesgo terrorismo vandalismo saqueo bloqueo heridos enfrentamiento "
    "recesion inflacion corrupcion division vacio inaceptable indignacion agresion "
    "agredida agredidas ilegal delincuencia crimen traicion mentira mentiras estéril esteril".split()
)


def _score_spanish_sentiment(text: str) -> float:
    normalized = " ".join(re.findall(r"[a-záéíóúüñ]+", unicodedata.normalize("NFD", text.lower())))
    # Remove accents for flexible matching
    normalized_plain = re.sub(r"[\u0300-\u036f]", "", normalized)
    words = normalized_plain.split()
    if not words:
        return 0.0
        
    pos_count = sum(1 for w in words if w in SPANISH_POSITIVE)
    neg_count = sum(1 for w in words if w in SPANISH_NEGATIVE)
    
    tb_polarity = float(TextBlob(text).sentiment.polarity)
    if pos_count + neg_count > 0:
        lex_polarity = (pos_count - neg_count) / (pos_count + neg_count)
        return float(np.clip(0.75 * lex_polarity + 0.25 * tb_polarity, -1.0, 1.0))
    return float(np.clip(tb_polarity, -1.0, 1.0))


def sentiment_agent(state: SpeechAnalysisState) -> dict:
    """
    Analyzes text emotional polarity and trajectory volatility across all speech segments 
    utilizing a Spanish-aware hybrid NLP processing engine.
    """
    segments_list = state.get("segments", [])
    
    if not segments_list or not any(segment.get("text") for segment in segments_list):
        print(f"  → [{AGENT_NAME} Agent] Alerta: No se detectó texto en los segmentos. Omitiendo análisis.")
        return {
            "sentiment_metrics": {
                "engine": "textblob_spanish_hybrid", 
                "by_segment": [], 
                "summary": {
                    "mean_polarity": 0.0,
                    "polarity_deviation": 0.0,
                    "volatility": 0.0,
                    "positive_ratio": 0.0,
                    "negative_ratio": 0.0
                },
                "markers_direction": {"sentiment_risk": +1, "polarity": 0}
            },
            "logs": [f"[{AGENT_NAME}] Ejecución omitida por falta de transcripción textual."]
        }

    segments_output = []
    polarity_timeline = []

    for segment in segments_list:
        segment_text = segment.get("text", "")
        polarity_score = _score_spanish_sentiment(segment_text)
        polarity_timeline.append(polarity_score)
        
        if polarity_score > 0.12:
            categorical_label = "POS"
        elif polarity_score < -0.12:
            categorical_label = "NEG"
        else:
            categorical_label = "NEU"
            
        # Riesgo de tensión emocional o crispación (0 a 100)
        sentiment_risk = float(np.clip(((-polarity_score) + 0.2) * 50.0 if polarity_score < 0 else 10.0, 0.0, 100.0))
            
        segments_output.append({
            "id": segment["id"],
            "polarity": polarity_score,
            "sentiment_risk": sentiment_risk,
            "label": categorical_label,
            "emotions": {}
        })

    polarity_array = np.array(polarity_timeline, dtype=float)
    if len(polarity_array) > 1:
        trajectory_changes = np.abs(np.diff(polarity_array))
        volatility_score = float(trajectory_changes.mean())
    else:
        volatility_score = 0.0

    summary_metrics = {
        "mean_polarity": float(polarity_array.mean()),
        "polarity_deviation": float(polarity_array.std()),
        "volatility": volatility_score,
        "positive_ratio": float(np.mean([x["label"] == "POS" for x in segments_output])),
        "negative_ratio": float(np.mean([x["label"] == "NEG" for x in segments_output])),
    }

    print(f"  → [{AGENT_NAME} Agent] Análisis completado. Polaridad Media: {summary_metrics['mean_polarity']:.2f}, Volatilidad: {summary_metrics['volatility']:.2f}")

    return {
        "sentiment_metrics": {
            "engine": "textblob_spanish_hybrid",
            "by_segment": segments_output,
            "summary": summary_metrics,
            "markers_direction": {"sentiment_risk": +1, "polarity": 0}
        },
        "logs": [f"[{AGENT_NAME}] Polaridad media: {summary_metrics['mean_polarity']:.2f}, Volatilidad: {summary_metrics['volatility']:.2f}"]
    }