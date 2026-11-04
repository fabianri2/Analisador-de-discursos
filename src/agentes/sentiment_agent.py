"""Agente de Sentimiento: polaridad y emociones del texto de cada segmento.
"""
from __future__ import annotations

import numpy as np
from textblob import TextBlob
from ..state import SpeechAnalysisState

AGENT_NAME = "Sentiment"


def sentiment_agent(state: SpeechAnalysisState) -> dict:
    """
    Analyzes text emotional polarity and trajectory volatility across all speech segments 
    utilizing a lightweight NLP processing engine to avoid C++ compilation overhead.
    """
    # El agente recupera la lista unificada de segmentos en ingles generada por Whisper
    segments_list = state.get("segments", [])
    
    # Sistema de degradacion controlada: Si no hay texto extraído por los motores, se omite el analisis
    if not segments_list or not any(segment.get("text") for segment in segments_list):
        print(f"  → [{AGENT_NAME} Agent] Alerta: No se detectó texto en los segmentos. Omitiendo análisis.")
        return {
            "sentiment_metrics": {
                "engine": "textblob_fallback", 
                "by_segment": [], 
                "summary": {
                    "mean_polarity": 0.0,
                    "polarity_deviation": 0.0,
                    "volatility": 0.0,
                    "positive_ratio": 0.0,
                    "negative_ratio": 0.0
                }
            },
            "logs": [f"[{AGENT_NAME}] Ejecución omitida por falta de transcripción textual."]
        }

    segments_output = []
    polarity_timeline = []

    # Procesamiento forense computacional segmento por segmento
    for segment in segments_list:
        segment_text = segment.get("text", "")
        
        # Invocacion al motor ligero de procesamiento de lenguaje natural
        text_analysis_blob = TextBlob(segment_text)
        # TextBlob retorna un valor continuo flotante entre -1.0 (Muy Negativo) y 1.0 (Muy Positivo)
        polarity_score = float(text_analysis_blob.sentiment.polarity)
        polarity_timeline.append(polarity_score)
        
        # Clasificación categórica de la polaridad en base a umbrales estadísticos
        if polarity_score > 0.15:
            categorical_label = "POS"
        elif polarity_score < -0.15:
            categorical_label = "NEG"
        else:
            categorical_label = "NEU"
            
        segments_output.append({
            "id": segment["id"],
            "polarity": polarity_score,
            "label": categorical_label,
            "emotions": {}  # Empty block reserved for future API metadata mapping
        })

    # Convertimos a matriz de numpy para ejecutar el análisis estadístico del mensaje
    polarity_array = np.array(polarity_timeline, dtype=float)
    
    # Cálculo de la Volatilidad Emocional: Mide los saltos bruscos en el tono político del presidente
    if len(polarity_array) > 1:
        trajectory_changes = np.abs(np.diff(polarity_array))
        volatility_score = float(trajectory_changes.mean())
    else:
        volatility_score = 0.0

    # Construcción de la matriz consolidada de KPIs emocionales del discurso
    summary_metrics = {
        "mean_polarity": float(polarity_array.mean()),
        "polarity_deviation": float(polarity_array.std()),
        "volatility": volatility_score,
        "positive_ratio": float(np.mean([x["label"] == "POS" for x in segments_output])),
        "negative_ratio": float(np.mean([x["label"] == "NEG" for x in segments_output])),
    }

    print(f"  → [{AGENT_NAME} Agent] Análisis completado. Polaridad Media: {summary_metrics['mean_polarity']:.2f}, Volatilidad: {summary_metrics['volatility']:.2f}")

    # Retorna las métricas estructuradas al Grafo de Estado de LangGraph
    return {
        "sentiment_metrics": {
            "engine": "textblob_lightweight",
            "by_segment": segments_output,
            "summary": summary_metrics
        },
        "logs": [f"[{AGENT_NAME}] Motor: textblob. Polaridad media: {summary_metrics['mean_polarity']:.2f}"]
    }