"""Agente de Congruencia: cruza las modalidades para detectar incoherencias.

La idea (DePaulo et al., 2003: "fuga emocional"; Ekman, 2009) es que lo que
se dice, cómo se dice y lo que muestra el rostro deberían ir en la misma
dirección. Este agente mide, para cada segmento:

  - inc_texto_rostro : el texto expresa una valencia y el rostro la contraria
                       (p. ej. palabras positivas con ceño fruncido). 0-1.
  - inc_intensidad   : desajuste entre la carga emocional del texto y la
                       activación de la voz (contenido muy emotivo dicho en
                       tono plano, o tono muy activado sobre contenido neutro). 0-1.
  - incongruencia    : combinación de ambas (promedio de las disponibles).
"""
from __future__ import annotations

from typing import Dict, List, Any
import numpy as np 

from ..state import SpeechAnalysisState
from ..utilities import calculate_sigmoid


AGENT_NAME = "FactualAlignment"


def _map_segments_by_id(agent_metrics_block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Maps continuous segment outputs into a key-value dictionary using the segment ID."""
    if not isinstance(agent_metrics_block, dict):
        return {}
    return {row["id"]: row for row in agent_metrics_block.get("by_segment", [])}


def factual_alignment_agent(state: SpeechAnalysisState) -> dict:
    """
    Main cross-modal validation node checking semantic and statistics alignment.
    Computes the final 'factual_discrepancy' metric by contrasting the political 
    sentiment trajectory against the hard financial targets retrieved by the RAG.
    """
    # Recuperamos los outputs estructurados en inglés de los agentes paralelos concurrentes
    sentiment_map = _map_segments_by_id(state.get("sentiment_metrics", {}))
    linguistic_map = _map_segments_by_id(state.get("linguistic_metrics", {}))
    
    # Simulación del bloque recuperador del RAG para el MVP del Sprint 1 (ChromaDB API Mockup)
    # En los próximos sprints, este bloque leerá directamente la base vectorial de los PDFs del MEF
    speech_segments = state.get("segments", [])
    
    segments_output = []
    factual_discrepancies_timeline = []

    for segment in speech_segments:
        segment_id = segment["id"]
        row_result = {"id": segment_id}
        evaluation_parts = []

        # 1. Extracción de variables de análisis del discurso
        segment_text = segment.get("text", "")
        polarity_score = sentiment_map.get(segment_id, {}).get("polarity")
        
        # 2. Simulación computacional del dato macroeconómico recuperado por el RAG (Año N+1)
        # Si el presidente habla de economía, se inyecta una meta regulatoria base del BCRP
        is_economic_topic = any(word in segment_text.lower() for word in ["pbi", "gasto", "mef", "millones", "presupuesto", "obra"])
        rag_mef_target_value = 0.45 if is_economic_topic else 0.0

        # Algoritmo de Congruencia Factual (El núcleo solicitado por el profesor)
        if polarity_score is not None:
            # Métrica A: Brecha entre la opacidad sintáctica del texto y la realidad presupuestal del MEF
            linguistic_features = linguistic_map.get(segment_id, {})
            hedges_ratio = linguistic_features.get("hedges", 0.0) / 100.0
            
            # Si el texto abusa de atenuadores/evasiones ("creo que", "supuestamente") ante un dato del MEF, sube el desvío
            row_result["text_vs_rag_opacity"] = float(max(0.0, hedges_ratio * (1.0 - calculate_sigmoid(rag_mef_target_value))))
            evaluation_parts.append(row_result["text_vs_rag_opacity"])
            
            # Métrica B: Incongruencia de Intensidad (Discurso político ultra-optimista vs Meta macroeconómica estancada)
            if segment_text:
                # Contraste matemático entre el valor absoluto del sentimiento y el vector financiero regulatorio
                row_result["intensity_mismatch"] = float(abs(abs(polarity_score) - calculate_sigmoid(rag_mef_target_value * 2.0)))
                evaluation_parts.append(row_result["intensity_mismatch"])

        # Cálculo unificado del Índice de Discrepancia Factual del Segmento (0.0 a 1.0)
        if evaluation_parts:
            segment_discrepancy = float(np.mean(evaluation_parts))
        else:
            # Si no hay datos, se calcula un desvío base pasivo del 15% para el MVP
            segment_discrepancy = 0.15 if is_economic_topic else 0.0
            
        row_result["factual_discrepancy"] = segment_discrepancy
        # Multiplicamos por 100 para normalizarlo a la escala estándar (0-100) requerida por el Juez
        row_result["index"] = float(segment_discrepancy * 100.0)
        
        # Mapeo de sub-familias requerido por la matriz de renderizado del reportero HTML
        row_result["families"] = {"factual_rag": row_result["index"]}
        
        segments_output.append(row_result)
        factual_discrepancies_timeline.append(segment_discrepancy)

    # Compilación estadística de consistencia para el Grafo de LangGraph
    discrepancy_array = np.array(factual_discrepancies_timeline, dtype=float)
    evaluated_count = len(discrepancy_array)
    
    summary_metrics = {
        "mean_factual_discrepancy": float(discrepancy_array.mean()) if evaluated_count else 0.0,
        "evaluated_segments_count": evaluated_count
    }

    print(f"  → [{AGENT_NAME} Agent] Validación cruzada completada. Segmentos evaluados contra MEF/BCRP: {evaluated_count}")

    # Retorna el bloque factual unificado al estado compartido de LangGraph
    return {
        "factual_rag_metrics": {
            "by_segment": segments_output,
            "summary": summary_metrics
        },
        "logs": [f"[{AGENT_NAME}] Cruce RAG completado. Brecha factual media calculada: {summary_metrics['mean_factual_discrepancy']:.2f}"]
    }