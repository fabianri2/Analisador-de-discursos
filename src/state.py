# NOTA: Definición del esquema del estado compartido para la arquitectura LangGraph.
from __future__ import annotations
import operator
from typing import TypedDict, List, Dict, Any, Annotated


class SpeechAnalysisState(TypedDict, total=False):
    # Variables de entrada y orquestación
    video: str
    video_path: str
    speaker: str
    working_dir: str
    config: Dict[str, Any]
    speech_id: str
    
    # Reducers for parallel nodes
    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    
    # Outputs de Ingesta
    audio: str
    duration: float
    fps: float
    metadata: Dict[str, Any]
    
    # Outputs de Transcripción
    segments: List[Dict[str, Any]]
    transcribed_text: str
    transcription_source: str
    
    # Outputs de Agentes de análisis
    sentiment_metrics: Dict[str, Any]
    linguistic_metrics: Dict[str, Any]
    syntactic_metrics: Dict[str, Any]
    promise_extraction_metrics: Dict[str, Any]
    extracted_promises: List[Dict[str, Any]]
    factual_rag_metrics: Dict[str, Any]
    
    # Auditoría, veredicto y reporte
    audit_results: Dict[str, Any]
    verdict: Dict[str, Any]
    final_verdict: str
    report: Dict[str, Any]