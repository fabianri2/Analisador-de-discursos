# NOTA: Definición del esquema del estado compartido para la arquitectura LangGraph.
from typing import TypedDict, List, Dict, Any

class SpeechAnalysisState(TypedDict):
    # Variables de entrada y transcripción
    speech_id: str
    transcribed_text: str
    
    # Outputs del Agente 2 (Sintáctico/Facial original)
    syntactic_metrics: Dict[str, float]
    
    # Outputs del Agente 3 (Extractor/Prosodia original)
    extracted_promises: List[Dict[str, Any]]
    
    # Outputs del RAG y Auditoría factual con MEF/BCRP
    audit_results: Dict[str, Any]
    final_verdict: str