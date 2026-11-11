from typing import TypedDict, List, Dict, Any

class SpeechAnalysisState(TypedDict):
    """
    ENTIDAD DE DOMINIO - CLEAN ARCHITECTURE
    Mapea la memoria compartida inmutable que transita por el grafo de LangGraph.
    Garantiza el tipado estricto y evita errores de clave (KeyErrors).
    """
    video_path: str
    audio_path: str
    transcription_text: str
    segments: List[Dict[str, Any]]
    
    current_claim: str
    factual_evidence: str
    confidence_score: float
    verdict: str
    
    is_waiting_approval: bool
    approval_granted: bool
    rag_retry_count: int
