from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.domain.models import SpeechAnalysisState

# Importación de los nodos de tus compañeros
try:
    from src.agentes.transcription_agent import transcription_agent
    from src.agentes.factual_agent import factual_agent
    from src.agentes.judge_agent import judge_agent
    from src.agentes.report_agent import report_agent
except ImportError:
    # Mocks de respaldo secuencial ante fallas en los paths del equipo
    def transcription_agent(state: SpeechAnalysisState) -> Dict[str, Any]: return {"transcription_text": "Mock"}
    def factual_agent(state: SpeechAnalysisState) -> Dict[str, Any]: return {"factual_evidence": "Mock"}
    def judge_agent(state: SpeechAnalysisState) -> Dict[str, Any]: return {"verdict": "DEVIATION_GRAVE", "confidence_score": 0.5}
    def report_agent(state: SpeechAnalysisState) -> Dict[str, Any]: return {}

def construir_grafo_sistema_core():
    """
    CONSTRUCTOR DE LA ARQUITECTURA MULTIAGENTE BASE
    Configura el pipeline e inyecta el control perimetral Human-in-the-Loop.
    """
    workflow = StateGraph(SpeechAnalysisState)
    
    # 1. Registro formal de estaciones de trabajo (Nodos Autónomos)
    workflow.add_node("transcriptor", transcription_agent)
    workflow.add_node("agente_factual", factual_agent)
    workflow.add_node("agente_juez", judge_agent)
    workflow.add_node("agente_reportero", report_agent)
    
    # 2. Configuración del Pipeline Secuencial Basal
    workflow.set_entry_point("transcriptor")
    workflow.add_edge("transcriptor", "agente_factual")
    workflow.add_edge("agente_factual", "agente_juez")
    workflow.add_edge("agente_juez", "agente_reportero")
    workflow.add_edge("agente_reportero", END)
    
    # 3. Capa de Persistencia en Memoria para los estados
    checkpointer = MemorySaver()
    
    # 4. COMPILACIÓN CON INTERRUPCIÓN MECÁNICA HUMAN-IN-THE-LOOP
    # Se interrumpe JUSTO ANTES de que actúe el agente reportero
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["agente_reportero"]
    )
