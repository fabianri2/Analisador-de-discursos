"""Orquestador multiagente (LangGraph).

Flujo:
                       ┌─> Sentiment ──┐
  Ingestion -> Transcription ─> Linguistic ─┤
                       ├─> Promise ────┼─> Factual -> Judge -> Report
                       └─> Syntactic ──┘

Los cuatro agentes de análisis corren en paralelo (misma superstep de
LangGraph) porque solo dependen de la transcripción y del audio/video.
Si un agente de análisis falla, el sistema registra el error y continúa
con las modalidades restantes (degradación controlada).
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Dict

from .agentes.factual_agent import factual_alignment_agent
from .agentes.syntactic_agent import text_opacity_analyst_agent
from .agentes.ingestion_agent import ingestion_agent
from .agentes.judge_agent import consolidation_judge_agent
from .agentes.linguistic_agent import linguistic_agent
from .agentes.promise_agent import promise_extractor_agent
from .agentes.report_agent import report_generator_agent
from .agentes.sentiment_agent import sentiment_agent
from .agentes.transcription_agent import transcription_agent
from .arquitecturas.plan_and_execute import plan_and_execute_agent
from .arquitecturas.supervisor import build_supervisor_graph, run_supervisor_sequential_fallback
from .state import SpeechAnalysisState

AGENTS: Dict[str, Callable] = {
   "ingestion": ingestion_agent,
    "transcription": transcription_agent,
    "sentiment": sentiment_agent,
    "linguistic": linguistic_agent,
    "promise": promise_extractor_agent,
    "syntactic": text_opacity_analyst_agent,
    "factual": factual_alignment_agent,
    "plan_execute": plan_and_execute_agent,
    "judge": consolidation_judge_agent,
    "report": report_generator_agent,
}
ANALYSIS_BRANCHES = ("sentiment", "linguistic", "promise", "syntactic")
CRITICAL_AGENTS = {"ingestion", "transcription", "judge", "report"}
SUPPORTED_ARCHITECTURES = ("pipeline", "plan-and-execute", "supervisor")


def _verification_node(architecture: str) -> str:
    """Devuelve el nodo de verificación factual según la arquitectura elegida."""
    return "plan_execute" if architecture == "plan-and-execute" else "factual"


def _wrap_node(name: str, fn: Callable, verbose: bool) -> Callable:
    """Wraps agent execution to measure performance metrics and manage controlled degradation."""
    def node(state: SpeechAnalysisState) -> dict:
        t0 = time.time()
        if verbose:
            print(f"  → Ejecutando nodo: {name}...", flush=True)
        try:
            output = fn(state)
        except Exception as e:  # noqa: BLE001
            if name in CRITICAL_AGENTS:
                raise
            if verbose:
                traceback.print_exc()
            output = {
                name: {"by_segment": [], "summary": {}},
                "errors": [f"[{name}] falló: {type(e).__name__}: {e}"]
            }
        output.setdefault("logs", [])
        output["logs"] = output["logs"] + [f"[{name}] {time.time() - t0:.1f}s"]
        return output
    return node


def build_workflow_graph(verbose: bool = True, architecture: str = "pipeline"):
    """Compiles the multi-agent execution architecture using LangGraph StateGraph."""
    if architecture == "supervisor":
        return build_supervisor_graph(verbose=verbose)

    from langgraph.graph import END, START, StateGraph  # type: ignore

    verification = _verification_node(architecture)
    graph = StateGraph(SpeechAnalysisState)
    
    for name, fn in AGENTS.items():
        graph.add_node(name, _wrap_node(name, fn, verbose))
        
    graph.add_edge(START, "ingestion")
    graph.add_edge("ingestion", "transcription")
    
    for branch in ANALYSIS_BRANCHES:
        graph.add_edge("transcription", branch)
        
    graph.add_edge(list(ANALYSIS_BRANCHES), verification)
    graph.add_edge(verification, "judge")
    graph.add_edge("judge", "report")
    graph.add_edge("report", END)
    
    return graph.compile()


def _run_sequential_fallback(state: SpeechAnalysisState, verbose: bool,
                             architecture: str = "pipeline") -> SpeechAnalysisState:
    """Local fallback architecture execution if LangGraph is missing."""
    if architecture == "supervisor":
        return run_supervisor_sequential_fallback(state, verbose=verbose)

    skipped = {"factual", "plan_execute"} - {_verification_node(architecture)}
    for name, fn in AGENTS.items():
        if name in skipped:
            continue
        output = _wrap_node(name, fn, verbose)(state)
        for key, val in output.items():
            if key in ("logs", "errors"):
                state[key] = state.get(key, []) + val
            else:
                state[key] = val
    return state


def analyze_video_pipeline(video_path: str, config: Dict, output_dir: str = "results",
                           speaker: str | None = None, verbose: bool = True,
                           architecture: str = "pipeline") -> SpeechAnalysisState:
    """Main execution portal for analyzing speech documents and video assets."""
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Arquitectura no soportada: {architecture}. Use una de {SUPPORTED_ARCHITECTURES}.")
    working_dir = Path(output_dir) / Path(video_path).stem
    
    # Inicialización del diccionario con la clave exacta 'video' requerida por ingestion_agent.py
    state: SpeechAnalysisState = {
        "video": str(video_path), 
        "speaker": speaker or Path(video_path).stem,
        "working_dir": str(working_dir), 
        "config": config,
        "logs": [], 
        "errors": [],
    }
    
    try:
        graph = build_workflow_graph(verbose, architecture)
    except ImportError:
        if verbose:
            print("  (LangGraph no detectado en el entorno: ejecutando modo de respaldo secuencial)")
        return _run_sequential_fallback(state, verbose, architecture)
        
    return graph.invoke(state)