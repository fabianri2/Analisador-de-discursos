"""Fachada de Arquitectura: Agente IA Multi-Agente con Supervisor (Clean Architecture).

Este módulo expone la interfaz de alto nivel para la arquitectura supervisada,
integrándola limpiamente con el orquestador general y la CLI de análisis.

Flujo:
    Supervisor  ──(Evalúa estado y decide especialista)──> Worker Asignado
         ▲                                                       │
         └────────────────(Devuelve control y métricas)──────────┘
"""
from __future__ import annotations

from typing import Any, Dict

from ..domain.entities import SupervisorDirective, SupervisorMetrics, SupervisorPhase, WorkerType
from ..infrastructure.supervisor_graph import build_supervisor_graph, run_supervisor_sequential_fallback
from ..state import SpeechAnalysisState

ARCHITECTURE_NAME = "supervisor"


def analyze_with_supervisor(state: SpeechAnalysisState, verbose: bool = True) -> SpeechAnalysisState:
    """Ejecuta el análisis forense completo gobernado por el Agente Supervisor."""
    state["architecture"] = ARCHITECTURE_NAME
    state.setdefault("supervisor_trace", [])

    try:
        graph = build_supervisor_graph(verbose=verbose)
        final_state = graph.invoke(state)
        return final_state
    except (ImportError, Exception) as exc:
        if verbose:
            print(f"  [Aviso Supervisor] Activando motor de ejecución supervisada directa ({exc})")
        return run_supervisor_sequential_fallback(state, verbose=verbose)
