"""Capa de Infraestructura - Grafo Multi-Agente con Supervisor en LangGraph.

Implementa la topología en estrella (Hub-and-Spoke) gestionada por el Agente Supervisor.
Cada agente especialista procesa su tarea y devuelve el control al supervisor central
hasta que la política determine que la auditoría está completa.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict

from ..adapters.supervisor_adapter import (
    default_supervisor_adapter,
    factual_audit_worker,
    ingestion_transcription_worker,
    judge_consolidation_worker,
    linguistic_analysis_worker,
    promise_extraction_worker,
    report_generation_worker,
    syntactic_analysis_worker,
)
from ..domain.entities import WorkerType
from ..state import SpeechAnalysisState

WORKER_NODES: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    WorkerType.INGESTION_TRANSCRIPTION.value: ingestion_transcription_worker,
    WorkerType.PROMISE_EXTRACTION.value: promise_extraction_worker,
    WorkerType.LINGUISTIC_ANALYSIS.value: linguistic_analysis_worker,
    WorkerType.SYNTACTIC_ANALYSIS.value: syntactic_analysis_worker,
    WorkerType.FACTUAL_AUDIT.value: factual_audit_worker,
    WorkerType.JUDGE_CONSOLIDATION.value: judge_consolidation_worker,
    WorkerType.REPORT_GENERATION.value: report_generation_worker,
}


def _wrap_supervised_node(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]], verbose: bool) -> Callable:
    """Envuelve la ejecución del especialista midiendo rendimiento y aplicando degradación controlada."""
    def node(state: SpeechAnalysisState) -> dict:
        t0 = time.time()
        if verbose:
            print(f"    ↳ [Ejecutando Especialista] {name}...", flush=True)
        try:
            output = fn(state)
        except Exception as e:
            if verbose:
                traceback.print_exc()
            output = {
                "errors": [f"[Worker {name}] Fallo controlado: {type(e).__name__}: {e}"]
            }
        output.setdefault("logs", [])
        output["logs"] = output["logs"] + [f"[{name}] {time.time() - t0:.2f}s"]
        return output
    return node


def build_supervisor_graph(verbose: bool = True):
    """Compila el StateGraph de LangGraph con arquitectura Multi-Agente con Supervisor."""
    from langgraph.graph import END, START, StateGraph

    workflow = StateGraph(SpeechAnalysisState)

    # 1. Registrar el nodo central del Supervisor
    workflow.add_node(
        "supervisor",
        _wrap_supervised_node("supervisor", default_supervisor_adapter.supervisor_node, verbose=False),
    )

    # 2. Registrar todos los nodos especialistas (Workers)
    for worker_name, worker_fn in WORKER_NODES.items():
        workflow.add_node(
            worker_name,
            _wrap_supervised_node(worker_name, worker_fn, verbose=verbose),
        )

    # 3. Punto de entrada: siempre inicia en el Agente Supervisor
    workflow.add_edge(START, "supervisor")

    # 4. Enrutamiento condicional desde el Supervisor hacia el especialista asignado o FINISH
    def _route_next_worker(state: SpeechAnalysisState) -> str:
        next_worker = state.get("next_worker", WorkerType.FINISH.value)
        if next_worker == WorkerType.FINISH.value or next_worker not in WORKER_NODES:
            return END
        return next_worker

    # Mapeo explícito de destinos para el validador estricto de LangGraph
    routing_destinations = {worker_name: worker_name for worker_name in WORKER_NODES}
    routing_destinations[END] = END

    workflow.add_conditional_edges(
        "supervisor",
        _route_next_worker,
        routing_destinations,
    )

    # 5. Bordes de retorno: CADA TRABAJADOR DEVUELVE EL CONTROL AL SUPERVISOR (Hub & Spoke)
    for worker_name in WORKER_NODES:
        workflow.add_edge(worker_name, "supervisor")

    return workflow.compile()


def run_supervisor_sequential_fallback(state: SpeechAnalysisState, verbose: bool = True) -> SpeechAnalysisState:
    """Modo de ejecución supervisada determinista sin requerir la máquina de estados de LangGraph."""
    adapter = default_supervisor_adapter
    max_cycles = 15

    for _ in range(max_cycles):
        decision = adapter.supervisor_node(state)
        for k, v in decision.items():
            if k in ("logs", "errors"):
                state[k] = state.get(k, []) + v
            else:
                state[k] = v

        target_worker = state.get("next_worker")
        if not target_worker or target_worker == WorkerType.FINISH.value:
            break

        worker_fn = WORKER_NODES.get(target_worker)
        if worker_fn:
            worker_output = _wrap_supervised_node(target_worker, worker_fn, verbose)(state)
            for k, v in worker_output.items():
                if k in ("logs", "errors"):
                    state[k] = state.get(k, []) + v
                else:
                    state[k] = v

    return state
