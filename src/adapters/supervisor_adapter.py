"""Capa de Adaptadores - Conexión de Agentes y Nodos del Supervisor.

Adapta las interfaces de dominio a los agentes existentes del sistema
y expone los nodos compatibles con la especificación de LangGraph.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from ..agentes.factual_agent import factual_alignment_agent
from ..agentes.ingestion_agent import ingestion_agent
from ..agentes.judge_agent import consolidation_judge_agent
from ..agentes.linguistic_agent import linguistic_agent
from ..agentes.promise_agent import promise_extractor_agent
from ..agentes.report_agent import report_generator_agent
from ..agentes.sentiment_agent import sentiment_agent
from ..agentes.syntactic_agent import text_opacity_analyst_agent
from ..agentes.transcription_agent import transcription_agent
from ..application.supervisor_policy import StandardSupervisorPolicy
from ..domain.entities import SupervisorDirective, SupervisorMetrics, WorkerType


class SupervisorAdapter:
    """Controlador y adaptador del nodo Supervisor en el grafo de LangGraph."""

    def __init__(self, policy: StandardSupervisorPolicy | None = None):
        self.policy = policy or StandardSupervisorPolicy()

    def supervisor_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Nodo del Agente Supervisor: evalúa el estado del sistema y emite directivas de enrutamiento."""
        trace: List[Dict[str, Any]] = list(state.get("supervisor_trace") or [])
        cycle = len(trace) + 1

        directive: SupervisorDirective = self.policy.evaluate(state, cycle)

        print(
            f"  [Supervisor | Ciclo {cycle}] Fase: {directive.phase.value} "
            f"→ Asignado a: '{directive.next_worker.value}' | Motivo: {directive.rationale}"
        )

        trace_entry = {
            "cycle": directive.cycle,
            "phase": directive.phase.value,
            "assigned_worker": directive.next_worker.value,
            "rationale": directive.rationale,
            "critical_alert": directive.critical_alert,
            "timestamp": time.time(),
        }
        trace.append(trace_entry)

        updates: Dict[str, Any] = {
            "next_worker": directive.next_worker.value,
            "supervisor_trace": trace,
            "logs": [f"[Supervisor c{cycle}] -> {directive.next_worker.value}: {directive.rationale}"],
        }

        # Consolidamos métricas auditables antes de generar el reporte y al cerrar en FINISH,
        # para que report.html y result.json reflejen la trazabilidad completa del Supervisor.
        if directive.next_worker in (WorkerType.REPORT_GENERATION, WorkerType.FINISH):
            updates["supervisor_metrics"] = self._compile_supervisor_metrics(state, trace)

        return updates

    @staticmethod
    def _compile_supervisor_metrics(state: Dict[str, Any], trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Construye la estructura auditable SupervisorMetrics a partir del trazo acumulado."""
        workers_sequence = [t["assigned_worker"] for t in trace if t["assigned_worker"] != "FINISH"]
        criticals = 0
        factual_metrics = state.get("factual_rag_metrics") or {}
        if "summary" in factual_metrics:
            criticals = factual_metrics["summary"].get("critical_segments_count", 0)

        verdict_val = state.get("verdict", {}).get("label")
        conf_val = state.get("verdict", {}).get("confidence", 0.0)

        metrics = SupervisorMetrics(
            architecture="supervisor",
            total_cycles=len(trace),
            workers_called=workers_sequence,
            critical_discrepancies_detected=criticals,
            rechecks_requested=0,
            final_verdict=verdict_val,
            confidence_score=conf_val,
        )
        return metrics.to_dict()


# Instancia por defecto del adaptador
default_supervisor_adapter = SupervisorAdapter()


# =====================================================================
# ADAPTADORES DE TRABAJADORES (WORKERS)
# =====================================================================

def ingestion_transcription_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Ingesta de Medios y Transcripción Fonética."""
    updates: Dict[str, Any] = {}
    video_source = Path(state.get("video", ""))

    # 1. Intento primario: Si el archivo es un .srt o si existen subtítulos locales alineados
    trans_res = transcription_agent(state)
    if trans_res.get("segments"):
        updates.update(trans_res)
        return updates

    # 2. Extracción de audio vía FFmpeg si es un archivo de video
    if not state.get("audio") and video_source.suffix.lower() not in (".srt", ".vtt"):
        ing_res = ingestion_agent(state)
        updates.update(ing_res)
        state = {**state, **ing_res}

    trans_res = transcription_agent(state)
    updates.update(trans_res)
    return updates


def promise_extraction_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Aislamiento de Compromisos y Promesas de Estado."""
    return promise_extractor_agent(state)


def linguistic_analysis_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Marcadores Psicolingüísticos y Sentimiento."""
    updates: Dict[str, Any] = {}
    # Se asegura también la polaridad afectiva base si no existiera
    if not state.get("sentiment_metrics"):
        updates.update(sentiment_agent(state))
    updates.update(linguistic_agent({**state, **updates}))
    return updates


def syntactic_analysis_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Complejidad Gramatical y Opacidad Sintáctica."""
    return text_opacity_analyst_agent(state)


def factual_audit_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Validación Cruzada RAG contra Fuentes Oficiales (MEF/BCRP/JNE)."""
    return factual_alignment_agent(state)


def judge_consolidation_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Arbitraje Multimodal y Ponderación de Veredictos."""
    return consolidation_judge_agent(state)


def report_generation_worker(state: Dict[str, Any]) -> Dict[str, Any]:
    """Especialista en Ensamblaje Forense y Publicación de Reportes HTML/JSON."""
    return report_generator_agent(state)
