"""Entidades de Dominio - Clean Architecture para el Patrón Supervisor.

Capa de Dominio Puro: No depende de frameworks externos (LangGraph, Flask, etc.).
Define los modelos, enums y estructuras que representan el estado, las decisiones
del supervisor y los contratos de datos del discurso.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkerType(str, Enum):
    """Identificador canónico de cada agente especialista subordinado."""
    INGESTION_TRANSCRIPTION = "ingestion_transcription"
    PROMISE_EXTRACTION = "promise_extraction"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    SYNTACTIC_ANALYSIS = "syntactic_analysis"
    FACTUAL_AUDIT = "factual_audit"
    JUDGE_CONSOLIDATION = "judge_consolidation"
    REPORT_GENERATION = "report_generation"
    FINISH = "FINISH"


class SupervisorPhase(str, Enum):
    """Fases operacionales del ciclo de vida de la auditoría supervisada."""
    INITIALIZING = "INITIALIZING"
    EXTRACTING_PROMISES = "EXTRACTING_PROMISES"
    ANALYZING_STYLE = "ANALYZING_STYLE"
    AUDITING_FACTS = "AUDITING_FACTS"
    EVALUATING_SEVERITY = "EVALUATING_SEVERITY"
    CONSOLIDATING_VERDICT = "CONSOLIDATING_VERDICT"
    GENERATING_REPORT = "GENERATING_REPORT"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class SupervisorDirective:
    """Decisión inmutable emitida por el Agente Supervisor en cada ciclo de control."""
    cycle: int
    phase: SupervisorPhase
    next_worker: WorkerType
    rationale: str
    target_evidence: Optional[str] = None
    critical_alert: bool = False


@dataclass
class SupervisorMetrics:
    """Métricas auditables del desempeño del supervisor a lo largo del discurso."""
    architecture: str = "supervisor"
    total_cycles: int = 0
    workers_called: List[str] = field(default_factory=list)
    critical_discrepancies_detected: int = 0
    rechecks_requested: int = 0
    final_verdict: Optional[str] = None
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "total_cycles": self.total_cycles,
            "workers_called": list(self.workers_called),
            "critical_discrepancies_detected": self.critical_discrepancies_detected,
            "rechecks_requested": self.rechecks_requested,
            "final_verdict": self.final_verdict,
            "confidence_score": self.confidence_score,
        }
