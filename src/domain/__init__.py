"""Capa de Dominio - Clean Architecture."""
from .entities import SupervisorDirective, SupervisorMetrics, SupervisorPhase, WorkerType
from .interfaces import ISupervisorPolicy, IWorker

__all__ = [
    "WorkerType",
    "SupervisorPhase",
    "SupervisorDirective",
    "SupervisorMetrics",
    "ISupervisorPolicy",
    "IWorker",
]
