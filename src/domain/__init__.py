"""Capa de Dominio - Clean Architecture."""
from .entities import (
    AlgorithmMode,
    AnalysisExecutionResult,
    AnalysisRequest,
    SupervisorDirective,
    SupervisorMetrics,
    SupervisorPhase,
    WorkerType,
)
from .interfaces import (
    IReportRepository,
    ISpeechAnalysisUseCase,
    ISpeechPipelineGateway,
    ISupervisorPolicy,
    IWorker,
)

__all__ = [
    "AlgorithmMode",
    "AnalysisRequest",
    "AnalysisExecutionResult",
    "WorkerType",
    "SupervisorPhase",
    "SupervisorDirective",
    "SupervisorMetrics",
    "ISupervisorPolicy",
    "IWorker",
    "ISpeechPipelineGateway",
    "IReportRepository",
    "ISpeechAnalysisUseCase",
]
