"""Contratos e Interfaces Abstractas de Dominio (Clean Architecture).

Define las interfaces que deben implementar los casos de uso, las políticas de
supervisión y los adaptadores de agentes especialistas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .entities import AnalysisExecutionResult, AnalysisRequest, SupervisorDirective


class ISupervisorPolicy(ABC):
    """Contrato abstracto para la política de decisión del Agente Supervisor."""

    @abstractmethod
    def evaluate(self, state: Dict[str, Any], cycle: int) -> SupervisorDirective:
        """Examina el estado actual de la investigación y determina el próximo agente a invocar."""
        raise NotImplementedError


class IWorker(ABC):
    """Contrato base para un agente especialista subordinado."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador del agente."""
        raise NotImplementedError

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta la tarea especializada y retorna el delta a fusionar en el estado."""
        raise NotImplementedError


class ISpeechPipelineGateway(ABC):
    """Puerto de salida hacia el motor de orquestación multiagente (LangGraph)."""

    @abstractmethod
    def run_pipeline(
        self,
        video_path: str,
        config: Dict[str, Any],
        output_dir: str,
        speaker: str | None,
        verbose: bool,
        architecture: str,
    ) -> Dict[str, Any]:
        """Ejecuta el análisis del discurso y retorna el estado final del grafo."""
        raise NotImplementedError


class IReportRepository(ABC):
    """Puerto de lectura de artefactos generados (result.json y report.html)."""

    @abstractmethod
    def load_artifacts(
        self,
        json_path: str | None,
        html_path: str | None,
    ) -> Tuple[Dict[str, Any], str]:
        """Recupera el contenido serializado del informe JSON y HTML."""
        raise NotImplementedError


class ISpeechAnalysisUseCase(ABC):
    """Contrato de caso de uso para ejecutar una auditoría desde cualquier capa de presentación."""

    @abstractmethod
    def execute(self, request: AnalysisRequest) -> AnalysisExecutionResult:
        """Orquesta la configuración, ejecución multiagente y consolidación del resultado."""
        raise NotImplementedError

