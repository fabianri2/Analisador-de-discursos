"""Contratos e Interfaces Abstractas de Dominio (Clean Architecture).

Define las interfaces que deben implementar los casos de uso, las políticas de
supervisión y los adaptadores de agentes especialistas.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .entities import SupervisorDirective


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
