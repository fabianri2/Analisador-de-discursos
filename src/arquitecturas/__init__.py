"""Arquitecturas agénticas alternativas al pipeline multiagente."""
from .plan_and_execute import plan_and_execute_agent
from .supervisor import analyze_with_supervisor

__all__ = ["plan_and_execute_agent", "analyze_with_supervisor"]
