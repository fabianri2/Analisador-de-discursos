"""Capa de Adaptadores - Clean Architecture."""
from .supervisor_adapter import (
    SupervisorAdapter,
    default_supervisor_adapter,
    factual_audit_worker,
    ingestion_transcription_worker,
    judge_consolidation_worker,
    linguistic_analysis_worker,
    promise_extraction_worker,
    report_generation_worker,
    syntactic_analysis_worker,
)

__all__ = [
    "SupervisorAdapter",
    "default_supervisor_adapter",
    "ingestion_transcription_worker",
    "promise_extraction_worker",
    "linguistic_analysis_worker",
    "syntactic_analysis_worker",
    "factual_audit_worker",
    "judge_consolidation_worker",
    "report_generation_worker",
]
