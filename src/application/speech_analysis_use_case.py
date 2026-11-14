"""Capa de Aplicación - Caso de Uso de Auditoría de Discursos (Clean Architecture).

Orquesta el flujo completo de auditoría independientemente de la capa de presentación
(CLI o Interfaz Web Streamlit) y de los frameworks externos (LangGraph / ChromaDB).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from ..domain.entities import AlgorithmMode, AnalysisExecutionResult, AnalysisRequest
from ..domain.interfaces import (
    IReportRepository,
    ISpeechAnalysisUseCase,
    ISpeechPipelineGateway,
)


class AnalyzeSpeechUseCase(ISpeechAnalysisUseCase):
    """Caso de uso principal para analizar un discurso bajo cualquiera de las arquitecturas."""

    SUPPORTED_ARCHITECTURES = ("supervisor", "pipeline", "plan-and-execute")

    def __init__(
        self,
        pipeline_gateway: ISpeechPipelineGateway,
        report_repository: IReportRepository,
        config_loader: Callable[[Optional[str]], Dict[str, Any]],
    ):
        self._pipeline_gateway = pipeline_gateway
        self._report_repository = report_repository
        self._config_loader = config_loader

    def execute(self, request: AnalysisRequest) -> AnalysisExecutionResult:
        """Ejecuta el análisis forense del discurso aplicando políticas de configuración limpias."""
        if request.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Arquitectura no soportada: '{request.architecture}'. "
                f"Opciones válidas: {self.SUPPORTED_ARCHITECTURES}"
            )

        config = deepcopy(self._config_loader(None))
        judge_cfg = config.setdefault("judge", {})
        if request.algorithm_mode == AlgorithmMode.HEURISTIC:
            judge_cfg["trained_model_path"] = None
        elif not judge_cfg.get("trained_model_path"):
            judge_cfg["trained_model_path"] = "models/classifier.joblib"

        raw_state = self._pipeline_gateway.run_pipeline(
            video_path=request.video_path,
            config=config,
            output_dir=request.output_dir,
            speaker=request.speaker,
            verbose=request.verbose,
            architecture=request.architecture,
        )

        report_info = raw_state.get("report") or {}
        json_path = report_info.get("json")
        html_path = report_info.get("html")
        json_data, html_content = self._report_repository.load_artifacts(json_path, html_path)

        resolved_architecture = (
            raw_state.get("architecture")
            or (raw_state.get("supervisor_metrics") or {}).get("architecture")
            or (raw_state.get("plan_metrics") or {}).get("architecture")
            or request.architecture
        )

        return AnalysisExecutionResult(
            architecture=resolved_architecture,
            video_path=raw_state.get("video", request.video_path),
            speaker=raw_state.get("speaker") or request.speaker or "Desconocido",
            verdict=raw_state.get("verdict") or {},
            supervisor_metrics=raw_state.get("supervisor_metrics"),
            supervisor_trace=list(raw_state.get("supervisor_trace") or []),
            plan_metrics=raw_state.get("plan_metrics"),
            plan=list(raw_state.get("plan") or []),
            left_wing_evidence=list(raw_state.get("left_wing_evidence") or []),
            right_wing_evidence=list(raw_state.get("right_wing_evidence") or []),
            omitted_topics=list(raw_state.get("omitted_topics") or []),
            report_html_path=html_path,
            report_json_path=json_path,
            report_html_content=html_content,
            report_json_data=json_data,
            logs=list(raw_state.get("logs") or []),
            errors=list(raw_state.get("errors") or []),
        )
