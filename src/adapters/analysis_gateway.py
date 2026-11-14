"""Capa de Adaptadores - Gateways y Repositorios para la Interfaz y Orquestador.

Conecta el caso de uso AnalyzeSpeechUseCase (Capa de Aplicación) con el orquestador
LangGraph, el sistema de archivos local y los buffers de subida de Streamlit,
evitando que la capa de presentación (app.py) dependa de frameworks o lógica interna.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..application.speech_analysis_use_case import AnalyzeSpeechUseCase
from ..domain.interfaces import IReportRepository, ISpeechPipelineGateway

SUPPORTED_MEDIA_EXTENSIONS = (".srt", ".vtt", ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


class OrchestratorPipelineGateway(ISpeechPipelineGateway):
    """Adaptador que expone un corredor de flujo bajo el contrato ISpeechPipelineGateway."""

    def __init__(self, pipeline_runner: Optional[Callable[..., Dict[str, Any]]] = None):
        self._pipeline_runner = pipeline_runner

    def run_pipeline(
        self,
        video_path: str,
        config: Dict[str, Any],
        output_dir: str,
        speaker: str | None,
        verbose: bool,
        architecture: str,
    ) -> Dict[str, Any]:
        runner = self._pipeline_runner
        if runner is None:
            from ..orchestrator import analyze_video_pipeline
            runner = analyze_video_pipeline
        return dict(
            runner(
                video_path=video_path,
                config=config,
                output_dir=output_dir,
                speaker=speaker,
                verbose=verbose,
                architecture=architecture,
            )
        )


class LocalReportRepository(IReportRepository):
    """Repositorio adaptador para leer result.json y report.html desde disco."""

    def load_artifacts(
        self,
        json_path: str | None,
        html_path: str | None,
    ) -> Tuple[Dict[str, Any], str]:
        json_data: Dict[str, Any] = {}
        html_content = ""

        if json_path:
            json_file = Path(json_path)
            if json_file.exists():
                try:
                    json_data = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    json_data = {}

        if html_path:
            html_file = Path(html_path)
            if html_file.exists():
                try:
                    html_content = html_file.read_text(encoding="utf-8")
                except Exception:
                    html_content = ""

        return json_data, html_content


def list_corpus_media_files(corpus_dir: str = "videos") -> List[str]:
    """Lista archivos multimedia o subtítulos locales disponibles para auditoría rápida."""
    base = Path(corpus_dir)
    if not base.exists() or not base.is_dir():
        return []
    files = [
        str(p.as_posix())
        for p in sorted(base.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_MEDIA_EXTENSIONS
    ]
    return files


def persist_uploaded_media(filename: str, content_bytes: bytes, upload_dir: str = "videos/_uploads") -> str:
    """Almacena de forma segura un archivo subido desde la interfaz Web para su análisis."""
    safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name).strip() or "discurso_subido.mp4"
    dest_dir = Path(upload_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / safe_name
    dest_path.write_bytes(content_bytes)
    return str(dest_path.as_posix())


def create_speech_analysis_use_case() -> AnalyzeSpeechUseCase:
    """Factory de composición Clean Architecture para inyectar dependencias al caso de uso."""
    from ..config import cargar_config
    from ..orchestrator import analyze_video_pipeline

    return AnalyzeSpeechUseCase(
        pipeline_gateway=OrchestratorPipelineGateway(pipeline_runner=analyze_video_pipeline),
        report_repository=LocalReportRepository(),
        config_loader=cargar_config,
    )
