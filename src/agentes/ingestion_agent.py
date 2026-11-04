"""Agente de Ingesta: valida el video, lee metadatos y extrae el audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

# Enlazado al esquema unificado de LangGraph en inglés
from ..state import SpeechAnalysisState

AGENT_NAME = "Ingesta"


def _run_ffprobe_metadata_scan(video_path: str) -> Dict[str, Any]:
    """Executes an internal ffprobe subprocess command to scan structural video stream matrices."""
    command_args = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
    ]
    stdout_stream = subprocess.run(command_args, capture_output=True, text=True, check=True).stdout
    return json.loads(stdout_stream)


def ingestion_agent(state: Any) -> dict:
    """
    Main ingestion agent validating video assets, reading encoding formats,
    and isolating monochromatic 16kHz PCM audio files required for Whisper transcription.
    """
    # SALVAGUARDA ABSOLUTA CONTROLADA: Extrae la ruta de forma segura sin importar cómo la envíe el grafo
    video_source = None
    if isinstance(state, dict):
        video_source = state.get("video") or state.get("video_path") or state.get("speech_id")
    elif hasattr(state, "video"):
        video_source = getattr(state, "video")
        
    # Si por alguna razón sigue vacío, busca el video por defecto en la carpeta del proyecto
    if not video_source:
        video_source = "videos/discurso.mp4"

    if not Path(video_source).exists():
        raise FileNotFoundError(f"Error de Ingesta: No existe el archivo de video especificado: {video_source}")
        
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Error de Dependencia: Se requiere ffmpeg instalado y configurado en el PATH del sistema.")

    # Mapeo seguro de configuración
    try:
        sampling_rate = state["config"]["audio"]["sampling_rate"]
        working_dir_base = state.get("working_dir", "results")
    except (KeyError, TypeError):
        sampling_rate = 16000
        working_dir_base = "results"
    
    working_directory = Path(working_dir_base) / Path(video_source).stem
    working_directory.mkdir(parents=True, exist_ok=True)
    audio_output_path = working_directory / "audio.wav"

    # Extracción computacional de metadatos forenses
    metadata_info = _run_ffprobe_metadata_scan(video_source)
    duration_seconds = float(metadata_info["format"].get("duration", 0.0))
    fps_rate = 0.0
    has_video_track = has_audio_track = False
    
    for stream in metadata_info.get("streams", []):
        if stream.get("codec_type") == "video":
            has_video_track = True
            numerator, _, denominator = stream.get("avg_frame_rate", "0/1").partition("/")
            fps_rate = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
        if stream.get("codec_type") == "audio":
            has_audio_track = True

    if not has_audio_track:
        raise RuntimeError("Falla de Ingesta: El archivo multimedia provisto no contiene ninguna pista de audio ejecutable.")

    # Invocación computacional a ffmpeg para aislar el canal de voz en PCM monocanal
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video_source, "-vn", "-ac", "1",
         "-ar", str(sampling_rate), "-acodec", "pcm_s16le", str(audio_output_path)],
        check=True,
    )

    print(f"  → [{AGENT_NAME} Agent] Archivo multimedia validado. Duración: {duration_seconds:.1f}s, Audio extraído con éxito.")

    # Retorna los metadatos y la ruta de audio limpia al estado de LangGraph
    return {
        "audio": str(audio_output_path),
        "duration": duration_seconds,
        "fps": fps_rate,
        "metadata": {
            "has_video_track": has_video_track, 
            "has_audio_track": has_audio_track,
            "format_name": metadata_info["format"].get("format_name")
        },
        "logs": [f"[{AGENT_NAME}] Duración {duration_seconds:.1f}s, {fps_rate:.1f} fps. Pista PCM aislada."]
    }