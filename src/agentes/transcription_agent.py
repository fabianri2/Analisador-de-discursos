"""Agente de Transcripción: convierte el audio en segmentos con texto y tiempos.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

from ..state import SpeechAnalysisState

AGENT_NAME = "Transcription"
TIMESTAMP_PATTERN = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _convert_timestamp_to_seconds(timestamp_str: str) -> float:
    """Converts standard clock or subtitle timestamps into pure numeric seconds."""
    match = TIMESTAMP_PATTERN.search(timestamp_str)
    if not match:
        time_parts = timestamp_str.strip().split(":")
        return sum(float(part) * 60 ** index for index, part in enumerate(reversed(time_parts)))
    hours, minutes, seconds, milliseconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / (10 ** len(milliseconds))


def _read_local_subtitles(file_path: Path) -> List[Dict[str, Any]]:
    """Parses local .srt or .vtt subtitle assets if available beside the target video file."""
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    text_blocks = re.split(r"\n\s*\n", raw_text.replace("\r", ""))
    parsed_segments: List[Dict[str, Any]] = []
    
    for block in text_blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        for index, line in enumerate(lines):
            if "-->" in line:
                start_str, end_str = [part.strip().split(" ") for part in line.split("-->")]
                content = " ".join(lines[index + 1:])
                content = re.sub(r"<[^>]+>", "", content).strip()
                if content:
                    parsed_segments.append({
                        "id": len(parsed_segments), 
                        "start": _convert_timestamp_to_seconds(start_str),
                        "end": _convert_timestamp_to_seconds(end_str), 
                        "text": content
                    })
                break
    return parsed_segments


def _execute_faster_whisper(audio_path: str, model_size: str, target_language: str) -> List[Dict[str, Any]]:
    """Runs high-performance local transcription utilizing the faster-whisper CTranslate2 engine."""
    from faster_whisper import WhisperModel  # type: ignore

    model_instance = WhisperModel(model_size, device="auto", compute_type="auto")
    segments_generator, _ = model_instance.transcribe(audio_path, language=target_language, vad_filter=True)
    return [
        {
            "id": index, 
            "start": float(segment.start), 
            "end": float(segment.end), 
            "text": segment.text.strip()
        }
        for index, segment in enumerate(segments_generator)
    ]


def _execute_openai_whisper(audio_path: str, model_size: str, target_language: str) -> List[Dict[str, Any]]:
    """Backup transcription routing utilizando el framework base openai-whisper."""
    import whisper  # type: ignore

    raw_result = whisper.load_model(model_size).transcribe(audio_path, language=target_language)
    return [
        {
            "id": index, 
            "start": float(segment["start"]), 
            "end": float(segment["end"]),
            "text": segment["text"].strip()
        } 
        for index, segment in enumerate(raw_result["segments"])
    ]


def _generate_fallback_windows(duration: float, window_width: float) -> List[Dict[str, Any]]:
    """Degradation fallback generating fixed timeline fragments when no transcription engine is present."""
    window_segments = []
    current_time = 0.0
    segment_index = 0
    
    while current_time < duration:
        window_segments.append({
            "id": segment_index, 
            "start": current_time, 
            "end": min(current_time + window_width, duration), 
            "text": ""
        })
        current_time += window_width
        segment_index += 1
    return window_segments


def transcription_agent(state: Any) -> dict:
    """
    Main transcription portal orchestrating localized file queries, 
    Whisper runtime pipelines, or sliding window fallbacks.
    """
    # SALVAGUARDA ABSOLUTA CONTROLADA: Evita el KeyError 'config' extrayendo valores seguros por defecto
    try:
        if isinstance(state, dict) and "config" in state:
            transcription_config = state["config"]["transcription"]
        else:
            transcription_config = {"whisper_model": "small", "language": "es", "segment_window_seconds": 8.0}
    except (KeyError, TypeError):
        transcription_config = {"whisper_model": "small", "language": "es", "segment_window_seconds": 8.0}

    # Control seguro de rutas de archivos inyectadas por el nodo 1
    video_source_str = state.get("video", "videos/discurso.mp4")
    video_path = Path(video_source_str)
    logged_errors: List[str] = []

    # 1. Prioridad: Verificar existencia de subtítulos locales (.srt o .vtt) junto al video
    for extension in (".srt", ".vtt"):
        subtitle_file = video_path.with_suffix(extension)
        if subtitle_file.exists():
            segments = _read_local_subtitles(subtitle_file)
            if segments:
                print(f"  → [{AGENT_NAME}] Ingestados {len(segments)} segmentos desde subtítulos locales.")
                return {
                    "segments": segments, 
                    "transcription_source": f"subtitles {subtitle_file.name}",
                    "logs": [f"[{AGENT_NAME}] {len(segments)} segmentos cargados desde {subtitle_file.name}"]
                }

    # 2. Segunda Opción: Ejecución computacional con motores de IA (Whisper)
    engines_pipeline = (
        ("faster-whisper", _execute_faster_whisper), 
        ("openai-whisper", _execute_openai_whisper)
    )
    
    # Busca la onda de audio generada por ffmpeg en la carpeta temporal
    audio_target = state.get("audio")
    if not audio_target or not Path(audio_target).exists():
        working_dir_base = state.get("working_dir", "results")
        audio_target = str(Path(working_dir_base) / video_path.stem / "audio.wav")

    for engine_name, transcription_function in engines_pipeline:
        try:
            if Path(audio_target).exists():
                segments = transcription_function(audio_target, transcription_config.get("whisper_model", "small"), transcription_config.get("language", "es"))
                if segments:
                    print(f"  → [{AGENT_NAME}] Procesamiento exitoso utilizando el motor de IA: {engine_name}")
                    return {
                        "segments": segments, 
                        "transcription_source": engine_name,
                        "logs": [f"[{AGENT_NAME}] {len(segments)} segmentos procesados con {engine_name}"]
                    }
        except (ImportError, Exception) as error_instance:
            logged_errors.append(f"[{AGENT_NAME}] {engine_name} falló o no está inicializado: {error_instance}")

    # 3. Tercera Opción: Degradación controlada por Ventanas Estáticas sin texto
    video_duration = state.get("duration", 72.0)
    window_width = transcription_config.get("segment_window_seconds", 8.0)
    segments = _generate_fallback_windows(video_duration, window_width)
    
    print(f"  → [{AGENT_NAME}] Alerta: Ejecutando modo alternativo de segmentación por ventanas fijas.")
    return {
        "segments": segments, 
        "transcription_source": "sliding_windows_fallback",
        "logs": [f"[{AGENT_NAME}] {len(segments)} ventanas generadas por degradación"], 
        "errors": logged_errors
    }