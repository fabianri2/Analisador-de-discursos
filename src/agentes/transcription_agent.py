"""Agente de Transcripción: convierte el audio en segmentos con texto y tiempos.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from ..state import SpeechAnalysisState

AGENT_NAME = "Transcription"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TIMESTAMP_PATTERN = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _convert_timestamp_to_seconds(timestamp_str: str) -> float:
    """Converts standard clock or subtitle timestamps into pure numeric seconds."""
    match = TIMESTAMP_PATTERN.search(timestamp_str)
    if not match:
        time_parts = timestamp_str.strip().split(":")
        return sum(float(part) * 60 ** index for index, part in enumerate(reversed(time_parts)))
    hours, minutes, seconds, milliseconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / (10 ** len(milliseconds))


def _repair_mojibake(raw_text: str) -> str:
    """Repairs UTF-8 subtitles that were decoded once as Latin-1 (típico 'Ãº', 'Ã­')."""
    suspicious_markers = ("Ã", "Â", "â€")
    if not any(marker in raw_text for marker in suspicious_markers):
        return raw_text
    try:
        repaired = raw_text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return raw_text
    return repaired


def _read_local_subtitles(file_path: Path) -> List[Dict[str, Any]]:
    """Parses local .srt or .vtt subtitle assets if available beside the target video file."""
    raw_text = _repair_mojibake(file_path.read_text(encoding="utf-8-sig", errors="ignore"))
    text_blocks = re.split(r"\n\s*\n", raw_text.replace("\r", ""))
    parsed_segments: List[Dict[str, Any]] = []
    
    for block in text_blocks:
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        for index, line in enumerate(lines):
            if "-->" in line:
                start_str, end_str = [part.strip().split(" ")[0] for part in line.split("-->")]
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


def _load_pcm_wav_samples(audio_path: str) -> "np.ndarray | None":
    """Reads the 16 kHz mono PCM WAV extracted by the ingestion agent as a float32 waveform.

    Entregar la onda ya decodificada evita depender de PyAV/ffmpeg al transcribir
    (faster-whisper usa av.open y algunas versiones de PyAV no aceptan sus argumentos).
    """
    import wave

    import numpy as np

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                return None
            raw_frames = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError):
        return None
    return np.frombuffer(raw_frames, dtype=np.int16).astype("float32") / 32768.0


def _build_whisper_model(model_loader, model_size: str):
    """Instantiates the CTranslate2 model on CUDA when available, otherwise on CPU (int8)."""
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:
            try:
                return model_loader(model_size, device="cuda", compute_type="float16")
            except Exception:  # noqa: BLE001 — driver presente pero sin las DLL de CUDA
                pass
    except Exception:  # noqa: BLE001
        pass
    return model_loader(model_size, device="cpu", compute_type="int8")


def _is_writable_directory(candidate: Path) -> bool:
    """Comprueba de verdad que se puede escribir: crear la carpeta no basta (ACL/antivirus)."""
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _resolve_huggingface_cache() -> Path:
    """Chooses a writable HuggingFace cache so Whisper can download its model.

    En algunos equipos Windows la carpeta por defecto (%USERPROFILE%\\.cache\\huggingface)
    devuelve WinError 5 (Acceso denegado). En ese caso se usa una carpeta local del
    proyecto, que siempre es escribible porque ahí se clonó el repositorio.
    """
    configured = os.environ.get("HF_HOME")
    if configured:
        return Path(configured)

    candidates = [
        REPOSITORY_ROOT / ".hf_cache",
        Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "huggingface",
        Path(tempfile.gettempdir()) / "huggingface_cache",
    ]
    for candidate in candidates:
        if _is_writable_directory(candidate):
            return candidate
    return Path(tempfile.gettempdir()) / "huggingface_cache"


def _guardar_cache_huggingface() -> None:
    """Configura HF_HOME/HF_HUB_DISABLE_SYMLINKS antes de importar faster-whisper."""
    os.environ.setdefault("HF_HOME", str(_resolve_huggingface_cache()))
    # Windows sin modo desarrollador no permite crear symlinks: forzamos copia real.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def _execute_faster_whisper(audio_path: str, model_size: str, target_language: str) -> List[Dict[str, Any]]:
    """Runs high-performance local transcription utilizing the faster-whisper CTranslate2 engine."""
    _guardar_cache_huggingface()
    from faster_whisper import WhisperModel  # type: ignore

    audio_input = _load_pcm_wav_samples(audio_path)
    audio_source = audio_input if audio_input is not None else audio_path

    try:
        model_instance = _build_whisper_model(WhisperModel, model_size)
        segments_generator, _ = model_instance.transcribe(
            audio_source, language=target_language, vad_filter=True
        )
        decoded_segments = list(segments_generator)
    except Exception:  # noqa: BLE001 — GPU disponible sin DLLs de CUDA: reintenta en CPU
        model_instance = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_generator, _ = model_instance.transcribe(
            audio_source, language=target_language, vad_filter=True
        )
        decoded_segments = list(segments_generator)

    return [
        {
            "id": index,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip()
        }
        for index, segment in enumerate(decoded_segments)
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
                transcribed_text = " ".join(s.get("text", "") for s in segments)
                print(f"  → [{AGENT_NAME}] Ingestados {len(segments)} segmentos desde subtítulos locales.")
                return {
                    "segments": segments, 
                    "transcribed_text": transcribed_text,
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
        audio_target = str(Path(working_dir_base) / "audio.wav")

    for engine_name, transcription_function in engines_pipeline:
        try:
            if Path(audio_target).exists():
                segments = transcription_function(audio_target, transcription_config.get("whisper_model", "small"), transcription_config.get("language", "es"))
                if segments:
                    print(f"  → [{AGENT_NAME}] Procesamiento exitoso utilizando el motor de IA: {engine_name}")
                    return {
                        "segments": segments, 
                        "transcribed_text": " ".join(s.get("text", "") for s in segments),
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