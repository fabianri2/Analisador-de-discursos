"""Capa de Presentación Web (Streamlit) bajo Clean Architecture.

Integra la maqueta de interfaz en la arquitectura limpia del sistema:
Presentación (app.py) -> Aplicación (AnalyzeSpeechUseCase) -> Adaptadores -> Infraestructura.
"""
from __future__ import annotations

import json
import re
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_youtube_id(url_or_id: str) -> str | None:
    """Extrae el ID de 11 caracteres desde cualquier URL o ID de YouTube."""
    match = re.search(r"(?:v=|youtu\.be\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id.strip())
    if match:
        return match.group(1)
    cleaned = url_or_id.strip()
    return cleaned if re.fullmatch(r"[0-9A-Za-z_-]{11}", cleaned) else None


def _seconds_to_srt_time(seconds: float) -> str:
    """Convierte segundos flotantes al formato de tiempo estándar SRT (HH:MM:SS,mmm)."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _fetch_youtube_as_srt_bytes(url_or_id: str) -> tuple[str, bytes]:
    """Descarga los subtítulos de YouTube y agrupa fragmentos cortos en bloques con contexto completo."""
    video_id = _extract_youtube_id(url_or_id)
    if not video_id:
        raise ValueError("El enlace de YouTube no tiene un ID de video válido.")

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["es", "es-419", "es-PE", "en"])
    except Exception as exc:
        err_msg = str(exc)
        if "RequestBlocked" in type(exc).__name__ or "blocking requests" in err_msg:
            raise RuntimeError(
                "YouTube bloqueó temporalmente la descarga directa desde la IP del servidor en la nube. "
                "Para auditar este discurso ahora mismo: selecciona '📂 Corpus local (videos/)' "
                "o sube su archivo de subtítulos (.srt) en '⬆️ Subir archivo (.mp4 / .srt)'."
            ) from exc
        raise

    # Agrupar subtítulos cortos (~3s) en bloques coherentes de ~25s o hasta signo de puntuación
    grouped_blocks: list[tuple[float, float, str]] = []
    buf_texts: list[str] = []
    block_start: float | None = None
    block_end: float = 0.0

    for entry in transcript:
        start_sec = float(getattr(entry, "start", 0.0) if not isinstance(entry, dict) else entry.get("start", 0.0))
        dur_sec = float(getattr(entry, "duration", 3.0) if not isinstance(entry, dict) else entry.get("duration", 3.0))
        raw_text = str(getattr(entry, "text", "") if not isinstance(entry, dict) else entry.get("text", ""))
        clean_text = re.sub(r"\[.*?\]", "", raw_text).replace("\n", " ").strip()
        if not clean_text:
            continue

        if block_start is None:
            block_start = start_sec
        buf_texts.append(clean_text)
        block_end = start_sec + max(dur_sec, 1.0)

        elapsed = block_end - block_start
        ends_sentence = clean_text.endswith((".", "!", "?"))
        if (elapsed >= 22.0 and ends_sentence) or elapsed >= 30.0 or len(" ".join(buf_texts)) >= 320:
            grouped_blocks.append((block_start, block_end, " ".join(buf_texts)))
            buf_texts = []
            block_start = None

    if buf_texts and block_start is not None:
        grouped_blocks.append((block_start, block_end, " ".join(buf_texts)))

    srt_lines = [
        f"{idx}\n{_seconds_to_srt_time(s)} --> {_seconds_to_srt_time(e)}\n{txt}\n"
        for idx, (s, e, txt) in enumerate(grouped_blocks, start=1)
    ]

    if not srt_lines:
        raise ValueError("El video no contiene subtítulos disponibles para extraer.")

    return f"youtube_{video_id}.srt", "\n".join(srt_lines).encode("utf-8")


import streamlit as st
import streamlit.components.v1 as components

from src.adapters.analysis_gateway import (
    create_speech_analysis_use_case,
    list_corpus_media_files,
    persist_uploaded_media,
)
from src.domain.entities import AlgorithmMode, AnalysisExecutionResult, AnalysisRequest

ARCHITECTURE_OPTIONS = {
    "supervisor": "Supervisor Multi-Agente (Clean Architecture — Hub & Spoke)",
    "pipeline": "Pipeline Paralelo / Secuencial",
    "plan-and-execute": "Plan-and-Execute (Planner / Executor / Replanner)",
}

ALGORITHM_MODES = {
    "Modo Heurístico Documental": AlgorithmMode.HEURISTIC,
    "Clasificador Machine Learning": AlgorithmMode.ML,
}


def _render_supervisor_panel(result: AnalysisExecutionResult) -> None:
    """Despliega las métricas y la bitácora ciclo por ciclo del Agente Supervisor."""
    if not result.supervisor_trace and not result.supervisor_metrics:
        return

    metrics = result.supervisor_metrics or {}
    trace = result.supervisor_trace or []
    workers_called = metrics.get("workers_called") or [
        step.get("assigned_worker")
        for step in trace
        if step.get("assigned_worker") and step.get("assigned_worker") != "FINISH"
    ]
    total_cycles = metrics.get("total_cycles") or len(trace)
    criticals = metrics.get("critical_discrepancies_detected", 0)

    st.subheader("🧭 Gobernanza del Agente Supervisor (Clean Architecture)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ciclos de control", total_cycles)
    m2.metric("Especialistas convocados", len(workers_called))
    m3.metric("Alertas críticas", criticals)
    m4.metric("Topología", "Hub & Spoke")

    if trace:
        rows = [
            {
                "Ciclo": step.get("cycle"),
                "Fase": step.get("phase"),
                "Especialista": step.get("assigned_worker"),
                "Alerta": "⚠ CRÍTICA" if step.get("critical_alert") else "Normal",
                "Motivo / Directiva": step.get("rationale"),
            }
            for step in trace
        ]
        st.dataframe(rows)


def _render_plan_panel(result: AnalysisExecutionResult) -> None:
    """Despliega las métricas del planner si se ejecutó la arquitectura Plan-and-Execute."""
    if not result.plan_metrics:
        return
    pm = result.plan_metrics
    st.subheader("🗺️ Plan de Verificación (Plan-and-Execute)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Pasos planificados", pm.get("planned_steps", 0))
    p2.metric("Pasos resueltos", pm.get("resolved_steps", 0))
    p3.metric("Replanificaciones", pm.get("replanned_steps", 0))
    p4.metric("Cobertura", f"{float(pm.get('coverage', 0.0)) * 100:.0f}%")
    if result.plan:
        st.dataframe()


def _render_chromadb_and_omissions(result: AnalysisExecutionResult) -> None:
    """Muestra auditoría RAG por sector (cumplimiento actual + línea de tiempo), omisiones y contraste político."""
    st.subheader("🏛️ Auditoría Factual RAG por Sector, Cumplimiento Actual y Línea de Tiempo")

    cards = result.sector_audit_cards or []
    if not cards:
        st.info("No se detectaron promesas sectoriales específicas en este fragmento del discurso.")
    else:
        available_sectors = ["Todos los sectores"] + sorted({c.get("sector", "General") for c in cards})
        selected_sector = st.selectbox(
            "Filtrar auditoría por sector gubernamental:",
            options=available_sectors,
            key="sector_filter_selectbox",
        )
        filtered_cards = (
            cards if selected_sector == "Todos los sectores"
            else [c for c in cards if c.get("sector") == selected_sector]
        )

        for card in filtered_cards:
            estado = card.get("estado_actual", "Parcial")
            if estado == "Cumplido":
                badge = "🟢 CUMPLIDO"
            elif estado == "Incumplido":
                badge = "🔴 INCUMPLIDO"
            else:
                badge = "🟡 CUMPLIMIENTO PARCIAL"

            with st.container():
                st.markdown("---")
                h_col1, h_col2 = st.columns([3, 1])
                h_col1.markdown(f"### 📌 Sector: {card.get('sector')} — {badge}")
                h_col2.metric("Discrepancia Factual", f"{card.get('discrepancy_score', 0):.1f}/100")

                st.markdown(f"**Afirmación en el discurso (seg. {card.get('timestamp_start', 0):.1f}s):** *“{card.get('promise_excerpt', '')}”*")
                st.markdown(f"**Evidencia Oficial Actual ({card.get('source', 'Fuente Oficial')}):** {card.get('evidencia_actual', '')}")

                timeline = card.get("timeline") or []
                if timeline:
                    st.markdown("**⏳ Línea de Tiempo Histórica del Compromiso:**")
                    t_cols = st.columns(len(timeline))
                    for idx, hito in enumerate(timeline):
                        partes = hito.split("|", 1)
                        fecha = partes[0].strip() if len(partes) > 1 else f"Hito {idx + 1}"
                        detalle = partes[1].strip() if len(partes) > 1 else hito
                        with t_cols[idx]:
                            st.info(f"**{fecha}**\n\n{detalle}")

    st.markdown("---")
    col_omit, col_spec = st.columns([1, 2])
    with col_omit:
        st.subheader("🚨 Omisiones Sectoriales")
        if result.omitted_topics:
            for topic in result.omitted_topics:
                st.warning(f"Sector omitido: {topic}")
        else:
            st.success("El discurso cubrió todos los sectores críticos monitoreados.")

    with col_spec:
        st.subheader("⚖️ Contraste ChromaDB (Izquierda vs. Derecha)")
        if not result.left_wing_evidence and not result.right_wing_evidence:
            st.info("No se recuperaron evidencias vectoriales adicionales en ChromaDB para este discurso.")
        else:
            st.markdown("**🔴 Perspectiva Izquierda / Crítica**")
            for item in result.left_wing_evidence:
                st.caption(f"{item.get('source_channel', 'Fuente')} · {item.get('related_topic', 'General')}")
                st.write(f"“{item.get('excerpt', '')}”")

            st.markdown("---")
            st.markdown("**🔵 Perspectiva Derecha / Oficialista**")
            for item in result.right_wing_evidence:
                st.caption(f"{item.get('source_channel', 'Fuente')} · {item.get('related_topic', 'General')}")
                st.write(f"“{item.get('excerpt', '')}”")



def _render_results(result: AnalysisExecutionResult) -> None:
    """Renderiza el tablero completo a partir de la entidad de dominio AnalysisExecutionResult."""
    verdict = result.verdict or {}
    label = str(verdict.get("label", "N/A"))
    opacity = verdict.get("opacity_index", 0.0)
    confidence = float(verdict.get("confidence", 0.0) or 0.0) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dictamen", label.upper())
    k2.metric("Índice IOF", f"{opacity:.1f}/100" if isinstance(opacity, (int, float)) else "n/d")
    k3.metric("Confianza", f"{confidence:.0f}%")
    k4.metric("Arquitectura", result.architecture)

    _render_supervisor_panel(result)
    _render_plan_panel(result)
    st.markdown("---")
    _render_chromadb_and_omissions(result)
    st.markdown("---")

    col1, col2 = st.columns([1, 1.35])

    with col1:
        st.subheader("📊 Estructura de Datos (result.json)")
        json_payload = result.report_json_data or {"verdict": verdict, "architecture": result.architecture}
        st.download_button(
            label="⬇️ Descargar result.json",
            data=json.dumps(json_payload, ensure_ascii=False, indent=2),
            file_name="result.json",
            mime="application/json",
        )
        st.json(json_payload, expanded=False)

    with col2:
        st.subheader("🖥️ Vista Previa del Dashboard (report.html)")
        if result.report_html_content:
            st.download_button(
                label="⬇️ Descargar report.html",
                data=result.report_html_content,
                file_name="report.html",
                mime="text/html",
            )
            components.html(result.report_html_content, height=820, scrolling=True)
        else:
            st.warning("No se encontró el archivo report.html generado en disco.")

    if result.errors:
        with st.expander("⚠️ Avisos de degradación controlada"):
            for err in result.errors:
                st.write(f"- {err}")



def main() -> None:
    st.set_page_config(
        page_title="Auditoría Forense de Discursos",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("🎙️ Analizador Agéntico Multiagente de Discursos Presidenciales")
    st.caption("Interfaz Web bajo Clean Architecture · Orquestación Multi-Agente, ChromaDB RAG y Reporte Forense")
    st.markdown("---")

    st.sidebar.header("⚙️ Configuración")
    speaker = st.sidebar.text_input(
        "Nombre del Orador / Presidente (opcional)",
        value="",
        placeholder="Ej. Francisco Sagasti, Martín Vizcarra, Dina Boluarte...",
    )
    architecture = st.sidebar.selectbox(
        "Arquitectura Agéntica",
        options=list(ARCHITECTURE_OPTIONS.keys()),
        format_func=lambda key: ARCHITECTURE_OPTIONS[key],
        index=0,
    )
    algo_label = st.sidebar.selectbox(
        "Criterio Algorítmico",
        options=list(ALGORITHM_MODES.keys()),
        index=0,
    )
    st.sidebar.info(
        "Capas Clean Architecture activas:\n"
        "`Presentación (Streamlit)` → `Aplicación (UseCase)` → `Adaptadores` → `Infraestructura (LangGraph)`"
    )

    st.write("### 📁 Cargar Archivo Multimedia, YouTube o Corpus")
    source_mode = st.radio(
        "Origen del discurso",
        options=[
            "🌐 Enlace de YouTube (Discursos largos)",
            "📂 Corpus local (videos/)",
            "⬆️ Subir archivo (.mp4 / .srt)",
        ],
    )

    selected_input_path: str | None = None
    uploaded_payload: tuple[str, bytes] | None = None
    youtube_url: str = ""

    if source_mode == "🌐 Enlace de YouTube (Discursos largos)":
        youtube_url = st.text_input(
            "Pega el enlace del video presidencial en YouTube",
            placeholder="https://youtube.com...",
        )
        if youtube_url.strip():
            st.caption("✅ Se extraerán automáticamente los subtítulos completos del discurso sin límite de duración.")
    elif source_mode == "⬆️ Subir archivo (.mp4 / .srt)":
        uploaded_file = st.file_uploader(
            "Selecciona un video presidencial (.mp4, .mkv, .webm) o subtítulos (.srt, .vtt)",
            type=["mp4", "mkv", "mov", "avi", "webm", "srt", "vtt"],
        )
        if uploaded_file is not None:
            st.info(f"Archivo cargado exitosamente: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            uploaded_payload = (uploaded_file.name, uploaded_file.getvalue())
    else:
        corpus_files = list_corpus_media_files("videos")
        if corpus_files:
            default_idx = next(
                (i for i, path in enumerate(corpus_files) if "Dina Boluarte" in path),
                0,
            )
            selected_input_path = st.selectbox("Archivos disponibles en videos/", options=corpus_files, index=default_idx)
        else:
            st.warning("No hay archivos en la carpeta `videos/`.")

    can_run = (
        (source_mode == "🌐 Enlace de YouTube (Discursos largos)" and bool(youtube_url.strip()))
        or uploaded_payload is not None
        or selected_input_path is not None
    )

    if can_run and st.button("🚀 Iniciar Auditoría Forense Computacional"):
        with st.spinner("Ejecutando flujo bajo Clean Architecture..."):
            try:
                st.write(f"1️⃣ Preparando entrada para arquitectura **{architecture}**...")
                if source_mode == "🌐 Enlace de YouTube (Discursos largos)":
                    yt_filename, yt_bytes = _fetch_youtube_as_srt_bytes(youtube_url)
                    target_path = persist_uploaded_media(yt_filename, yt_bytes)
                elif uploaded_payload is not None:
                    target_path = persist_uploaded_media(uploaded_payload[0], uploaded_payload[1])
                else:
                    target_path = str(selected_input_path)

                st.write("2️⃣ Invocando Caso de Uso (`AnalyzeSpeechUseCase`) y agentes especialistas...")
                use_case = create_speech_analysis_use_case()
                request = AnalysisRequest(
                    video_path=target_path,
                    speaker=speaker.strip() or None,
                    architecture=architecture,
                    output_dir="results",
                    algorithm_mode=ALGORITHM_MODES[algo_label],
                    verbose=False,
                )
                result = use_case.execute(request)
                st.session_state["last_analysis_result"] = result
                st.success("✅ ¡Auditoría Finalizada con Éxito!")
            except Exception as exc:
                st.error(f"No se pudo completar el análisis ({type(exc).__name__}): {exc}")

    if "last_analysis_result" in st.session_state:
        st.markdown("---")
        _render_results(st.session_state["last_analysis_result"])


if __name__ == "__main__":
    main()
