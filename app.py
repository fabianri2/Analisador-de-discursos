"""Capa de Presentación Web (Streamlit) bajo Clean Architecture.

Integra la maqueta de interfaz en la arquitectura limpia del sistema:
Presentación (app.py) -> Aplicación (AnalyzeSpeechUseCase) -> Adaptadores -> Infraestructura.
"""
from __future__ import annotations

import json
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
        st.dataframe(rows, width="stretch", hide_index=True)


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
        st.dataframe(
            [
                {
                    "Paso": step.get("step"),
                    "Afirmación": str(step.get("claim", ""))[:120],
                    "Resolución": step.get("resolution") or step.get("status"),
                    "Fuente": step.get("source") or "—",
                }
                for step in result.plan
            ],
            width="stretch",
            hide_index=True,
        )


def _render_chromadb_and_omissions(result: AnalysisExecutionResult) -> None:
    """Muestra las omisiones sectoriales y el contraste político recuperado de ChromaDB."""
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
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Perspectiva Izquierda / Crítica**")
                for item in result.left_wing_evidence:
                    st.caption(f"{item.get('source_channel', 'Fuente')} · {item.get('related_topic', 'General')}")
                    st.write(f"“{item.get('excerpt', '')}”")
            with c_right:
                st.markdown("**Perspectiva Derecha / Oficialista**")
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
            width="stretch",
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
                width="stretch",
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
    speaker = st.sidebar.text_input("Nombre del Orador/Presidente", value="Dina Boluarte")
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

    st.write("### 📁 Cargar Archivo Multimedia o Seleccionar del Corpus")
    source_mode = st.radio(
        "Origen del discurso",
        options=["📂 Corpus local (videos/)", "⬆️ Subir archivo (.mp4 / .srt)"],
        horizontal=True,
    )

    selected_input_path: str | None = None
    uploaded_payload: tuple[str, bytes] | None = None

    if source_mode == "⬆️ Subir archivo (.mp4 / .srt)":
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

    can_run = uploaded_payload is not None or selected_input_path is not None

    if can_run and st.button("🚀 Iniciar Auditoría Forense Computacional", type="primary"):
        with st.status("Ejecutando flujo bajo Clean Architecture...", expanded=True) as status:
            st.write(f"1️⃣ Preparando entrada para arquitectura **{architecture}**...")
            target_path = (
                persist_uploaded_media(uploaded_payload[0], uploaded_payload[1])
                if uploaded_payload is not None
                else str(selected_input_path)
            )
            st.write("2️⃣ Invocando Caso de Uso (`AnalyzeSpeechUseCase`) y agentes especialistas...")
            try:
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
                status.update(label="✅ ¡Auditoría Finalizada con Éxito!", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Error durante la auditoría", state="error", expanded=True)
                st.error(f"No se pudo completar el análisis ({type(exc).__name__}): {exc}")

    if "last_analysis_result" in st.session_state:
        st.markdown("---")
        _render_results(st.session_state["last_analysis_result"])


if __name__ == "__main__":
    main()
