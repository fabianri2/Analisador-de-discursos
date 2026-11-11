"""Agente Reportero: genera el informe JSON y un reporte HTML autocontenido."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np 
from ..state import SpeechAnalysisState
from ..utilities import round_nested_data

AGENT_NAME = "Reportero"

METRIC_WARNING = (
    "Este reporte es una auditoría algorítmica de consistencia factual, presupuestal y de contraste mediático. "
    "Un índice de opacidad o desviación elevado indica que las afirmaciones macroeconómicas, "
    "jurídicas o sociales del ejecutivo presentan discrepancias estadísticas frente a las "
    "memorias anuales del BCRP, partidas del MEF y fuentes periodísticas contrastadas en ChromaDB. "
    "Los resultados sirven de apoyo para que el ciudadano saque sus propias conclusiones."
)

FAMILY_LABELS = {
    "sentiment": "Sentimiento Político", 
    "linguistic": "Densidad Lingüística", 
    "syntactic": "Opacidad Sintáctica",
    "promise_extraction": "Extracción de Promesas",
    "factual_rag": "Auditoría Factual (RAG)"
}


def _convert_seconds_to_mmss(timestamp_seconds: float) -> str:
    """Formats float numerical seconds into a standard MM:SS string."""
    return f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}"


def _resolve_real_duration(state: Any, data_rows: List[Dict]) -> float:
    """Computes actual speech duration from segments if metadata duration is missing or default."""
    raw_duration = float(state.get("duration", 0.0)) if isinstance(state, dict) else 0.0
    if raw_duration > 0.0 and raw_duration != 72.0:
        return raw_duration
    segments = state.get("segments", []) if isinstance(state, dict) else []
    if segments:
        max_end = max(float(s.get("end", 0.0)) for s in segments)
        if max_end > 0.0:
            return max_end
    if data_rows:
        max_row_end = max(float(r.get("end", 0.0)) for r in data_rows)
        if max_row_end > 0.0:
            return max_row_end
    return 72.0


def _generate_timeline_svg(data_rows: List[Dict], total_duration: float, truth_threshold: float, deviation_threshold: float) -> str:
    """Generates a responsive visual SVG timeline grid mapping speech deviation points."""
    W, H, pad = 900, 220, 36
    duration = max(total_duration, 1e-6)
    
    get_x = lambda t: pad + (W - 2 * pad) * t / duration
    get_y = lambda v: H - pad - (H - 2 * pad) * v / 100
    
    svg_bars = []
    for row in data_rows:
        index_value = row.get("index")
        if index_value is None or not np.isfinite(index_value):
            continue
            
        factual_val = row.get("families", {}).get("factual_rag", 0.0)
        visual_score = max(index_value, factual_val if factual_val >= 60.0 else index_value)
        bar_class = "b-alto" if visual_score > deviation_threshold else ("b-medio" if visual_score >= truth_threshold else "b-bajo")
        
        x0 = get_x(row.get("start", 0.0))
        x1 = get_x(row.get("end", 0.0))
        
        bar_height = max(5.0, get_y(0) - get_y(visual_score))
        bar_y = get_y(visual_score)
        svg_bars.append(
            f'<rect class="{bar_class}" x="{x0:.1f}" y="{bar_y:.1f}" '
            f'width="{max(2.0, x1 - x0 - 1):.1f}" height="{bar_height:.1f}">'
            f'<title>Seg. {row["id"]} ({_convert_seconds_to_mmss(row.get("start", 0.0))}): Índice {index_value:.0f}/100 | Factual {factual_val:.0f}/100</title></rect>'
        )
        
    timeline_markers = "".join(
        f'<text class="eje" x="{get_x(t):.0f}" y="{H - 12}" text-anchor="middle">{_convert_seconds_to_mmss(t)}</text>'
        for t in np.linspace(0, duration, 6)
    )
    
    return f"""<svg viewBox="0 0 {W} {H}" role="img" aria-label="Índice de desviación por segmento">
<line class="umbral" x1="{pad}" x2="{W - pad}" y1="{get_y(deviation_threshold):.1f}" y2="{get_y(deviation_threshold):.1f}"/>
<line class="umbral" x1="{pad}" x2="{W - pad}" y1="{get_y(truth_threshold):.1f}" y2="{get_y(truth_threshold):.1f}"/>
<text class="eje" x="{pad - 6}" y="{get_y(deviation_threshold) + 4:.1f}" text-anchor="end">{deviation_threshold:.0f}</text>
<text class="eje" x="{pad - 6}" y="{get_y(truth_threshold) + 4:.1f}" text-anchor="end">{truth_threshold:.0f}</text>
<line class="base" x1="{pad}" x2="{W - pad}" y1="{get_y(0):.1f}" y2="{get_y(0):.1f}"/>
{"".join(svg_bars)}{timeline_markers}</svg>"""


def _build_political_spectrum_section(state: Any) -> str:
    """Renders side-by-side ChromaDB evidence contrasting left-wing and right-wing sources."""
    if not isinstance(state, dict):
        return ""
    rag_metrics = state.get("factual_rag_metrics") or {}
    left_evidence = state.get("left_wing_evidence") or rag_metrics.get("left_wing_evidence") or []
    right_evidence = state.get("right_wing_evidence") or rag_metrics.get("right_wing_evidence") or []


    if not left_evidence and not right_evidence:
        return """<section class="card">
    <h2>Contraste de Espectro Político (ChromaDB Web RAG)</h2>
    <p class="mut">No se recuperaron posturas externas indexadas en ChromaDB para los temas de este discurso.</p>
</section>"""

    left_cards = "".join(
        f'<div class="spec-item"><span class="badge-izq">{html.escape(str(item.get("source_channel", "Fuente Crítica")))}</span>'
        f'<div class="t" style="margin-top:4px">Tema: <b>{html.escape(str(item.get("related_topic", "General")))}</b> · Relevancia: {float(item.get("similarity_score", 0))*100:.0f}%</div>'
        f'<p style="margin:6px 0 0">“{html.escape(str(item.get("excerpt", "")))}”</p></div>'
        for item in left_evidence
    ) or '<p class="mut">Sin registros en el espectro de izquierda.</p>'

    right_cards = "".join(
        f'<div class="spec-item"><span class="badge-der">{html.escape(str(item.get("source_channel", "Prensa Tradicional")))}</span>'
        f'<div class="t" style="margin-top:4px">Tema: <b>{html.escape(str(item.get("related_topic", "General")))}</b> · Relevancia: {float(item.get("similarity_score", 0))*100:.0f}%</div>'
        f'<p style="margin:6px 0 0">“{html.escape(str(item.get("excerpt", "")))}”</p></div>'
        for item in right_evidence
    ) or '<p class="mut">Sin registros en el espectro de derecha.</p>'

    return f"""<section class="card">
    <h2>Contraste de Espectro Político (ChromaDB: Izquierda vs. Derecha)</h2>
    <p class="mut" style="margin:0 0 12px">Recuperación vectorial cruzada de medios independientes, análisis en YouTube y prensa tradicional para contraste ciudadano.</p>
    <div class="spectrum-grid">
        <div class="spectrum-col">
            <h3 style="margin:0 0 8px;font-size:15px;color:var(--bad)">Perspectiva Izquierda / Oposición Crítica</h3>
            {left_cards}
        </div>
        <div class="spectrum-col">
            <h3 style="margin:0 0 8px;font-size:15px;color:var(--ac)">Perspectiva Derecha / Prensa Oficialista</h3>
            {right_cards}
        </div>
    </div>
</section>"""


def _build_omitted_sectors_section(state: Any) -> str:
    """Renders critical national sectors omitted across the entire speech."""
    if not isinstance(state, dict):
        return ""
    rag_metrics = state.get("factual_rag_metrics") or {}
    omitted = state.get("omitted_topics") or rag_metrics.get("omitted_topics") or []

    if not omitted:
        return """<section class="card">
    <h2>Auditoría de Cobertura y Omisiones Sectoriales</h2>
    <p class="mut" style="margin:0">El discurso abordó los sectores críticos monitoreados (Salud, Regiones Andinas, Educación y Agro).</p>
</section>"""

    pills = "".join(f'<span class="omit-pill">⚠ {html.escape(str(topic))}</span>' for topic in omitted)
    return f"""<section class="card aviso" style="border-left-color:var(--bad)">
    <h2>Auditoría de Omisiones Críticas en el Discurso</h2>
    <p class="mut" style="margin:0 0 10px">Los siguientes sectores prioritarios del país <b>no fueron mencionados</b> en ningún segmento de la intervención:</p>
    <div style="display:flex;flex-wrap:wrap;gap:8px">{pills}</div>
</section>"""
def _compile_html_template(state: Any, verdict: Dict[str, Any]) -> str:
    """Assembles the complete forensic static HTML report interface using unified English keys."""
    try:
        judge_config = state["config"]["judge"]
    except Exception:
        judge_config = {"truth_threshold": 40.0, "deviation_threshold": 65.0}
        
    data_rows = verdict.get("by_segment", [])
    speech_segments = {seg["id"]: seg for seg in state.get("segments", [])} if isinstance(state, dict) and "segments" in state else {}
    
    consistency_index = verdict.get("opacity_index", 0.0)
    deviation_probability = verdict.get("deviation_probability")
    
    categorical_label = verdict.get("label", "neutral")
    ui_theme = {"truth": "verdad", "deviation": "engano"}.get(categorical_label, "neutro")
    ui_title_label = {
        "truth": "Consistencia Factual (Truth)",
        "deviation": "Desviación Crítica Detectada (Deviation)",
        "neutral": "Discrepancia Moderada / Alertas Factuales"
    }.get(categorical_label, "Discrepancia Moderada / Alertas Factuales")

    speech_segments_list = state.get("segments", []) if isinstance(state, dict) else []
    transcription_missing = bool(verdict.get("transcription_missing")) or not any(
        str(segment.get("text", "")).strip() for segment in speech_segments_list
    )
    if transcription_missing:
        ui_theme = "neutro"
        ui_title_label = "Sin transcripción: contenido no auditable"
        display_value = "—"
    else:
        display_value = f"{consistency_index:.0f}%" if consistency_index >= 1.0 else (f"{deviation_probability * 100:.0f}%" if deviation_probability else "0%")

    # Detección dinámica del método real usado por el juez (evita texto estático erróneo)
    classifier_method = html.escape(str(verdict.get("method", "Algoritmo Heurístico Agéntico Multi-Modal")))

    # Componente: Promedio por familia de agentes
    family_metrics_html = []
    for family_key, label_name in FAMILY_LABELS.items():
        family_values = [row["families"][family_key] for row in data_rows if "families" in row and family_key in row["families"]]
        if family_values:
            mean_value = float(np.mean(family_values))
            family_metrics_html.append(
                f'<div class="fam"><span>{label_name}</span><div class="barra">'
                f'<i style="width:{mean_value:.0f}%"></i></div><b>{mean_value:.0f}</b></div>'
            )

    # Componente: un hallazgo por tema regulatorio
    critical_segments_html = []
    critical_findings = verdict.get("critical_findings") or []
    critical_topics_by_segment = {finding["segment_id"]: finding for finding in critical_findings}

    if critical_findings:
        for finding in critical_findings:
            segment_data = speech_segments.get(finding["segment_id"])
            if not segment_data:
                continue
            escaped_text = html.escape(segment_data.get("text", "(Sin texto transcribir)"))
            det_note = html.escape(finding.get("audit_note") or (
                "Alerta Forense: Discrepancia crítica frente a la evidencia documental oficial "
                "de entidades del Estado (MEF/BCRP/JNE)."
            ))
            mentions = int(finding.get("mentions", 1) or 1)
            repeat_badge = (
                f'<span class="rep">Mismo tema en {mentions} segmentos</span>' if mentions > 1 else ""
            )
            source_short = (finding.get("source") or "MEF/BCRP/JNE").split("(")[0].strip()

            critical_segments_html.append(
                f'<li><div class="t">{_convert_seconds_to_mmss(finding.get("start", 0.0))}–'
                f'{_convert_seconds_to_mmss(finding.get("end", 0.0))} · '
                f'<b class="tema">{html.escape(finding.get("topic", "Discrepancia factual"))}</b> '
                f'{repeat_badge}</div>'
                f'<div class="t">Discrepancia Factual: <b style="color:var(--bad)">'
                f'{finding.get("factual_score", 0):.0f}/100</b> · '
                f'Índice Compuesto: {finding.get("index", 0):.0f}/100</div>'
                f'<p>“{escaped_text}”</p>'
                f'<p class="det" style="color:var(--bad);margin:4px 0 0"><b>Hallazgo RAG ({html.escape(source_short)}):</b> {det_note}</p></li>'
            )
    else:
        for segment_id in verdict.get("critical_segments", []):
            row_data = next((x for x in data_rows if x["id"] == segment_id), None)
            segment_data = speech_segments.get(segment_id) if speech_segments else None

            if row_data and segment_data:
                escaped_text = html.escape(segment_data.get("text", "(Sin texto transcribir)"))
                det_note = ""
                for det in row_data.get("detonantes", []):
                    if det.get("note"):
                        det_note = html.escape(det["note"])
                        break
                if not det_note:
                    det_note = "Alerta Forense: Discrepancia crítica frente a la evidencia documental oficial de entidades del Estado (MEF/BCRP/JNE)."

                factual_val = row_data.get("families", {}).get("factual_rag", row_data.get("index", 0.0))
                composite_val = row_data.get("index", 0.0)

                critical_segments_html.append(
                    f'<li><div class="t">{_convert_seconds_to_mmss(segment_data.get("start", 0.0))}–{_convert_seconds_to_mmss(segment_data.get("end", 0.0))} · '
                    f'<b style="color:var(--bad)">Discrepancia Factual: {factual_val:.0f}/100</b> · '
                    f'<span class="mut">Índice Compuesto: {composite_val:.0f}/100</span></div><p>“{escaped_text}”</p>'
                    f'<p class="det" style="color:var(--bad);margin:4px 0 0"><b>Hallazgo RAG (MEF/BCRP/JNE):</b> {det_note}</p></li>'
                )

    # Componente: Tabla secuencial completa
    table_rows_html = []
    for row in data_rows:
        segment_data = speech_segments.get(row["id"]) if speech_segments else None
        idx_val = row.get("index", 0.0)
        row_class = "alto" if idx_val > judge_config.get("deviation_threshold", 65.0) else ""
        display_idx = f"{idx_val:.0f}" if np.isfinite(idx_val) else "–"

        critical_finding = critical_topics_by_segment.get(row["id"])
        if critical_finding:
            finding_cell = f'<span class="tema">{html.escape(critical_finding["topic"])}</span>'
        elif row.get("topic") and row.get("factual_primary") is False:
            finding_cell = f'<span class="mut">Reitera: {html.escape(row["topic"])}</span>'
        else:
            finding_cell = '<span class="mut">—</span>'

        if segment_data:
            escaped_text = html.escape(segment_data.get("text", ""))
            time_str = _convert_seconds_to_mmss(segment_data.get("start", 0.0))
        else:
            escaped_text = "(Análisis de Ventana Estática / Entrada Documental)"
            time_str = _convert_seconds_to_mmss(row.get("start", 0.0))

        table_rows_html.append(
            f'<tr class="{row_class}"><td>{time_str}</td><td>{display_idx}</td>'
            f'<td>{finding_cell}</td><td>{escaped_text}</td></tr>'
        )

    # Componente: plan de verificación
    plan_metrics = state.get("plan_metrics") if isinstance(state, dict) else None
    plan_section_html = ""
    if plan_metrics:
        plan_rows_html = "".join(
            f'<tr><td>{step.get("step")}</td>'
            f'<td>{html.escape(str(step.get("claim", ""))[:160])}</td>'
            f'<td>{html.escape(str(step.get("resolution") or step.get("status")))}</td>'
            f'<td>{html.escape(str(step.get("source") or "—"))}</td></tr>'
            for step in state.get("plan", [])
        )
        coverage = float(plan_metrics.get("coverage", 0.0)) * 100
        plan_section_html = f"""<section class="card">
    <h2>Plan de verificación (Plan-and-Execute)</h2>
    <p class="mut" style="margin:0 0 6px">El planner descompuso el discurso en {plan_metrics.get('planned_steps', 0)} pasos;
    se resolvieron {plan_metrics.get('resolved_steps', 0)} con evidencia documental
    ({plan_metrics.get('replanned_steps', 0)} por replanificación), cobertura {coverage:.0f}%.</p>
    <table><thead><tr><th>#</th><th>Afirmación</th><th>Resultado</th><th>Fuente</th></tr></thead>
    <tbody>{plan_rows_html}</tbody></table>
</section>
"""

    raw_logs = state.get("logs", []) if isinstance(state, dict) else []
    formatted_logs = []
    for entry in raw_logs:
        text_entry = str(entry)
        if text_entry.strip().endswith("0.00s"):
            continue
        escaped_entry = html.escape(text_entry)
        if text_entry.startswith("[Supervisor"):
            styled = f'<span class="log-sup">{escaped_entry}</span>'
        elif any(tag in text_entry for tag in ["[FactualAlignment]", "[Juez]", "[PromiseExtractor]", "[Transcription]", "[Linguistic]", "[Syntactic]"]):
            styled = f'<span class="log-agent">{escaped_entry}</span>'
        else:
            styled = escaped_entry
        formatted_logs.append(f'<li class="log-item"><span>{styled}</span></li>')
    execution_logs = "".join(formatted_logs)
    execution_errors = "".join(f"<li>{html.escape(err)}</li>" for err in state.get("errors", [])) if isinstance(state, dict) else ""

    
    speaker_name = html.escape(state.get("speaker", "Desconocido")) if isinstance(state, dict) else "Desconocido"
    video_name_display = "discurso.mp4"
    if isinstance(state, dict) and "video" in state:
        video_name_display = Path(state["video"]).name
        
    confidence_text = f"{verdict.get('confidence', 0.9) * 100:.0f}%"
    video_duration = _resolve_real_duration(state, data_rows)

    missing_banner_html = ""
    if transcription_missing:
        missing_banner_html = (
            '<section class="card aviso" style="border-left-color:var(--bad)">'
            '<b>Sin transcripción:</b> <span class="mut">No se pudo convertir el audio a texto, '
            'así que el índice y el dictamen <b>no evalúan el contenido</b> del discurso.</span></section>'
        )

    omitted_section_html = _build_omitted_sectors_section(state)
    spectrum_section_html = _build_political_spectrum_section(state)

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auditoría Forense de Discurso – {speaker_name}</title><style>
:root{{--bg:#f7f6f2;--card:#fff;--tx:#1d1d1b;--mut:#6b6a65;--ln:#e4e2db;--ac:#1f4e79;
--ok:#2e7d4f;--mid:#b7791f;--bad:#b23a2e;--base:#a9a79f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--card:#1f2125;--tx:#ecebe6;--mut:#a3a29c;
--ln:#33353a;--ac:#7fb0dd;--ok:#5fbf86;--mid:#e0a84a;--bad:#e2705f;--base:#5d5f63}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:960px;margin:0 auto;padding:28px 16px 60px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:17px;margin:0 0 12px}}
.mut{{color:var(--mut);font-size:13px}}
.card{{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:20px;margin-top:16px}}
.hero{{display:grid;grid-template-columns:auto 1fr;gap:24px;align-items:center}}
.cifra{{font-size:54px;font-weight:700;line-height:1}}
.etq{{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;margin-top:8px}}
.verdad{{color:var(--ok)}}.engano{{color:var(--bad)}}.neutro{{color:var(--mid)}}
.fam{{display:grid;grid-template-columns:180px 1fr 36px;gap:10px;align-items:center;margin:8px 0}}
.barra{{height:8px;background:var(--ln);border-radius:4px;overflow:hidden}}
.barra i{{display:block;height:100%;background:var(--ac)}}
svg{{width:100%;height:auto}}.eje{{font-size:11px;fill:var(--mut)}}
.umbral{{stroke:var(--mut);stroke-dasharray:4 4;stroke-width:1}}.base{{stroke:var(--ln)}}
.b-bajo{{fill:var(--ok)}}.b-medio{{fill:var(--mid)}}.b-alto{{fill:var(--bad)}}
.ley span{{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:12px;color:var(--mut)}}
.ley i{{width:10px;height:10px;border-radius:2px;display:inline-block}}
table{{width:100%;border-collapse:collapse;font-size:14px;margin-top:12px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid var(--ln)}}
tr.alto td{{background:color-mix(in srgb,var(--bad) 9%,transparent)}}
ul{{padding-left:20px}}li{{margin-bottom:12px}}.t{{font-size:13px;color:var(--mut)}}
.tema{{color:var(--ac)}}.rep{{display:inline-block;margin-left:6px;padding:1px 8px;border-radius:999px;
font-size:11px;background:color-mix(in srgb,var(--mid) 18%,transparent);color:var(--mid)}}
.aviso{{border-left:4px solid var(--mid)}}
.spectrum-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.spectrum-col{{background:var(--bg);padding:14px;border-radius:8px;border:1px solid var(--ln)}}
.spec-item{{padding-bottom:10px;margin-bottom:10px;border-bottom:1px solid var(--ln)}}
.spec-item:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0}}
.badge-izq{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}}
.badge-der{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:color-mix(in srgb,var(--ac) 15%,transparent);color:var(--ac)}}
.omit-pill{{display:inline-block;padding:5px 12px;border-radius:999px;font-size:13px;font-weight:600;background:color-mix(in srgb,var(--bad) 14%,transparent);color:var(--bad)}}
.log-list{{list-style:none;padding:0;margin:12px 0 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}}
.log-item{{padding:6px 10px;border-bottom:1px solid var(--ln);display:flex;justify-content:space-between;gap:12px;align-items:baseline}}
.log-item:last-child{{border-bottom:none}}
.log-sup{{color:var(--ac);font-weight:600}}
.log-agent{{color:var(--ok);font-weight:600}}
.log-time{{color:var(--mut);font-size:11px;white-space:nowrap}}
details summary{{cursor:pointer;font-weight:600}}
@media (max-width:700px){{.hero,.spectrum-grid{{grid-template-columns:1fr}}.cifra{{font-size:44px}}}}
</style></head><body><main>
<h1>Reporte Forense Agéntico: {speaker_name}</h1>
<div class="mut">Archivo original: {html.escape(video_name_display)} · Duración real: {_convert_seconds_to_mmss(video_duration)} · Ejecutado el {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>

<section class="card hero">
    <div class="cifra {ui_theme}">{display_value}</div>
    <div>
        <h2>Dictamen General: <span class="{ui_theme}">{ui_title_label}</span></h2>
        <p class="mut" style="margin:0">Confianza del sistema: {confidence_text} · Método: {classifier_method}</p>
    </div>
</section>
{missing_banner_html}
<section class="card aviso">
    <b>Metodología de Lectura Regulatoria:</b> <span class="mut">{METRIC_WARNING}</span>
</section>

{omitted_section_html}

{spectrum_section_html}

<section class="card">
    <h2>Índice Medio por Familias de Agentes</h2>
    {"".join(family_metrics_html) if family_metrics_html else '<p class="mut">Sin datos registrados</p>'}
</section>

<section class="card">
    <h2>Línea de Tiempo de Consistencia Factual</h2>
    {_generate_timeline_svg(data_rows, video_duration, judge_config['truth_threshold'], judge_config['deviation_threshold'])}
    <div class="ley" style="margin-top:10px">
        <span><i style="background:var(--ok)"></i>Consistente</span>
        <span><i style="background:var(--mid)"></i>Desviación Moderada</span>
        <span><i style="background:var(--bad)"></i>Desviación Crítica</span>
    </div>
</section>

{plan_section_html}
<section class="card">
    <h2>Segmentos con Discrepancias Críticas (Cruce MEF/BCRP)</h2>
    <p class="mut" style="margin:0 0 6px">Se muestra un solo hallazgo por tema regulatorio: si la misma promesa se repite en varios segmentos, se agrupa en la primera mención.</p>
    {f"<ul>{''.join(critical_segments_html)}</ul>" if critical_segments_html else "<p class='mut'>Ningún segmento superó el umbral de desvío crítico de datos duros.</p>"}
</section>

<section class="card">
    <h2>Transcripción y Mapeo Secuencial de Opacidad</h2>
    <table><thead><tr><th>Tiempo</th><th>Índice</th><th>Hallazgo</th><th>Texto del Segmento</th></tr></thead>
    <tbody>{"".join(table_rows_html)}</tbody></table>
</section>

<section class="card">
    <details><summary>Bitácora de Ejecución del Sistema (LangGraph Logs)</summary>
    <ul class="log-list">{execution_logs}</ul>
    {f'<h3>Alertas de Degradación Controlada</h3><ul>{execution_errors}</ul>' if execution_errors else ''}
    </details>
</section>
</main></body></html>"""


def report_generator_agent(state: Any) -> dict:
    """Main function entry point for compiling the analytical HTML asset inside the working directory."""
    verdict_data = {}
    if isinstance(state, dict):
        verdict_data = state.get("verdict", {})
        
    if "config" not in state or not isinstance(state["config"], dict):
        state = dict(state)
        state["config"] = {
            "judge": {"truth_threshold": 40.0, "deviation_threshold": 65.0}
        }
    
    real_duration = _resolve_real_duration(state, verdict_data.get("by_segment", []))

    compiled_json_data = {
        "video": state.get("video", "videos/discurso.mp4"), 
        "speaker": state.get("speaker", "Desconocido"),
        "duration_seconds": real_duration,
        "transcription_source": state.get("transcription_source", "fallback"),
        "verdict": {k: v for k, v in verdict_data.items() if k != "by_segment"},
        "left_wing_evidence": state.get("left_wing_evidence", []),
        "right_wing_evidence": state.get("right_wing_evidence", []),
        "omitted_topics": state.get("omitted_topics", []),
        "segments": [{**s, **{"index": f.get("index", 0.0), "families": f.get("families", {}), "es_linea_base": f.get("es_linea_base", False)}}
                     for s, f in zip(state.get("segments", []), verdict_data.get("by_segment", []))],
        "logs": state.get("logs", []),
        "errors": state.get("errors", []),
        "architecture": (state.get("plan_metrics") or {}).get("architecture", "pipeline"),
        "plan_metrics": state.get("plan_metrics"),
        "plan": state.get("plan", []),
        "methodology_warning": METRIC_WARNING,
    }
    
    working_dir_base = state.get("working_dir", "results/output")
    working_dir_path = Path(working_dir_base)
    working_dir_path.mkdir(parents=True, exist_ok=True)
    
    report_json_destination = working_dir_path / "result.json"
    report_html_destination = working_dir_path / "report.html"
    
    report_json_destination.write_text(json.dumps(round_nested_data(compiled_json_data), ensure_ascii=False, indent=2), encoding="utf-8")
    report_html_destination.write_text(_compile_html_template(state, verdict_data), encoding="utf-8")
    
    print(f"  → [{AGENT_NAME} Agent] Reporte HTML generado con éxito en: {report_html_destination}")
    
    return {
        "report": {
            "json": str(report_json_destination),
            "html": str(report_html_destination)
        },
        "logs": [f"[{AGENT_NAME}] Archivos result.json y report.html guardados en el disco local."]
    }