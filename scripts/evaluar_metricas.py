"""Evaluación de las métricas v2 de la arquitectura actual (texto + RAG + juez).

Mide lo que el sistema puede demostrar hoy:
  - RTF (tiempo de análisis / duración del discurso)
  - memoria pico del proceso
  - reproducibilidad entre dos corridas idénticas
  - porcentaje de hallazgos críticos explicados
  - presencia de la advertencia de validez en el reporte
  - métricas del plan (cobertura y replanificación) y del supervisor
  - ejecuciones completas

Uso:
    python scripts/evaluar_metricas.py
    python scripts/evaluar_metricas.py --video "videos/Discurso Dina Boluarte 30 de enero de 2023.mp4"

Genera `resultados_metricas/evaluacion.json` y muestra una tabla Markdown.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import cargar_config
from src.orchestrator import SUPPORTED_ARCHITECTURES, analyze_video_pipeline

DEFAULT_VIDEO = "videos/Discurso Dina Boluarte 30 de enero de 2023.mp4"
OUTPUT_DIR = "resultados_metricas/evaluacion"


class MemoryMonitor:
    """Muestrea el RSS del proceso actual mientras corre el análisis."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_bytes = 0

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            except Exception:
                pass
            time.sleep(0.05)

    def __enter__(self) -> "MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        return round(self.peak_bytes / (1024 * 1024), 1)


def _without_logs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Copia comparable del resultado, sin bitácoras que incluyen tiempos."""
    clean = deepcopy(state)
    for key in ("logs", "errors", "report", "working_dir", "audio"):
        clean.pop(key, None)
    return clean


def run_once(video: str, architecture: str, run_name: str, speaker: str) -> Dict[str, Any]:
    """Ejecuta el pipeline una vez y devuelve estado, tiempo y memoria."""
    config = cargar_config(None)
    output_dir = str(Path(OUTPUT_DIR) / run_name)
    monitor = MemoryMonitor()
    started_at = time.perf_counter()
    with monitor:
        state = analyze_video_pipeline(
            video_path=video,
            config=config,
            output_dir=output_dir,
            speaker=speaker,
            verbose=False,
            architecture=architecture,
        )
    elapsed = time.perf_counter() - started_at
    duration = float(state.get("duration", 0.0) or 0.0)
    return {
        "architecture": architecture,
        "elapsed_s": round(elapsed, 2),
        "duration_s": round(duration, 2),
        "rtf": round(elapsed / duration, 4) if duration else None,
        "peak_mb": monitor.peak_mb,
        "state": state,
    }


def explained_ratio(verdict: Dict[str, Any]) -> tuple[int, int]:
    findings = verdict.get("critical_findings") or []
    explained = [f for f in findings if str(f.get("audit_note", "")).strip()]
    return len(explained), len(findings)


def warning_present(state: Dict[str, Any]) -> bool:
    html_path = (state.get("report") or {}).get("html")
    if not html_path:
        return False
    path = Path(html_path)
    if not path.exists():
        return False
    html = path.read_text(encoding="utf-8", errors="ignore")
    return "auditoría algorítmica de consistencia factual" in html


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa las métricas v2 del sistema")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--speaker", default="Dina Boluarte")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Any] = {}
    attempts = 0
    completed = 0

    for run_name, architecture in (
        ("pipeline_1", "pipeline"),
        ("pipeline_2", "pipeline"),
        ("plan", "plan-and-execute"),
        ("supervisor", "supervisor"),
    ):
        if architecture not in SUPPORTED_ARCHITECTURES:
            runs[run_name] = {"architecture": architecture, "skipped": "no soportada en esta rama"}
            print(f"[skip] {run_name}: arquitectura no soportada en esta rama")
            continue
        attempts += 1
        try:
            result = run_once(args.video, architecture, run_name, args.speaker)
            runs[run_name] = result
            completed += 1
            print(f"[ok] {run_name}: {result['elapsed_s']}s, RTF={result['rtf']}, RAM={result['peak_mb']} MB")
        except Exception as exc:  # noqa: BLE001
            runs[run_name] = {"architecture": architecture, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[error] {run_name}: {exc}")

    pipeline_runs = [runs.get("pipeline_1", {}), runs.get("pipeline_2", {})]
    reproducible = False
    if all("state" in run for run in pipeline_runs):
        reproducible = _without_logs(pipeline_runs[0]["state"]) == _without_logs(pipeline_runs[1]["state"])

    baseline = runs.get("pipeline_1", {})
    verdict = (baseline.get("state") or {}).get("verdict", {})
    explained, total = explained_ratio(verdict)

    plan_metrics = (runs.get("plan", {}).get("state") or {}).get("plan_metrics") or {}
    supervisor_metrics = (runs.get("supervisor", {}).get("state") or {}).get("supervisor_metrics") or {}

    summary = {
        "video": args.video,
        "duration_s": baseline.get("duration_s"),
        "runs": {
            name: {key: value for key, value in run.items() if key != "state"}
            for name, run in runs.items()
        },
        "checks": {
            "rtf_pipeline": baseline.get("rtf"),
            "rtf_plan": runs.get("plan", {}).get("rtf"),
            "rtf_supervisor": runs.get("supervisor", {}).get("rtf"),
            "peak_mb": max((run.get("peak_mb") or 0) for run in runs.values()),
            "reproducible": reproducible,
            "explained_findings": explained,
            "total_findings": total,
            "explained_ratio": round(explained / total, 3) if total else None,
            "warning_present": warning_present(baseline.get("state") or {}),
            "plan_coverage": plan_metrics.get("coverage"),
            "plan_steps": plan_metrics.get("planned_steps"),
            "plan_resolved": plan_metrics.get("resolved_steps"),
            "plan_replanned": plan_metrics.get("replanned_steps"),
            "supervisor_cycles": supervisor_metrics.get("total_cycles"),
            "supervisor_workers": len(supervisor_metrics.get("workers_called") or []),
            "supervisor_alerts": supervisor_metrics.get("critical_discrepancies_detected"),
            "completed_runs": completed,
            "attempted_runs": attempts,
            "completion_ratio": round(completed / attempts, 3),
        },
    }

    output_path = Path(OUTPUT_DIR) / "evaluacion.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Resumen de métricas v2 ===")
    print(json.dumps(summary["checks"], ensure_ascii=False, indent=2))
    print(f"\nGuardado en: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
