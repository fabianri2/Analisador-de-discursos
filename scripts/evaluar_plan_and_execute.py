"""Evaluación de la arquitectura Plan-and-Execute.

Uso local (sin LangSmith):
    python scripts/evaluar_plan_and_execute.py

Con LangSmith (opcional):
    pip install -r requirements-eval.txt
    $env:LANGSMITH_API_KEY = "lsv2_..."
    python scripts/evaluar_plan_and_execute.py --langsmith

El script construye el estado desde los subtítulos de ``videos/``, ejecuta el
planner + executor + replanner y reporta métricas de plan. Si se pide LangSmith,
crea (o reutiliza) un dataset y registra un experimento con los evaluadores de
cobertura y de temas críticos detectados.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agentes.linguistic_agent import linguistic_agent
from src.agentes.promise_agent import promise_extractor_agent
from src.agentes.transcription_agent import _read_local_subtitles
from src.arquitecturas.plan_and_execute import plan_and_execute_agent
from src.config import cargar_config

CASES = [
    {
        "name": "demo",
        "srt": "videos/discurso.srt",
        "expected_topics": ["Cronograma Electoral y Mandato Presidencial"],
    },
    {
        "name": "boluarte_2023",
        "srt": "videos/Discurso Dina Boluarte 30 de enero de 2023.srt",
        "expected_topics": [
            "Cronograma Electoral y Mandato Presidencial",
            "Reforma Constitucional",
            "Seguridad y Paz Social",
        ],
    },
]


def run_case(case: dict) -> dict:
    """Ejecuta la arquitectura sobre un subtítulo y devuelve las métricas del plan."""
    subtitle_path = REPOSITORY_ROOT / case["srt"]
    segments = _read_local_subtitles(subtitle_path)
    state = {
        "video": case["srt"],
        "speaker": case["name"],
        "working_dir": f"results/_eval/{case['name']}",
        "duration": segments[-1]["end"] if segments else 0.0,
        "config": cargar_config(None),
        "segments": segments,
        "logs": [],
        "errors": [],
    }
    state.update(linguistic_agent(state))
    state.update(promise_extractor_agent(state))
    output = plan_and_execute_agent(state)

    metrics = output["plan_metrics"]
    findings = [
        {
            "topic": row["topic"],
            "factual_score": round(row["index"], 1),
            "mentions": row.get("mention_count", 1),
        }
        for row in output["factual_rag_metrics"]["by_segment"]
        if row.get("is_primary") and row["index"] >= 65.0
    ]
    return {
        "case": case["name"],
        "segments": len(segments),
        "plan_metrics": metrics,
        "findings": findings,
        "critical_topics": [finding["topic"] for finding in findings],
        "plan": output["plan"],
    }


def _print_summary(results: list[dict]) -> None:
    print("\n=== Resumen Plan-and-Execute ===")
    for result in results:
        metrics = result["plan_metrics"]
        print(
            f"- {result['case']}: {metrics['planned_steps']} pasos, "
            f"{metrics['resolved_steps']} resueltos, {metrics['replanned_steps']} replanificados, "
            f"cobertura {metrics['coverage'] * 100:.0f}%, "
            f"{len(result['findings'])} hallazgos, {metrics['duration_s']:.2f}s"
        )


def _publish_to_langsmith(results: list[dict], dataset_name: str = "plan-and-execute-discursos") -> None:
    """Registra dataset + experimento en LangSmith (requiere LANGSMITH_API_KEY)."""
    try:
        from langsmith import Client, evaluate  # type: ignore
    except ImportError:
        print("LangSmith no está instalado. Ejecuta: pip install -r requirements-eval.txt")
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("Define LANGSMITH_API_KEY para publicar el experimento.")
        return

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Discursos peruanos para evaluar la arquitectura Plan-and-Execute.",
        )
        for case in CASES:
            client.create_example(
                inputs={"srt": case["srt"], "name": case["name"]},
                outputs={"expected_topics": case.get("expected_topics", [])},
                dataset_id=dataset.id,
            )

    def target(inputs: dict) -> dict:
        case = next(item for item in CASES if item["srt"] == inputs["srt"])
        return run_case(case)

    def coverage_evaluator(run, example) -> dict:
        return {"key": "coverage", "score": float(run.outputs["plan_metrics"]["coverage"])}

    def topics_evaluator(run, example) -> dict:
        expected = set((example.outputs or {}).get("expected_topics", []))
        found = set(run.outputs.get("critical_topics", []))
        score = len(expected & found) / len(expected) if expected else 1.0
        return {"key": "topic_recall", "score": score}

    evaluate(
        target,
        data=dataset_name,
        evaluators=[coverage_evaluator, topics_evaluator],
        experiment_prefix="plan-and-execute",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa la arquitectura Plan-and-Execute")
    parser.add_argument("--langsmith", action="store_true", help="Publica el experimento en LangSmith")
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON")
    args = parser.parse_args()

    results = [run_case(case) for case in CASES]
    _print_summary(results)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.langsmith:
        _publish_to_langsmith(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
