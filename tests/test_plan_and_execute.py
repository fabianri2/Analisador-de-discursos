"""Pruebas de la arquitectura Plan-and-Execute.

Ejecutar con:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agentes.judge_agent import consolidation_judge_agent
from src.agentes.linguistic_agent import linguistic_agent
from src.agentes.promise_agent import promise_extractor_agent
from src.agentes.report_agent import _compile_html_template
from src.arquitecturas.plan_and_execute import (
    build_plan,
    execute_plan,
    plan_and_execute_agent,
)
from src.config import cargar_config

SPEECH_TEXTS = [
    "Anuncio que vamos a aprobar el adelanto de elecciones para este año 2023.",
    "Insisto en que el adelanto de elecciones se dará este año, cueste lo que cueste.",
    "Por eso proponemos el adelanto de elecciones generales sin más demora.",
    "Saludo cordialmente a las autoridades y al público presente esta mañana.",
]

REPLAN_TEXT = "Anuncio que el sufragio será este año, pese a todo."


def _build_state(texts: list[str]) -> dict:
    segments = [
        {"id": index, "start": index * 5.0, "end": (index + 1) * 5.0, "text": text}
        for index, text in enumerate(texts)
    ]
    state = {
        "video": "videos/prueba.mp4",
        "speaker": "Prueba",
        "working_dir": "results/_prueba_plan",
        "duration": float(len(texts) * 5),
        "config": cargar_config(None),
        "segments": segments,
        "logs": [],
        "errors": [],
    }
    state.update(linguistic_agent(state))
    state.update(promise_extractor_agent(state))
    return state


class PlanAndExecuteTest(unittest.TestCase):
    def test_plan_only_includes_claims(self):
        state = _build_state(SPEECH_TEXTS)
        plan = build_plan(state)

        self.assertEqual(len(plan), 3)
        self.assertTrue(all(step["status"] == "pendiente" for step in plan))
        self.assertEqual([step["segment_id"] for step in plan], [0, 1, 2])

    def test_executor_resolves_every_planned_step(self):
        state = _build_state(SPEECH_TEXTS)
        plan = execute_plan(build_plan(state), state)

        self.assertTrue(all(step["status"] == "resuelto" for step in plan))
        self.assertTrue(all(step["resolution"] == "keyword_estricto" for step in plan))

    def test_replanner_recovers_claim_without_strict_keyword(self):
        state = _build_state([REPLAN_TEXT])
        plan = build_plan(state)
        self.assertEqual(len(plan), 1)
        executed = execute_plan(plan, state)

        self.assertEqual(executed[0]["status"], "resuelto")
        self.assertTrue(executed[0]["replanned"])
        self.assertEqual(executed[0]["resolution"], "replanificacion_tematica")
        self.assertEqual(executed[0]["topic"], "Cronograma Electoral y Mandato Presidencial")

    def test_output_matches_judge_schema_and_deduplicates_topics(self):
        state = _build_state(SPEECH_TEXTS)
        output = plan_and_execute_agent(state)
        state.update(output)

        metrics = output["plan_metrics"]
        self.assertEqual(metrics["planned_steps"], 3)
        self.assertEqual(metrics["resolved_steps"], 3)
        self.assertEqual(metrics["replanned_steps"], 0)
        self.assertEqual(metrics["coverage"], 1.0)

        rows = state["factual_rag_metrics"]["by_segment"]
        primaries = [row for row in rows if row.get("is_primary") and row["index"] >= 65.0]
        self.assertEqual(len(primaries), 1)
        self.assertEqual(primaries[0]["mention_count"], 3)
        self.assertEqual(sum(1 for row in rows if row.get("reiteration_of") is not None), 2)

        verdict = consolidation_judge_agent(state)["verdict"]
        state["verdict"] = verdict
        self.assertEqual(len(verdict["critical_findings"]), 1)
        self.assertIn("Plan de verificación", _compile_html_template(state, verdict))


if __name__ == "__main__":
    unittest.main()
