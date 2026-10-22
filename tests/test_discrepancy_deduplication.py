"""Pruebas de regresión del reporte de discrepancias.

Objetivo: una misma promesa repetida en varios segmentos debe generar UN solo
hallazgo crítico (en su primera mención) y no una lista con la misma nota
repetida nueve veces.

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

from src.agentes.factual_agent import factual_alignment_agent
from src.agentes.judge_agent import consolidation_judge_agent
from src.agentes.linguistic_agent import linguistic_agent
from src.agentes.report_agent import _compile_html_template
from src.config import cargar_config

SPEECH_TEXTS = [
    "Anuncio que vamos a aprobar el adelanto de elecciones para este año 2023.",
    "Insisto en que el adelanto de elecciones se dará este año, cueste lo que cueste.",
    "Por eso proponemos el adelanto de elecciones generales sin más demora.",
    "Saludo cordialmente a las autoridades y al público presente esta mañana.",
]


def _build_state() -> dict:
    segments = [
        {"id": index, "start": index * 5.0, "end": (index + 1) * 5.0, "text": text}
        for index, text in enumerate(SPEECH_TEXTS)
    ]
    state = {
        "video": "videos/prueba.mp4",
        "speaker": "Prueba",
        "working_dir": "results/_prueba_unittest",
        "duration": 20.0,
        "config": cargar_config(None),
        "segments": segments,
        "logs": [],
        "errors": [],
    }
    state.update(linguistic_agent(state))
    state.update(factual_alignment_agent(state))
    return state


class DiscrepancyDeduplicationTest(unittest.TestCase):
    def test_factual_reports_a_single_primary_finding_per_topic(self):
        state = _build_state()
        rows = state["factual_rag_metrics"]["by_segment"]

        primary_critical = [row for row in rows if row.get("is_primary") and row["index"] >= 65.0]

        self.assertEqual(len(primary_critical), 1, "solo la primera mención debe ser el hallazgo crítico")
        self.assertEqual(primary_critical[0]["mention_count"], 3)
        self.assertEqual(sum(1 for row in rows if row.get("reiteration_of") is not None), 2)

    def test_reiterations_do_not_reach_the_critical_threshold(self):
        state = _build_state()
        repeats = [row for row in state["factual_rag_metrics"]["by_segment"] if row.get("reiteration_of") is not None]

        self.assertTrue(repeats)
        self.assertTrue(all(row["index"] < 65.0 for row in repeats))

    def test_judge_emits_one_finding_and_report_shows_one_block(self):
        state = _build_state()
        verdict = consolidation_judge_agent(state)["verdict"]
        state["verdict"] = verdict

        self.assertEqual(len(verdict["critical_findings"]), 1)
        self.assertEqual(verdict["critical_segments"], [0])

        rendered = _compile_html_template(state, verdict)
        critical_section = rendered.split("Segmentos con Discrepancias Críticas")[1].split("Transcripción")[0]

        self.assertEqual(critical_section.count("Hallazgo RAG"), 1)
        self.assertIn("Mismo tema en 3 segmentos", critical_section)


if __name__ == "__main__":
    unittest.main()
