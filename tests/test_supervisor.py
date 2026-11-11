"""Pruebas unitarias y de integración de la Arquitectura Multi-Agente con Supervisor (Clean Architecture).

Ejecutar con:
    python -m unittest discover -s tests -v
    python -m unittest tests/test_supervisor.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.application.supervisor_policy import StandardSupervisorPolicy
from src.arquitecturas.supervisor import analyze_with_supervisor, build_supervisor_graph
from src.config import cargar_config
from src.domain.entities import (
    SupervisorDirective,
    SupervisorMetrics,
    SupervisorPhase,
    WorkerType,
)

SPEECH_SEGMENTS = [
    {
        "id": 0,
        "start": 0.0,
        "end": 8.0,
        "text": "Anuncio que vamos a aprobar el adelanto de elecciones generales para este año 2023.",
    },
    {
        "id": 1,
        "start": 8.0,
        "end": 15.0,
        "text": "Las elecciones generales serán indefectiblemente este año, primera vuelta en octubre.",
    },
    {
        "id": 2,
        "start": 15.0,
        "end": 22.0,
        "text": "Presentamos esta iniciativa de reforma constitucional para recortar el periodo presidencial.",
    },
    {
        "id": 3,
        "start": 22.0,
        "end": 28.0,
        "text": "Agradezco a todos los compatriotas y a las autoridades por su presencia esta noche.",
    },
]


def _build_base_state() -> dict:
    return {
        "video": "videos/discurso_prueba.mp4",
        "speaker": "Dina Boluarte",
        "working_dir": "results/_prueba_supervisor",
        "duration": 28.0,
        "config": cargar_config(None),
        "segments": list(SPEECH_SEGMENTS),
        "audio": "dummy_audio.wav",
        "logs": [],
        "errors": [],
        "supervisor_trace": [],
    }


class SupervisorArchitectureTest(unittest.TestCase):
    """Batería de pruebas para la Arquitectura de Agente IA con Supervisor."""

    def test_domain_entities_integrity(self):
        """Verifica que las entidades de dominio puro mantengan inmutabilidad y tipado estricto."""
        directive = SupervisorDirective(
            cycle=1,
            phase=SupervisorPhase.INITIALIZING,
            next_worker=WorkerType.INGESTION_TRANSCRIPTION,
            rationale="Auditoría inicial requerida.",
            critical_alert=False,
        )
        self.assertEqual(directive.cycle, 1)
        self.assertEqual(directive.phase, SupervisorPhase.INITIALIZING)
        self.assertEqual(directive.next_worker, WorkerType.INGESTION_TRANSCRIPTION)

        metrics = SupervisorMetrics(
            architecture="supervisor",
            total_cycles=7,
            workers_called=["promise_extraction", "factual_audit"],
            critical_discrepancies_detected=2,
            final_verdict="neutral",
            confidence_score=0.85,
        )
        m_dict = metrics.to_dict()
        self.assertEqual(m_dict["architecture"], "supervisor")
        self.assertEqual(m_dict["total_cycles"], 7)
        self.assertEqual(len(m_dict["workers_called"]), 2)

    def test_supervisor_policy_progression(self):
        """Valida que la política del supervisor direccione ordenadamente a cada especialista según la falta de datos."""
        policy = StandardSupervisorPolicy(max_cycles=15)

        # 1. Sin segmentos -> Transcripción
        s0 = {"logs": []}
        d0 = policy.evaluate(s0, cycle=1)
        self.assertEqual(d0.next_worker, WorkerType.INGESTION_TRANSCRIPTION)

        # 2. Con segmentos pero sin promesas -> Extracción de promesas
        s1 = {"segments": list(SPEECH_SEGMENTS)}
        d1 = policy.evaluate(s1, cycle=2)
        self.assertEqual(d1.next_worker, WorkerType.PROMISE_EXTRACTION)

        # 3. Con promesas pero sin análisis estilométrico -> Lingüístico y luego Sintáctico
        s2 = {**s1, "extracted_promises": [{"quote": "elecciones"}]}
        d2 = policy.evaluate(s2, cycle=3)
        self.assertEqual(d2.next_worker, WorkerType.LINGUISTIC_ANALYSIS)

        s3 = {**s2, "linguistic_metrics": {"by_segment": []}}
        d3 = policy.evaluate(s3, cycle=4)
        self.assertEqual(d3.next_worker, WorkerType.SYNTACTIC_ANALYSIS)

        # 4. Estilo completo -> Auditoría Factual RAG
        s4 = {**s3, "syntactic_metrics": {"by_segment": []}}
        d4 = policy.evaluate(s4, cycle=5)
        self.assertEqual(d4.next_worker, WorkerType.FACTUAL_AUDIT)

        # 5. Factual completo -> Juez
        s5 = {**s4, "factual_rag_metrics": {"summary": {"critical_segments_count": 3}}}
        d5 = policy.evaluate(s5, cycle=6)
        self.assertEqual(d5.next_worker, WorkerType.JUDGE_CONSOLIDATION)
        self.assertTrue(d5.critical_alert)  # Debe levantar alerta por criticidad

        # 6. Veredicto emitido -> Reportero
        s6 = {**s5, "verdict": {"label": "deviation", "confidence": 0.90}}
        d6 = policy.evaluate(s6, cycle=7)
        self.assertEqual(d6.next_worker, WorkerType.REPORT_GENERATION)

        # 7. Reporte generado -> FINISH
        s7 = {**s6, "report": {"html": "results/report.html"}}
        d7 = policy.evaluate(s7, cycle=8)
        self.assertEqual(d7.next_worker, WorkerType.FINISH)
        self.assertEqual(d7.phase, SupervisorPhase.COMPLETED)

    def test_supervisor_critical_discrepancy_alert(self):
        """Verifica que el supervisor detecte discrepancias críticas (e.g., comicios 2023) y levante una alerta auditable."""
        policy = StandardSupervisorPolicy()
        state = {
            "segments": list(SPEECH_SEGMENTS),
            "extracted_promises": [{"promise": "adelanto 2023"}],
            "linguistic_metrics": {"by_segment": []},
            "syntactic_metrics": {"by_segment": []},
            "factual_rag_metrics": {
                "summary": {"critical_segments_count": 2, "mean_factual_discrepancy": 0.85}
            },
        }
        directive = policy.evaluate(state, cycle=1)
        self.assertEqual(directive.next_worker, WorkerType.JUDGE_CONSOLIDATION)
        self.assertTrue(directive.critical_alert)
        self.assertIn("ALERTA CRÍTICA", directive.rationale)

    def test_supervisor_full_graph_execution(self):
        """Ejecuta el grafo compilado de LangGraph completo con el Supervisor como hub central."""
        base_state = _build_base_state()

        final_state = analyze_with_supervisor(base_state, verbose=False)

        # Verificaciones del flujo supervisado
        self.assertIn("supervisor_trace", final_state)
        trace = final_state["supervisor_trace"]
        self.assertGreater(len(trace), 3)

        # Validar que los workers clave fueron delegados por el supervisor
        workers_delegated = [entry["assigned_worker"] for entry in trace]
        self.assertIn(WorkerType.PROMISE_EXTRACTION.value, workers_delegated)
        self.assertIn(WorkerType.FACTUAL_AUDIT.value, workers_delegated)
        self.assertIn(WorkerType.JUDGE_CONSOLIDATION.value, workers_delegated)
        self.assertIn(WorkerType.REPORT_GENERATION.value, workers_delegated)
        self.assertIn(WorkerType.FINISH.value, workers_delegated)

        # Validar métricas consolidadas del supervisor
        self.assertIn("supervisor_metrics", final_state)
        metrics = final_state["supervisor_metrics"]
        self.assertEqual(metrics["architecture"], "supervisor")
        self.assertGreater(metrics["total_cycles"], 0)

        # Validar veredicto emitido por el Juez tras la supervisión
        self.assertIn("verdict", final_state)
        self.assertIn(final_state["verdict"]["label"], ("truth", "deviation", "neutral"))

        # Validar generación del reporte
        self.assertIn("report", final_state)
        self.assertIn("html", final_state["report"])


if __name__ == "__main__":
    unittest.main()
