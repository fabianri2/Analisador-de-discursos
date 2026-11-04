"""Capa de Aplicación - Política de Decisión del Agente Supervisor.

Implementa ISupervisorPolicy con la lógica de negocio para dirigir la auditoría
forense del discurso de forma dinámica, auditable y robusta.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..domain.entities import SupervisorDirective, SupervisorPhase, WorkerType
from ..domain.interfaces import ISupervisorPolicy


class StandardSupervisorPolicy(ISupervisorPolicy):
    """Política estándar de orquestación supervisada basada en completitud y criticidad factual."""

    def __init__(self, max_cycles: int = 15):
        self.max_cycles = max_cycles

    def evaluate(self, state: Dict[str, Any], cycle: int) -> SupervisorDirective:
        """Evalúa el estado acumulado y retorna la directiva de trabajo inmediata."""
        # 1. Salvaguarda de límite de ciclos para prevenir bucles infinitos
        if cycle >= self.max_cycles:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.COMPLETED,
                next_worker=WorkerType.FINISH,
                rationale=f"Límite de seguridad alcanzado ({self.max_cycles} ciclos). Supervisor fuerza cierre de grafo.",
            )

        # 2. Paso 1: Transcripción y segmentación de audio
        segments = state.get("segments") or []
        if not segments:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.INITIALIZING,
                next_worker=WorkerType.INGESTION_TRANSCRIPTION,
                rationale="No se cuenta con transcripción ni segmentos. Delegar en especialista de Ingesta y Transcripción.",
            )

        # 3. Paso 2: Aislamiento de compromisos y promesas
        extracted_promises = state.get("extracted_promises")
        if extracted_promises is None:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.EXTRACTING_PROMISES,
                next_worker=WorkerType.PROMISE_EXTRACTION,
                rationale="Transcripción disponible. Delegar en especialista de Promesas para aislar compromisos verificables.",
            )

        # 4. Paso 3: Análisis estilométrico y opacidad sintáctica
        linguistic = state.get("linguistic_metrics")
        syntactic = state.get("syntactic_metrics")
        if linguistic is None:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.ANALYZING_STYLE,
                next_worker=WorkerType.LINGUISTIC_ANALYSIS,
                rationale="Faltan marcadores léxicos LIWC. Delegar en especialista Lingüístico.",
            )
        if syntactic is None:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.ANALYZING_STYLE,
                next_worker=WorkerType.SYNTACTIC_ANALYSIS,
                rationale="Falta cálculo de opacidad sintáctica (Índice de Niebla). Delegar en especialista Sintáctico.",
            )

        # 5. Paso 4: Contrastación Factual RAG (MEF / BCRP / JNE / Defensoría)
        factual_metrics = state.get("factual_rag_metrics")
        if factual_metrics is None:
            total_promises = len(extracted_promises)
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.AUDITING_FACTS,
                next_worker=WorkerType.FACTUAL_AUDIT,
                rationale=(
                    f"Métricas estilísticas completas y {total_promises} promesas aisladas. "
                    "Delegar en especialista Factual RAG para contrastación regulatoria contra MEF/BCRP/JNE."
                ),
            )

        # 6. Paso 5: Evaluación de Criticidad y Alertas Forenses
        critical_count = factual_metrics.get("summary", {}).get("critical_segments_count", 0)
        has_critical_alert = critical_count > 0

        # 7. Paso 6: Arbitraje y Veredicto Consolidado por el Juez
        verdict = state.get("verdict")
        if verdict is None:
            crit_note = (
                f" ¡ALERTA CRÍTICA! Se detectaron {critical_count} segmentos con discrepancia severa."
                if has_critical_alert else " Verificación preliminar sin desvíos anómalos."
            )
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.CONSOLIDATING_VERDICT,
                next_worker=WorkerType.JUDGE_CONSOLIDATION,
                rationale=f"Evidencia factual completa.{crit_note} Delegar en Agente Juez para arbitraje multi-modal.",
                critical_alert=has_critical_alert,
            )

        # 8. Paso 7: Publicación y Generación de Reportes
        report = state.get("report")
        if report is None:
            return SupervisorDirective(
                cycle=cycle,
                phase=SupervisorPhase.GENERATING_REPORT,
                next_worker=WorkerType.REPORT_GENERATION,
                rationale=f"Veredicto emitido ({verdict.get('label')}). Delegar en Agente Reportero para generar HTML y JSON.",
            )

        # 9. Paso 8: Misión Cumplida
        return SupervisorDirective(
            cycle=cycle,
            phase=SupervisorPhase.COMPLETED,
            next_worker=WorkerType.FINISH,
            rationale="Todos los artefactos forenses (transcripción, análisis, RAG, veredicto y reporte) han sido generados exitosamente.",
        )
