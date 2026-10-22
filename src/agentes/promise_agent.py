"""Agente Extractor de Promesas: aisla compromisos cuantitativos, electorales y normativos."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Any
import numpy as np

AGENT_NAME = "PromiseExtractor"

PROMISE_TRIGGERS = [
    (r"\b(adelanto de elecciones|elecciones generales|convoque a elecciones|comicios)\b", "Electoral / Institucional", "Convocatoria o cronograma de comicios generales anticipados."),
    (r"\b(generales sean indefectiblemente|indefectiblemente|este a[ñn]o 2023|octubre de este a[ñn]o|primera vuelta|segunda vuelta)\b", "Electoral / Institucional", "Meta categórica de fecha para elecciones generales."),
    (r"\b(reforma constitucional|reforma total de la constituci[oó]n|asamblea constituyente|ley 27\.?600|refer[eé]ndum)\b", "Reforma Constitucional", "Modificación o proyecto de ley para alterar la carta magna."),
    (r"\b(recort[oe] del periodo|nos vamos todos)\b", "Electoral / Institucional", "Reducción formal del mandato presidencial y congresal."),
    (r"\b(iniciativas? legislativas?|proyecto de ley|presentando inmediatamente|car[aá]cter urgencia)\b", "Legislativo", "Presentación inmediata de proyectos de ley prioritarios."),
    (r"\b(vamos a|anuncio que|propongo que|proponemos que|disponemos|se dispone|me refiero de proponer)\b", "Compromiso de Gestión", "Compromiso explícito de acción ejecutiva."),
    (r"\b(pbi|crecimiento|inversi[oó]n|presupuesto|mef|millones|obras? p[uú]blicas?|d[eé]ficit)\b", "Económico / Presupuestal", "Asignación de partidas de gasto o metas macroeconómicas."),
    (r"\b(garantizar|salvaguardar|pacificar|di[aá]logo|paz social|tranquilidad)\b", "Paz Social y Seguridad", "Promesa de estabilización ciudadana o canales de diálogo.")
]


def promise_extractor_agent(state: dict) -> dict:
    """
    Extracts verifiable promises, policy commitments, and political announcements from speech segments.
    """
    segments = state.get("segments", [])
    extracted_promises: List[Dict[str, Any]] = []
    segments_output = []
    
    promise_counter = 1
    for segment in segments:
        text = segment.get("text", "")
        normalized_text = unicodedata.normalize("NFC", text.lower())
        
        detected_categories = []
        matched_triggers = 0
        
        for pattern, cat, desc in PROMISE_TRIGGERS:
            if re.search(pattern, normalized_text):
                matched_triggers += 1
                detected_categories.append(cat)
                extracted_promises.append({
                    "promise_id": promise_counter,
                    "segment_id": segment["id"],
                    "start": segment.get("start", 0.0),
                    "category": cat,
                    "description": desc,
                    "quote": text.strip()
                })
                promise_counter += 1
                
        # Densidad de promesas por segmento normalizada de 0 a 100
        promise_score = float(np.clip(matched_triggers * 35.0, 0.0, 100.0))
        
        segments_output.append({
            "id": segment["id"],
            "promise_score": promise_score,
            "promise_count": matched_triggers,
            "categories": detected_categories
        })
        
    print(f"  → [{AGENT_NAME}] Aisladas {len(extracted_promises)} promesas y compromisos de gestión en el discurso.")
    
    return {
        "promise_extraction_metrics": {
            "by_segment": segments_output,
            "summary": {
                "total_promises": len(extracted_promises),
                "high_commitment_segments": sum(1 for s in segments_output if s["promise_score"] >= 50.0)
            },
            "markers_direction": {"promise_score": +1}
        },
        "extracted_promises": extracted_promises,
        "logs": [f"[{AGENT_NAME}] {len(extracted_promises)} promesas y compromisos de Estado detectados."]
    }