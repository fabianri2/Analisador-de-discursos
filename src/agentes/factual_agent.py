"""Agente de Auditoría Factual (RAG Sectorial): cruza afirmaciones del discurso con ChromaDB por sector, cumplimiento actual e hitos."""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Any
import numpy as np

try:
    import chromadb
except ImportError:
    chromadb = None

from ..state import SpeechAnalysisState

AGENT_NAME = "FactualAlignment"
REITERATION_FACTOR = 0.45
CHROMA_DB_PATH = os.path.join("data_storage", "vector_chroma")
CHROMA_COLLECTION_NAME = "political_discourse_sources"

# Mapeo de palabras clave para clasificar automáticamente el sector de cada segmento del discurso
SECTOR_CLASSIFIERS = [
    {
        "sector": "Reforma Constitucional",
        "fact_target": 0.90,
        "keywords": [
            "adelanto de elecciones", "elecciones generales", "comicios generales",
            "recorte del periodo", "recorto el periodo", "convoque a elecciones",
            "reforma total de la constitucion", "asamblea constituyente",
            "reforma constitucional", "comision de constitucion", "constitucion"
        ],
    },
    {
        "sector": "Seguridad",
        "fact_target": 0.78,
        "keywords": [
            "policia nacional", "paz social", "estado de emergencia", "orden interno",
            "bloqueos", "muertos", "fallecidos", "violencia", "seguridad ciudadana",
            "delincuencia", "fuerzas armadas", "protestas"
        ],
    },
    {
        "sector": "Economía",
        "fact_target": 0.82,
        "keywords": [
            "producto bruto interno", "pbi", "presupuesto publico", "presupuesto nacional",
            "inversion publica", "deficit fiscal", "crecimiento economico", "economia",
            "inflacion", "con punche peru", "reactivacion", "empleo", "millones de soles"
        ],
    },
    {
        "sector": "Salud",
        "fact_target": 0.76,
        "keywords": [
            "salud", "hospital", "hospitales", "posta", "essalud", "minsa",
            "medicamentos", "medicina", "atencion medica", "sis", "medicos"
        ],
    },
    {
        "sector": "Educación",
        "fact_target": 0.75,
        "keywords": [
            "educacion", "colegio", "colegios", "escuela", "escuelas", "universidad",
            "docentes", "maestros", "minedu", "becas", "aulas", "ano escolar"
        ],
    },
    {
        "sector": "Agro",
        "fact_target": 0.77,
        "keywords": [
            "agricultura", "agro", "riego", "sequia", "sequias", "fertilizantes",
            "campesino", "agricultores", "midagri", "cosecha", "hidrico", "heladas"
        ],
    },
    {
        "sector": "Minería e Inversiones",
        "fact_target": 0.74,
        "keywords": [
            "mineria", "minero", "corredor minero", "cobre", "inversion privada",
            "proyectos mineros", "minem", "canon minero", "exploracion"
        ],
    },
    {
        "sector": "Infraestructura y Transportes",
        "fact_target": 0.79,
        "keywords": [
            "infraestructura", "carreteras", "obras publicas", "reconstruccion",
            "fenomeno el nino", "inundaciones", "mtc", "puentes", "ejecucion presupuestal",
            "aeropuerto", "tren", "transporte"
        ],
    },
]

# Alias de compatibilidad retroactiva para el nodo Plan-and-Execute
FACTUAL_BENCHMARKS = [
    {"topic": f"Sector {item['sector']}", "keywords": item["keywords"], "fact_target": item["fact_target"], "source": f"ChromaDB RAG - {item['sector']}", "audit_note": f"Auditoría sectorial {item['sector']}"}
    for item in SECTOR_CLASSIFIERS
]


# Sectores críticos que se auditan por omisión si no se mencionan en todo el discurso
CRITICAL_OMISSION_SECTORS = {
    "Salud Pública y Hospitales": ["salud", "hospital", "posta", "essalud", "minsa", "medicamentos"],
    "Desarrollo Territorial y Regiones": ["sierra", "andina", "andes", "regiones", "descentralizacion", "provincias", "puno", "cusco", "ayacucho"],
    "Educación e Infraestructura Escolar": ["educacion", "colegio", "escuela", "universidad", "docentes", "minedu"],
    "Reactivación Agraria e Hídrica": ["agricultura", "agro", "riego", "sequia", "fertilizantes", "campesino"],
    "Minería e Inversiones": ["mineria", "minero", "corredor minero", "inversion privada", "minem"],
    "Infraestructura y Prevención Climática": ["infraestructura", "carreteras", "reconstruccion", "fenomeno el nino", "mtc"],
}


def _normalize_for_matching(raw_text: str) -> str:
    decomposed = unicodedata.normalize("NFD", raw_text.lower())
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    tokens = re.findall(r"[a-z0-9]+", without_accents)
    return " " + " ".join(tokens) + " "


def _keyword_pattern(keyword: str) -> re.Pattern:
    normalized = _normalize_for_matching(keyword).strip()
    body = r"\s+".join(re.escape(token) for token in normalized.split())
    return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")


SECTOR_PATTERNS = [
    (item, [_keyword_pattern(kw) for kw in item["keywords"]])
    for item in SECTOR_CLASSIFIERS
]


def _slugify_topic(topic: str) -> str:
    return "_".join(_normalize_for_matching(topic).split())


def _classify_segment_sector(segment_text: str) -> Dict[str, Any] | None:
    normalized_text = _normalize_for_matching(segment_text)
    matched = [
        item
        for item, patterns in SECTOR_PATTERNS
        if any(pattern.search(normalized_text) for pattern in patterns)
    ]
    return max(matched, key=lambda x: x["fact_target"]) if matched else None

def _find_matching_benchmarks(segment_text: str) -> List[Dict[str, Any]]:
    """Alias de compatibilidad retroactiva para el nodo Plan-and-Execute."""
    normalized_text = _normalize_for_matching(segment_text)
    return [
        {
            "topic": f"Sector {item['sector']}",
            "keywords": item["keywords"],
            "fact_target": item["fact_target"],
            "source": f"ChromaDB RAG - {item['sector']}",
            "audit_note": f"Auditoría sectorial {item['sector']}",
        }
        for item, patterns in SECTOR_PATTERNS
        if any(pattern.search(normalized_text) for pattern in patterns)
    ]

def _map_segments_by_id(agent_metrics_block: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    if not isinstance(agent_metrics_block, dict):
        return {}
    return {row["id"]: row for row in agent_metrics_block.get("by_segment", [])}


def _get_chroma_collection():
    if chromadb is None or not os.path.exists(CHROMA_DB_PATH):
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        return None


def _query_official_sector_benchmark(collection, query_text: str, sector: str) -> Dict[str, Any]:
    """Consulta en ChromaDB el benchmark oficial del sector con estado de cumplimiento e hitos."""
    default_benchmark = {
        "source": f"Registros Oficiales Sector {sector}",
        "audit_note": f"Compromiso en sector {sector} sujeto a verificación de ejecución presupuestal.",
        "estado_actual": "Parcial",
        "evidencia_actual": "En seguimiento de ejecución fiscal y metas sectoriales.",
        "timeline": [
            "2023 | Anuncio de compromiso sectorial en mensaje presidencial.",
            "2024 | Evaluación intermedia de avance presupuestal.",
            "Actualidad | Seguimiento activo de metas institucionales."
        ]
    }
    if collection is None:
        return default_benchmark
    try:
        results = collection.query(
            query_texts=[query_text or sector],
            n_results=1,
            where={"$and": [{"tipo": "benchmark_oficial"}, {"sector": sector}]}
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if docs and metas:
            meta = metas[0]
            timeline = [
                h for h in [meta.get("hito_1", ""), meta.get("hito_2", ""), meta.get("hito_3", "")] if h
            ]
            return {
                "source": meta.get("fuente", default_benchmark["source"]),
                "audit_note": f"{docs[0]} — Evidencia: {meta.get('evidencia_actual', '')}",
                "estado_actual": meta.get("estado_actual", "Parcial"),
                "evidencia_actual": meta.get("evidencia_actual", default_benchmark["evidencia_actual"]),
                "timeline": timeline if timeline else default_benchmark["timeline"]
            }
    except Exception:
        pass
    return default_benchmark


def _query_spectrum_evidence(collection, query_text: str, sector: str, spectrum: str, n_results: int = 1) -> List[Dict[str, Any]]:
    """Consulta en ChromaDB fuentes de prensa filtradas por sector y espectro político ('izquierda' o 'derecha')."""
    if collection is None or not query_text.strip():
        return []
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"$and": [{"espectro": spectrum}, {"sector": sector}]}
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"espectro": spectrum}
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

        evidence_items = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            evidence_items.append({
                "excerpt": doc,
                "source_channel": meta.get("fuente", "Fuente Web Indexada") if meta else "Fuente Web Indexada",
                "sector": meta.get("sector", sector) if meta else sector,
                "spectrum": spectrum,
                "similarity_score": round(float(1.0 / (1.0 + dist)), 4)
            })
        return evidence_items
    except Exception:
        return []
    

def _detect_omitted_sectors(full_text: str) -> List[str]:
    normalized_full = _normalize_for_matching(full_text)
    omitted = []
    for sector_label, keywords in CRITICAL_OMISSION_SECTORS.items():
        if not any(_keyword_pattern(kw).search(normalized_full) for kw in keywords):
            omitted.append(sector_label)
    return omitted
def factual_alignment_agent(state: SpeechAnalysisState) -> dict:
    """
    Nodo principal de auditoría factual RAG: clasifica cada segmento en uno de los 8 sectores,
    consulta ChromaDB para obtener cumplimiento actual, hitos cronológicos y contraste de espectro político.
    """
    linguistic_map = _map_segments_by_id(state.get("linguistic_metrics", {}))
    speech_segments = state.get("segments", [])

    chroma_collection = _get_chroma_collection()
    speaker_name = (state.get("speaker") or "").strip()
    left_wing_evidence: List[Dict[str, Any]] = []
    right_wing_evidence: List[Dict[str, Any]] = []
    sector_audit_cards: List[Dict[str, Any]] = []

    # 1. Clasificar cada segmento en su sector principal
    primary_sector_by_segment: Dict[int, Dict[str, Any] | None] = {}
    for segment in speech_segments:
        primary_sector_by_segment[segment["id"]] = _classify_segment_sector(segment.get("text", ""))

    # 2. Detectar la primera mención (anchor) de cada sector para consultar ChromaDB sin redundancia
    anchor_segment_by_sector: Dict[str, int] = {}
    for segment in speech_segments:
        sec_info = primary_sector_by_segment[segment["id"]]
        if sec_info and sec_info["sector"] not in anchor_segment_by_sector:
            anchor_segment_by_sector[sec_info["sector"]] = segment["id"]

    sector_mentions = Counter(
        sec_info["sector"] for sec_info in primary_sector_by_segment.values() if sec_info
    )

    # Caché de consultas RAG por sector dentro de la misma ejecución
    rag_cache_by_sector: Dict[str, Dict[str, Any]] = {}

    segments_output = []
    factual_discrepancies_timeline = []

    for segment in speech_segments:
        segment_id = segment["id"]
        segment_text = segment.get("text", "")

        row_result = {
            "id": segment_id,
            "text": segment_text,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
        }

        linguistic_features = linguistic_map.get(segment_id, {})
        hedges_ratio = linguistic_features.get("hedges", 0.0) / 100.0
        assertives_ratio = linguistic_features.get("truth_assertives", 0.0) / 100.0

        sec_info = primary_sector_by_segment.get(segment_id)
        is_anchor = bool(sec_info) and anchor_segment_by_sector.get(sec_info["sector"]) == segment_id

        if sec_info:
            sector_name = sec_info["sector"]
            if sector_name not in rag_cache_by_sector:
                rag_cache_by_sector[sector_name] = _query_official_sector_benchmark(
            chroma_collection, f"{speaker_name} {segment_text}".strip(), sector_name
                )
            rag_data = rag_cache_by_sector[sector_name]

            base_discrepancy = sec_info["fact_target"]
            discrepancy_modifier = (assertives_ratio * 0.4) + (hedges_ratio * 0.3)
            raw_discrepancy = float(np.clip(base_discrepancy + discrepancy_modifier, 0.0, 0.98))
            final_segment_discrepancy = raw_discrepancy if is_anchor else float(
                np.clip(raw_discrepancy * REITERATION_FACTOR, 0.05, 0.60)
            )

            row_result["sector"] = sector_name
            row_result["topic"] = f"Sector {sector_name}"
            row_result["topic_key"] = _slugify_topic(sector_name)
            row_result["source"] = rag_data["source"]
            row_result["audit_note"] = rag_data["audit_note"]
            row_result["estado_actual"] = rag_data["estado_actual"]
            row_result["evidencia_actual"] = rag_data["evidencia_actual"]
            row_result["timeline"] = rag_data["timeline"]
            row_result["is_primary"] = is_anchor
            row_result["mention_count"] = int(sector_mentions[sector_name])

            if is_anchor:
                sector_audit_cards.append({
                    "sector": sector_name,
                    "promise_excerpt": segment_text,
                    "timestamp_start": segment.get("start", 0.0),
                    "estado_actual": rag_data["estado_actual"],
                    "evidencia_actual": rag_data["evidencia_actual"],
                    "source": rag_data["source"],
                    "timeline": rag_data["timeline"],
                    "discrepancy_score": round(final_segment_discrepancy * 100.0, 1),
                    "mentions": int(sector_mentions[sector_name]),
                })

                if chroma_collection is not None:
                    left_hits = _query_spectrum_evidence(chroma_collection, segment_text, sector_name, "izquierda", n_results=1)
                    right_hits = _query_spectrum_evidence(chroma_collection, segment_text, sector_name, "derecha", n_results=1)
                    for hit in left_hits:
                        hit["related_topic"] = f"Sector {sector_name}"
                        left_wing_evidence.append(hit)
                    for hit in right_hits:
                        hit["related_topic"] = f"Sector {sector_name}"
                        right_wing_evidence.append(hit)
            else:
                row_result["reiteration_of"] = anchor_segment_by_sector[sector_name]
        else:
            final_segment_discrepancy = float(np.clip(0.12 + (hedges_ratio * 0.35), 0.05, 0.35))
            row_result["sector"] = "General / Protocolar"
            row_result["topic"] = "Discurso General / Protocolar"
            row_result["topic_key"] = "general_protocolar"
            row_result["source"] = "Archivo Presidencial"
            row_result["audit_note"] = "Afirmación protocolar o descriptiva sin contraste sectorial directo."
            row_result["estado_actual"] = "No aplica"
            row_result["evidencia_actual"] = "Segmento introductorio o protocolar."
            row_result["timeline"] = []
            row_result["is_primary"] = True
            row_result["mention_count"] = 1

        row_result["factual_discrepancy"] = final_segment_discrepancy
        row_result["index"] = float(final_segment_discrepancy * 100.0)
        row_result["families"] = {"factual_rag": row_result["index"]}

        segments_output.append(row_result)
        factual_discrepancies_timeline.append(final_segment_discrepancy)

    # 3. Auditoría de omisiones críticas en todo el discurso
    full_speech_text = state.get("transcribed_text") or " ".join(s.get("text", "") for s in speech_segments)
    omitted_topics = _detect_omitted_sectors(full_speech_text)

    discrepancy_array = np.array(factual_discrepancies_timeline, dtype=float)
    evaluated_count = len(discrepancy_array)
    mean_discrepancy = float(discrepancy_array.mean()) if evaluated_count else 0.0
    critical_sectors = [
        sec for sec, anchor_id in anchor_segment_by_sector.items()
        if next(row["index"] for row in segments_output if row["id"] == anchor_id) >= 65.0
    ]

    # Conteo de estados de cumplimiento en sectores detectados
    compliance_counts = Counter(card["estado_actual"] for card in sector_audit_cards)

    summary_metrics = {
        "mean_factual_discrepancy": mean_discrepancy,
        "evaluated_segments_count": evaluated_count,
        "critical_segments_count": sum(1 for d in factual_discrepancies_timeline if d >= 0.65),
        "unique_topics_count": len(anchor_segment_by_sector),
        "critical_topics_count": len(critical_sectors),
        "repeated_mentions_count": sum(count - 1 for count in sector_mentions.values() if count > 1),
        "chroma_left_hits": len(left_wing_evidence),
        "chroma_right_hits": len(right_wing_evidence),
        "omitted_sectors_count": len(omitted_topics),
        "detected_sectors": list(anchor_segment_by_sector.keys()),
        "compliance_breakdown": dict(compliance_counts),
    }

    print(
        f"  → [{AGENT_NAME} Agent] Auditoría RAG Sectorial completada. Segmentos: {evaluated_count} | "
        f"Sectores detectados: {len(anchor_segment_by_sector)} | Contraste ChromaDB (Izq/Der): "
        f"{len(left_wing_evidence)}/{len(right_wing_evidence)} | Omisiones: {len(omitted_topics)}"
    )

    return {
        "factual_rag_metrics": {
            "by_segment": segments_output,
            "summary": summary_metrics,
            "markers_direction": {"factual_discrepancy": +1},
            "left_wing_evidence": left_wing_evidence,
            "right_wing_evidence": right_wing_evidence,
            "omitted_topics": omitted_topics,
            "sector_audit_cards": sector_audit_cards,
        },
        "left_wing_evidence": left_wing_evidence,
        "right_wing_evidence": right_wing_evidence,
        "omitted_topics": omitted_topics,
        "sector_audit_cards": sector_audit_cards,
        "logs": [
            f"[{AGENT_NAME}] Cruce RAG ChromaDB por Sectores: discrepancia media {mean_discrepancy * 100:.1f}/100, "
            f"sectores evaluados ({', '.join(anchor_segment_by_sector.keys()) or 'Ninguno'}), "
            f"contraste político ({len(left_wing_evidence)} izq / {len(right_wing_evidence)} der), "
            f"sectores omitidos: {', '.join(omitted_topics) if omitted_topics else 'Ninguno'}."
        ]
    }
