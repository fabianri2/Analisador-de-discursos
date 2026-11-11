import os
import re
import chromadb
from youtube_transcript_api import YouTubeTranscriptApi

CHROMA_DB_PATH = os.path.join("data_storage", "vector_chroma")
COLLECTION_NAME = "political_discourse_sources"

# Lista de fuentes web de YouTube 
YOUTUBE_SOURCES = [
    {
        "url_or_id": "https://www.youtube.com/watch?v=fDwQR8a9fmY",
        "espectro": "izquierda",
        "fuente": "Podcast Político (YouTube)"
    },
    {
        "url_or_id": "https://www.youtube.com/watch?v=sHs3L6YpX00",
        "espectro": "derecha",
        "fuente": "Prensa Nacional (YouTube)"
    }
]

CURATED_PRESS_SOURCES = [
    # --- ESPECTRO IZQUIERDA / CRÍTICA ---
    {
        "text": "El Ejecutivo condicionó el adelanto de elecciones al Congreso y evadió su responsabilidad política frente a los fallecidos en las protestas del sur andino.",
        "espectro": "izquierda",
        "fuente": "Análisis Crítico / Canal Curwen (YouTube)"
    },
    {
        "text": "Prometer una reforma total de la Constitución desde el Congreso actual es una maniobra dilatoria para postergar los comicios generales hasta el 2026.",
        "espectro": "izquierda",
        "fuente": "Análisis Crítico / Canal Curwen (YouTube)"
    },
    {
        "text": "La declaración de estados de emergencia y el despliegue policial agravaron la conflictividad social sin abrir canales reales de diálogo con las regiones.",
        "espectro": "izquierda",
        "fuente": "Análisis Crítico / Canal Curwen (YouTube)"
    },
    # --- ESPECTRO DERECHA / PRENSA TRADICIONAL ---
    {
        "text": "La presidenta presentó una iniciativa legislativa con carácter de urgencia para adelantar las elecciones generales a 2023 y pacificar el país.",
        "espectro": "derecha",
        "fuente": "Juliana Oxenford / Análisis de Coyuntura"
    },
    {
        "text": "El debate sobre reformas constitucionales debe canalizarse estrictamente dentro del marco democrático del Congreso sin ceder ante presiones de una Asamblea Constituyente.",
        "espectro": "derecha",
        "fuente": "Juliana Oxenford / Análisis de Coyuntura"
    },
    {
        "text": "La intervención de la Policía Nacional y las Fuerzas Armadas resulta indispensable para desbloquear carreteras y garantizar el abastecimiento y el orden interno.",
        "espectro": "derecha",
        "fuente": "Juliana Oxenford / Análisis de Coyuntura"
    }
]


def _extract_video_id(url_or_id: str) -> str:
    """Extracts the 11-character YouTube video ID from a full URL or raw ID."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def _fetch_youtube_chunks(video_id: str, chunk_size: int = 450) -> list[str]:
    """Downloads Spanish transcripts from YouTube and groups them into clean semantic chunks."""
    try:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id, languages=["es", "es-419", "es-PE"])
        full_text = " ".join(snippet.text.replace("\n", " ") for snippet in transcript_obj)
        words = full_text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as exc:
        print(f"  ⚠️ No se pudo extraer subtítulos del video '{video_id}' ({exc}). Se usará respaldo documental.")
        return []



def build_vector_database():
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = []
    metadatas = []
    ids = []

    # 1. Cargar fuentes curadas de prensa escrita y análisis
    for idx, item in enumerate(CURATED_PRESS_SOURCES, start=1):
        documents.append(item["text"])
        metadatas.append({"espectro": item["espectro"], "fuente": item["fuente"]})
        ids.append(f"prensa_{item['espectro']}_{idx}")

    # 2. Extraer transcripciones automáticas desde YouTube
    yt_counter = 1
    for source in YOUTUBE_SOURCES:
        vid = _extract_video_id(source["url_or_id"])
        chunks = _fetch_youtube_chunks(vid)
        for chunk in chunks[:6]:  # Limitamos a los primeros 6 bloques relevantes por video
            documents.append(chunk)
            metadatas.append({"espectro": source["espectro"], "fuente": source["fuente"]})
            ids.append(f"yt_{source['espectro']}_{vid}_{yt_counter}")
            yt_counter += 1

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Base de datos ChromaDB actualizada con {collection.count()} fragmentos en '{CHROMA_DB_PATH}'.")


if __name__ == "__main__":
    build_vector_database()

