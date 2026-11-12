"""Pruebas de ingesta cuando la entrada es un archivo de subtítulos.

La interfaz lista archivos .srt/.vtt del corpus; la ingesta debe aceptarlos
sin exigir una pista de audio.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agentes.ingestion_agent import ingestion_agent
from src.agentes.transcription_agent import transcription_agent
from src.config import cargar_config


class SubtitleIngestionTest(unittest.TestCase):
    def _state(self, working_dir: str) -> dict:
        return {
            "video": "videos/discurso.srt",
            "working_dir": working_dir,
            "config": cargar_config(None),
        }

    def test_srt_input_skips_audio_and_exposes_duration(self):
        state = self._state("results/_test_ingesta")
        output = ingestion_agent(state)

        self.assertIsNone(output["audio"])
        self.assertEqual(output["metadata"]["format_name"], "subtitle")
        self.assertFalse(output["metadata"]["has_audio_track"])
        self.assertGreater(output["duration"], 0.0)

    def test_transcription_reads_the_subtitle_file(self):
        state = self._state("results/_test_ingesta")
        state.update(ingestion_agent(state))
        transcription = transcription_agent(state)

        self.assertTrue(transcription["segments"])
        self.assertIn("subtitles", transcription["transcription_source"])


if __name__ == "__main__":
    unittest.main()
