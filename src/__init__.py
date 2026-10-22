"""Analizador multiagente de discursos presidenciales."""
from __future__ import annotations

import sys

# Los agentes imprimen trazas con flechas (→) y acentos. En consolas Windows con
# cp1252 (o cuando la salida se redirige a un archivo) eso lanzaba UnicodeEncodeError.
# main.py ya lo configuraba, pero los tests y scripts que importan 'src' directamente no.
for _output_stream in (sys.stdout, sys.stderr):
    if hasattr(_output_stream, "reconfigure"):
        try:
            _output_stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
