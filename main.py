"""Analizador multiagente de discursos presidenciales.

Uso:
  python main.py analyze videos/discurso.mp4 [--speaker "Name"] [--output results]
  python main.py analyze videos/                     # all videos in folder
  python main.py train labels.csv                    # trains the core model
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import cargar_config

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def _yield_valid_videos(target_paths: list[str]):
    """Recursively scans and yields supported video files from directories or direct arguments."""
    for path_str in target_paths:
        target_path = Path(path_str)
        if target_path.is_dir():
            yield from sorted(x for x in target_path.iterdir() if x.suffix.lower() in VIDEO_EXTENSIONS)
        else:
            yield target_path


def cmd_analyze(args: argparse.Namespace):
    """Executes the operational LangGraph multi-agent forensic analysis pipeline."""
    # Sincronizado con el nuevo portal en inglés del orquestador
    from src.orchestrator import analyze_video_pipeline

    system_config = cargar_config(args.config)
    for video in _yield_valid_videos(args.videos):
        print(f"\n=== Ejecutando Análisis Factual: {video.name} ===")
        
        # Corrección Crítica: Pasamos el argumento 'video_path' explícito como requiere el orquestador
        state_result = analyze_video_pipeline(
            video_path=str(video), 
            config=system_config, 
            output_dir=args.output, 
            speaker=args.speaker, 
            verbose=not args.silent
        )
        
        verdict = state_result.get("verdict", {})
        opacity_index = verdict.get("opacity_index", 0.0)
        
        # Impresiones en consola formateadas para la defensa del MVP
        print(f"\nResultado  : {verdict.get('label', 'N/A')}")  # Muestra: 'truth' o 'deviation'
        print(f"Método     : {verdict.get('method', 'Multi-Agent RAG Pipeline')}")
        
        if verdict.get("deviation_probability") is not None:
            print(f"P(desvío)  : {verdict['deviation_probability'] * 100:.1f}%")
            
        # Índice de Opacidad Sintáctica / Gunning Fog adaptado
        if opacity_index == opacity_index and not math.isinf(opacity_index):
            print(f"Índice IOF : {opacity_index:.1f}/100")
        else:
            print("Índice IOF : n/d")
            
        print(f"Confianza  : {verdict.get('confidence', 0) * 100:.0f}%")
        print(f"Reporte HTML: {state_result.get('report', {}).get('html', 'Pending')}")
        
        # Sistema de degradación controlada: Alerta de fallos menores en agentes
        for error in state_result.get("errors", []):
            print(f"  [Aviso de Degradación]: {error}")


def cmd_train(args: argparse.Namespace):
    """Optimizes decision boundaries utilizing historical MEF/BCRP consistency training data."""
    # Sincronizado con el nuevo portal de entrenamiento en inglés
    from src.training import train_classifier

    train_classifier(
        args.csv, cargar_config(args.config), args.output, args.model, reuse_cache=not args.recalculate
    )


def main(argv: list[str] | None = None) -> int:
    """Configures input argument sub-parsers in clean programmatic English syntax."""
    parser = argparse.ArgumentParser(description="Multi-Agent Agéntico Speech Analyzer Framework")
    parser.add_argument("--config", help="Optional custom JSON configuration parameters override")
    
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Command mapping: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Executes multi-agent analysis on target video corpus")
    analyze_parser.add_argument("videos", nargs="+")
    analyze_parser.add_argument("--speaker")
    analyze_parser.add_argument("--output", default="results")
    analyze_parser.add_argument("--silent", action="store_true")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Command mapping: train
    train_parser = subparsers.add_parser("train", help="Optimizes decision metrics via factual deviation csv matrices")
    train_parser.add_argument("csv")
    train_parser.add_argument("--output", default="results")
    train_parser.add_argument("--model", default="models/classifier.joblib")
    train_parser.add_argument("--recalculate", action="store_true", help="Forces recalculation over cache")
    train_parser.set_defaults(func=cmd_train)

    parsed_args = parser.parse_args(argv)
    parsed_args.func(parsed_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())