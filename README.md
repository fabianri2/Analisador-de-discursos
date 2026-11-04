# Analizador Agéntico Multiagente de Discursos Presidenciales con Arquitectura RAG

Sistema avanzado de ingeniería y auditoría forense computacional desarrollado en Python utilizando **LangGraph** y **RAG (Generación Aumentada por Recuperación)** para la verificación automatizada de consistencia socioeconómica y cumplimiento de promesas en discursos presidenciales peruanos (2016 - 2026). El sistema contrasta de forma 100% automatizada las promesas cuantitativas del ejecutivo (Año N) contra la evidencia factual de los reportes oficiales del **MEF** y las memorias del **BCRP** (Año N+1).

> **Descargo de Responsabilidad Metodológica:** Esta plataforma opera como una herramienta algorítmica de auditoría documental. Un índice de opacidad o desviación elevado aísla discrepancias estadísticas entre las metas políticas del orador y la realidad fría de las partidas presupuestales de ejecución del Estado, eliminando cualquier sesgo u opinión humana.

---

## Arquitectura del Sistema

La infraestructura opera de forma asíncrona mediante un grafo de ejecución paralelo gestionado por LangGraph, garantizando un Atributo de Calidad de rendimiento óptimo (Real-Time Factor RTF ≤ 0.5):

┌─> Sentiment Agent ──┐
Ingestion -> Transcription ─> Linguistic Agent ─┤
├─> Promise Agent ────┼─> Factual Agent -> Judge -> Report
└─> Syntactic Agent ──┘

---

## Matriz de Agentes y Stack Tecnológico

| Agente (Nodo) | Responsabilidad / Tarea Forense Computacional | Stack Tecnológico |
| :--- | :--- | :--- |
| **Ingestion** | Validación del archivo multimedia y extracción de la pista de voz limpia en formato PCM mono a 16 kHz. | `ffmpeg` / `ffprobe` / `subprocess` |
| **Transcription** | Segmentación temporal del audio y conversión de voz a texto frase por frase con marcas de tiempo exactas. | `faster-whisper` (CTranslate2) / Subtítulos Locales |
| **Sentiment** | Medición de la polaridad continua segmento por segmento y cálculo de la volatilidad emocional del tono político. | `TextBlob` (Motor ligero de NLP) |
| **Linguistic** | Conteo de categorías semánticas (LIWC), densidad de pronombres en primera persona y sobre-afirmaciones de sinceridad. | Estructura LIWC nativa en Python |
| **Syntactic** | Análisis de la complejidad gramatical y cálculo del Índice de Niebla para evaluar la opacidad en la redacción del mensaje. | `spaCy` / Núcleo de Densidad Sintáctica |
| **Promise** | Aislamiento automatizado y estructuración en JSON de las micro-promesas, cifras, montos y ministerios implicados. | Reconocimiento de Patrones Gramaticales |
| **Factual** | Validación cruzada utilizando RAG para comparar el optimismo del discurso contra las metas macroeconómicas reales del año posterior. | Consultas vectoriales automatizadas |
| **Judge** | Arbitraje final del veredicto (`truth` o `deviation`) mediante un Random Forest clasificador u optimización heurística ponderada. | `scikit-learn` / `RandomForest` / `GroupKFold` |
| **Report** | Compilación de la bitácora del grafo y exportación de la base de datos `result.json` y del dashboard web interactivo `report.html`. | HTML5 / CSS3 variables adaptativas / SVG |

---

## Ejecución del Sistema (Comandos de Consola)

Asegúrate de tener activo tu entorno virtual (`.venv`) en la raíz del proyecto antes de ejecutar la aplicación:

### 1. Ejecutar el Análisis Forense Multiagente sobre un Discurso:
```bash
python main.py analyze videos/discurso.mp4 --output results --speaker "NombrePresidente"
```

### 2. Optimizar y Entrenar las Fronteras de Decisión del Clasificador (Machine Learning):
```bash
python main.py train datasets/etiquetas_factores.csv --model models/classifier.joblib
```

---

## Instalación e Infraestructura Local

Para desplegar el entorno de desarrollo modular en sus computadoras locales, ejecuten la siguiente secuencia en la terminal:

```powershell
# 1. Clonar el repositorio oficial e ingresar a la raíz
git clone <URL_DE_SU_REPOSITORIO>
cd analizador_discursos

# 2. Inicializar y activar el entorno virtual aislado (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# 3. Instalar el stack de dependencias analíticas y de procesamiento
pip install numpy pandas scikit-learn joblib langgraph streamlit textblob
```

> ⚠️ **Requisito del Sistema:** Es mandatorio tener instalado **`ffmpeg`** y configurado correctamente en el `PATH` del sistema de Windows para permitir que el agente de ingesta aísle el canal de audio PCM.

---

## Estructura de Datos y Resultados

Toda la ejecución agéntica genera de forma 100% automatizada una carpeta de resultados estructurada bajo la ruta `results/<nombre_del_video>/`:

*   **`report.html`**: Interfaz web estática e interactiva que despliega el dictamen consolidado, gráficos de líneas de tiempo SVG y la bitácora de ejecución del grafo.
*   **`result.json`**: Base de datos técnica que contiene la matriz completa de indicadores macroeconómicos y desvíos formateados para analistas.

---

## Modelamiento de Decisiones (Criterio Algorítmico)

El sistema procesa la información mediante dos canales paralelos independientes:

1.  **Modo Heurístico Documental (Por Defecto):** Evalúa el texto mediante 5 familias concurrentes (**Sentimiento, Densidad Lingüística, Opacidad Sintáctica, Extracción de Promesas y Auditoría Factual RAG**). Los pesos de análisis ponderan con un **30% de dominancia matemática al nodo `factual_rag`**, cruzando las cifras directamente con ChromaDB.
2.  **Mapeo de Umbrales del Dictamen:** El Índice de Opacidad final se normaliza en una escala estándar de 0 a 100:
    *   **Índice < 40.0** ➔ `truth` (Consistencia Factual demostrada por el RAG).
    *   **Índice > 65.0** ➔ `deviation` (Desviación Detectada / Discrepancia crítica con el MEF y BCRP).
    *   **40.0 ≤ Índice ≤ 65.0** ➔ `neutral` (Evaluación Inconclusa / Falta de datos oficiales en la base vectorial).

---

## Entrenamiento y Optimización del Clasificador

El sistema permite entrenar un clasificador de Machine Learning para optimizar la toma de decisiones basada en datos históricos de consistencia. El archivo de matriz (`.csv`) debe contener datos estructurados bajo el estándar del proyecto:

```csv
video,speaker,label
videos/mensaje_01.mp4,Presidente_A,truth
videos/mensaje_02.mp4,Presidente_A,deviation
videos/mensaje_03.mp4,Presidente_B,truth
```

La validación cruzada se ejecuta mediante la estrategia **`GroupKFold` agrupada por orador (speaker)**. Esto previene el sobreajuste y garantiza que el modelo aprenda patrones reales de desviación factual en el discurso y no los rasgos particulares de una persona. El pipeline compara automáticamente algoritmos de **Regresión Logística** y **Random Forest**, exportando el modelo con mejor AUC hacia la ruta `models/classifier.joblib`.

---

## Estructura del Repositorio

El árbol de directorios del proyecto sigue una arquitectura limpia y modular en estricto orden de ingeniería de software:

```text
├── main.py                     # Interfaz de Línea de Comandos (CLI Portal)
└── src/                        # Código fuente del sistema unificado
    ├── orchestrator.py         # Orquestador del flujo paralelo (LangGraph Core)
    ├── state.py                # Definición del esquema del estado compartido (TypedDict)
    ├── config.py               # Hiperparámetros del sistema y pesos del RAG
    ├── utils.py                # Funciones auxiliares de cálculo y z-score factual
    ├── training.py             # Pipeline de entrenamiento y validación cruzada
    └── agentes/                # Nódos especializados del Grafo Multiagente
        ├── ingestion_agent.py      # Extracción de audio PCM con ffmpeg
        ├── transcription_agent.py  # Conversión de voz a texto con faster-whisper
        ├── sentiment_agent.py      # Análisis de trayectoria emocional (TextBlob)
        ├── linguistic_agent.py     # Análisis semántico y marcadores lingüísticos
        ├── syntactic_agent.py      # Medición de opacidad e Índice de Niebla
        ├── promise_agent.py        # Extracción y estructuración JSON de metas
        ├── factual_agent.py        # Validación cruzada contra la base de datos RAG
        ├── judge_agent.py          # Árbitro de veredicto final e inferencia ML
        └── report_agent.py         # Generación del dashboard analítico interactivo
```

---

## Referencias

- Bond, C. F., y DePaulo, B. M. (2006). Accuracy of deception judgments. *Personality and Social Psychology Review, 10*(3), 214–234.
- DePaulo, B. M., et al. (2003). Cues to deception. *Psychological Bulletin, 129*(1), 74–118.
- Hauch, V., et al. (2015). Are computers effective lie detectors? A meta-analysis of linguistic cues to deception. *Personality and Social Psychology Review, 19*(4), 307–342.
- Newman, M. L., Pennebaker, J. W., Berry, D. S., y Richards, J. M. (2003). Lying words: Predicting deception from linguistic styles. *Personality and Social Psychology Bulletin, 29*(5), 665–675.
- Vrij, A. (2008). *Detecting lies and deceit: Pitfalls and opportunities* (2.ª ed.). Wiley.
