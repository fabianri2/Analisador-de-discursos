"""Descarga los modelos faciales.

- face_landmarker.task (MediaPipe): 478 puntos + 52 blendshapes. Motor principal.
- face_detection_yunet.onnx (OpenCV): detector de respaldo con 5 puntos.
"""
from pathlib import Path
from urllib.request import urlretrieve

MODELOS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                            "face_landmarker/float16/1/face_landmarker.task",
    "face_detection_yunet.onnx": "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                                 "main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}
CARPETA = Path(__file__).resolve().parents[1] / "modelos"

if __name__ == "__main__":
    CARPETA.mkdir(parents=True, exist_ok=True)
    for nombre, url in MODELOS.items():
        destino = CARPETA / nombre
        if destino.exists() and destino.stat().st_size > 10_000:
            print(f"Ya existe {nombre}")
            continue
        print(f"Descargando {nombre}...")
        try:
            urlretrieve(url, destino)
            print(f"  OK ({destino.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"  No se pudo descargar: {e}")
