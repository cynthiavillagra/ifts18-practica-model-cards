"""Exporta el modelo de scikit-learn en formato ONNX: model.onnx.

No vuelve a entrenar: parte de ../sklearn-entrenamiento-evaluacion/model.joblib
(correr antes ese train.py).

Uso:
    python exportar.py
"""

from pathlib import Path

import joblib
import numpy as np
from skl2onnx import to_onnx

CARPETA = Path(__file__).parent
modelo = joblib.load(CARPETA.parent / "sklearn-entrenamiento-evaluacion" / "model.joblib")

# ONNX traduce el modelo a un grafo de operaciones. Necesita un ejemplo de entrada
# para saber la forma y el tipo de los datos (3 columnas, float32).
ejemplo = np.zeros((1, 3), dtype="float32")
onx = to_onnx(modelo, ejemplo)
(CARPETA / "model.onnx").write_bytes(onx.SerializeToString())
print("Modelo guardado en model.onnx")
