"""Exporta el modelo de scikit-learn en formato skops: model.skops.

No vuelve a entrenar: parte de ../sklearn-entrenamiento-evaluacion/model.joblib
(correr antes ese train.py).

Uso:
    python exportar.py
"""

from pathlib import Path

import joblib
import skops.io as sio

CARPETA = Path(__file__).parent
modelo = joblib.load(CARPETA.parent / "sklearn-entrenamiento-evaluacion" / "model.joblib")

sio.dump(modelo, CARPETA / "model.skops")  # guarda el objeto sin usar pickle
print("Modelo guardado en model.skops")
