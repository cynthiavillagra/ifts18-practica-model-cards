"""Exporta el modelo de scikit-learn en formato pickle: model.pkl.

No vuelve a entrenar: parte de ../sklearn-entrenamiento-evaluacion/model.joblib
(correr antes ese train.py).

Uso:
    python exportar.py
"""

from pathlib import Path

import joblib
import pickle

CARPETA = Path(__file__).parent
modelo = joblib.load(CARPETA.parent / "sklearn-entrenamiento-evaluacion" / "model.joblib")

with open(CARPETA / "model.pkl", "wb") as f:
    pickle.dump(modelo, f)  # guarda el objeto entero de Python
print("Modelo guardado en model.pkl")
