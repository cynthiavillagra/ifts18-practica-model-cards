"""Carga model.keras y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python predict.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import keras
import numpy as np

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]

modelo = keras.saving.load_model(CARPETA / "model.keras")

# No hace falta normalizar a mano: la capa Normalization viene dentro del modelo
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
precios = modelo.predict(casas_nuevas, verbose=0).ravel()

for casa, precio in zip(casas_nuevas, precios):
    print(f"{dict(zip(ENTRADAS, casa.astype(int).tolist()))} -> {precio:.1f} miles de US$")
