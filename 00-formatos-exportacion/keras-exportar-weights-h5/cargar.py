"""Carga model.weights.h5 y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # SavedModel y TFLite son formatos de TensorFlow

from pathlib import Path

import numpy as np
import keras

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")

# El archivo no tiene la arquitectura: hay que volver a armar el mismo modelo vacío
modelo = keras.Sequential([
    keras.Input(shape=(len(ENTRADAS),)),
    keras.layers.Normalization(),
    keras.layers.Dense(1),
])
modelo.load_weights(CARPETA / "model.weights.h5")
precios = modelo.predict(casas_nuevas, verbose=0)

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
