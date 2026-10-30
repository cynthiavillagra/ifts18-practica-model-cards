"""Exporta el modelo de Keras en formato .weights.h5: model.weights.h5.

No vuelve a entrenar: parte de ../keras-entrenamiento-evaluacion/model.keras
(correr antes ese train.py).

Uso:
    python exportar.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # SavedModel y TFLite son formatos de TensorFlow

from pathlib import Path

import keras

CARPETA = Path(__file__).parent
modelo = keras.saving.load_model(CARPETA.parent / "keras-entrenamiento-evaluacion" / "model.keras")

# Guarda solo los pesos (incluida la media y el desvío de la capa Normalization)
modelo.save_weights(CARPETA / "model.weights.h5")
print("Modelo guardado en model.weights.h5")
