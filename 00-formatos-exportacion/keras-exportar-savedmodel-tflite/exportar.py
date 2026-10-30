"""Exporta el modelo de Keras en formato SavedModel y TFLite: model_savedmodel/ y model.tflite.

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

import tensorflow as tf  # noqa: E402

# 1. SavedModel: carpeta con el grafo de TensorFlow y los pesos
modelo.export(CARPETA / "model_savedmodel", verbose=False)
print("Modelo guardado en model_savedmodel/")

# 2. TFLite: se convierte a partir del SavedModel. Es un archivo chico para celulares.
tflite = tf.lite.TFLiteConverter.from_saved_model(str(CARPETA / "model_savedmodel")).convert()
(CARPETA / "model.tflite").write_bytes(tflite)
print("Modelo guardado en model.tflite")
