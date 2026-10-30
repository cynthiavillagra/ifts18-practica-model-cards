"""Carga model_savedmodel/ y model.tflite y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # SavedModel y TFLite son formatos de TensorFlow

from pathlib import Path

import numpy as np
import tensorflow as tf

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")

# 1. SavedModel: se carga con TensorFlow, sin Keras
savedmodel = tf.saved_model.load(str(CARPETA / "model_savedmodel"))
print("SavedModel:", savedmodel.serve(casas_nuevas).numpy().ravel().round(1))

# 2. TFLite: se carga con un intérprete, como en un celular
interprete = tf.lite.Interpreter(model_path=str(CARPETA / "model.tflite"))
entrada = interprete.get_input_details()[0]["index"]
interprete.resize_tensor_input(entrada, casas_nuevas.shape)
interprete.allocate_tensors()
interprete.set_tensor(entrada, casas_nuevas)
interprete.invoke()
precios = interprete.get_tensor(interprete.get_output_details()[0]["index"])
print("TFLite:")

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
