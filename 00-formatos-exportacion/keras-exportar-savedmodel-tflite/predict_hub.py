"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.tflite. El modelo vive en el Hub; la inferencia la hace este código.

Usa el archivo TFLite, el que iría en un celular. No hace falta Keras.

Uso:
    python predict_hub.py tu-usuario/casas-keras-exportar-savedmodel-tflite
"""

import sys
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-keras-exportar-savedmodel-tflite"

# La primera vez descarga el archivo; después usa la copia en caché
interprete = tf.lite.Interpreter(model_path=hf_hub_download(repo_id=REPO_ID, filename="model.tflite"))
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
entrada = interprete.get_input_details()[0]["index"]
interprete.resize_tensor_input(entrada, casas_nuevas.shape)
interprete.allocate_tensors()
interprete.set_tensor(entrada, casas_nuevas)
interprete.invoke()
precios = interprete.get_tensor(interprete.get_output_details()[0]["index"])

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
