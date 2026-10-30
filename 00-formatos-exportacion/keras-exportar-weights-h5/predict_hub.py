"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.weights.h5. El modelo vive en el Hub; la inferencia la hace este código.

El archivo trae solo los pesos: la arquitectura se vuelve a armar en el código.

Uso:
    python predict_hub.py tu-usuario/casas-keras-exportar-weights-h5
"""

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-keras-exportar-weights-h5"

# La primera vez descarga el archivo; después usa la copia en caché
modelo = keras.Sequential([
    keras.Input(shape=(3,)),
    keras.layers.Normalization(),
    keras.layers.Dense(1),
])
modelo.load_weights(hf_hub_download(repo_id=REPO_ID, filename="model.weights.h5"))
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
precios = modelo.predict(casas_nuevas, verbose=0)

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
