"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.keras. El modelo vive en el Hub; la inferencia la hace este código.

Uso:
    python predict_hub.py tu-usuario/casas-keras-entrenamiento-evaluacion
"""

import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-keras-entrenamiento-evaluacion"

# La primera vez descarga el archivo; después usa la copia en caché
modelo = keras.saving.load_model(hf_hub_download(repo_id=REPO_ID, filename="model.keras"))
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
precios = modelo.predict(casas_nuevas, verbose=0)

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
