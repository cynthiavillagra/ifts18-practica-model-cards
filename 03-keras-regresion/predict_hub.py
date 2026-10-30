"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC (o en el servidor donde corra).

Uso:
    python predict_hub.py tu-usuario/regresion-casas-keras
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import sys

import keras
import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/regresion-casas-keras"
ENTRADAS = ["m2", "balcon", "ascensor"]

# 1. Bajar el archivo .keras
#    (ojo: keras.saving.load_model("hf://...") espera otro formato de guardado; con un
#    model.keras subido así, se descarga con hf_hub_download y se carga como archivo)
ruta = hf_hub_download(repo_id=REPO_ID, filename="model.keras")

# 2. Cargarlo y predecir: la normalización viene adentro del modelo
modelo = keras.saving.load_model(ruta)
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
for casa, precio in zip(casas_nuevas, modelo.predict(casas_nuevas, verbose=0).ravel()):
    print(f"{dict(zip(ENTRADAS, casa.astype(int).tolist()))} -> {precio:.1f} miles de US$")
