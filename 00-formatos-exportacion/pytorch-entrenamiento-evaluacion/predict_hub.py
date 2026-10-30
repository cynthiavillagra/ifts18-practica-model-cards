"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.safetensors + config.json. El modelo vive en el Hub; la inferencia la hace este código.

La arquitectura (RegresionLineal) tiene que estar en tu código: por eso se importa modelo.py.

Uso:
    python predict_hub.py tu-usuario/casas-pytorch-entrenamiento-evaluacion
"""

import json
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from modelo import RegresionLineal

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-pytorch-entrenamiento-evaluacion"

# La primera vez descarga el archivo; después usa la copia en caché
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
with open(hf_hub_download(repo_id=REPO_ID, filename="config.json"), encoding="utf-8") as f:
    config = json.load(f)
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")
x = torch.tensor((casas_nuevas - media) / desvio)

modelo = RegresionLineal(n_entradas=len(config["entradas"]))
modelo.load_state_dict(load_file(hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")))
modelo.eval()
with torch.no_grad():
    precios = modelo(x).numpy()

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
