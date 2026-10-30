"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC (o en el servidor donde corra).

En PyTorch del Hub se bajan los pesos y la configuración, pero la arquitectura
(la clase RegresionLineal) tiene que estar en tu código: por eso se importa modelo.py.

Uso:
    python predict_hub.py tu-usuario/regresion-casas-pytorch
"""

import json
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from modelo import RegresionLineal

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/regresion-casas-pytorch"

# 1. Bajar los dos archivos que hacen falta
ruta_pesos = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors")
ruta_config = hf_hub_download(repo_id=REPO_ID, filename="config.json")

# 2. Armar el modelo vacío, cargarle los pesos y leer la normalización
with open(ruta_config, encoding="utf-8") as f:
    config = json.load(f)
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")

modelo = RegresionLineal(n_entradas=len(config["entradas"]))
modelo.load_state_dict(load_file(ruta_pesos))
modelo.eval()

# 3. Predecir
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
with torch.no_grad():
    precios = modelo(torch.tensor((casas_nuevas - media) / desvio)).numpy().ravel()
for casa, precio in zip(casas_nuevas, precios):
    print(f"{dict(zip(config['entradas'], casa.astype(int).tolist()))} -> {precio:.1f} miles de US$")
