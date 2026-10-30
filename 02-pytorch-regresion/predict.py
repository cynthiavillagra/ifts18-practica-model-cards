"""Carga model.safetensors + config.json y predice el precio de casas nuevas.

Uso:
    python predict.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from modelo import RegresionLineal

CARPETA = Path(__file__).parent

# 1. Leer la configuración (entradas y normalización)
config = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")

# 2. Crear el modelo vacío y cargarle los pesos entrenados
modelo = RegresionLineal(n_entradas=len(config["entradas"]))
modelo.load_state_dict(load_file(CARPETA / "model.safetensors"))
modelo.eval()

# 3. Predecir: las entradas se normalizan igual que en el entrenamiento
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
with torch.no_grad():
    precios = modelo(torch.tensor((casas_nuevas - media) / desvio)).numpy().ravel()

for casa, precio in zip(casas_nuevas, precios):
    print(f"{dict(zip(config['entradas'], casa.astype(int).tolist()))} -> {precio:.1f} miles de US$")
