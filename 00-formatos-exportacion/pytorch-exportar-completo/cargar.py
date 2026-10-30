"""Carga model_completo.pt y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

import json
from pathlib import Path

import numpy as np
import torch

CARPETA = Path(__file__).parent
from modelo import RegresionLineal  # noqa: F401  (pickle la necesita para reconstruir el objeto)
ENTRADAS = ["m2", "balcon", "ascensor"]

# La normalización no viaja dentro del archivo: se lee de config.json
config = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")

casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
x = torch.tensor((casas_nuevas - media) / desvio)

# weights_only=False: el archivo usa pickle, por eso PyTorch pide confirmarlo.
# Además la clase RegresionLineal tiene que poder importarse.
modelo = torch.load(CARPETA / "model_completo.pt", weights_only=False)
modelo.eval()
with torch.no_grad():
    precios = modelo(x).numpy()

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
