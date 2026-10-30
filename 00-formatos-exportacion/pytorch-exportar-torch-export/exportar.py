"""Exporta el modelo de PyTorch en formato torch.export: model.pt2.

No vuelve a entrenar: parte de ../pytorch-entrenamiento-evaluacion/model.safetensors
(correr antes ese train.py).

Uso:
    python exportar.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

CARPETA = Path(__file__).parent
ORIGEN = Path(__file__).parent.parent / "pytorch-entrenamiento-evaluacion"
sys.path.insert(0, str(ORIGEN))
from modelo import RegresionLineal  # noqa: E402

# La normalización no viaja dentro del archivo: se lee de config.json
config = json.loads((ORIGEN / "config.json").read_text(encoding="utf-8"))
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")

modelo = RegresionLineal(n_entradas=len(config["entradas"]))
modelo.load_state_dict(load_file(ORIGEN / "model.safetensors"))
modelo.eval()

# torch.export captura el grafo de operaciones con un ejemplo de entrada.
# Dim("casas") deja variable la cantidad de filas (con 1 sola fila quedaría fija).
ejemplo = torch.zeros(2, 3)
programa = torch.export.export(modelo, (ejemplo,), dynamic_shapes=({0: torch.export.Dim("casas")},))
torch.export.save(programa, CARPETA / "model.pt2")
print("Modelo guardado en model.pt2")

# Se copian al lado del modelo: así la carpeta se puede subir sola al Hub
for archivo in ["config.json"]:
    shutil.copy(ORIGEN / archivo, CARPETA / archivo)
print("Copiados: config.json")
