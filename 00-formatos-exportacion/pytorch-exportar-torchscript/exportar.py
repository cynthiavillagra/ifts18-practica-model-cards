"""Exporta el modelo de PyTorch en formato TorchScript: model_ts.pt.

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

# trace pasa un ejemplo por el modelo y graba las operaciones que se ejecutan.
# En PyTorch nuevo muestra un aviso de deprecated: recomiendan torch.export.
ejemplo = torch.zeros(1, 3)
torch.jit.trace(modelo, ejemplo).save(CARPETA / "model_ts.pt")
print("Modelo guardado en model_ts.pt")

# Se copian al lado del modelo: así la carpeta se puede subir sola al Hub
for archivo in ["config.json"]:
    shutil.copy(ORIGEN / archivo, CARPETA / archivo)
print("Copiados: config.json")
