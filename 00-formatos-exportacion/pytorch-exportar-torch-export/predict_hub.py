"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.pt2 + config.json. El modelo vive en el Hub; la inferencia la hace este código.

No hace falta la clase RegresionLineal: se carga el programa exportado.

Uso:
    python predict_hub.py tu-usuario/casas-pytorch-exportar-torch-export
"""

import json
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-pytorch-exportar-torch-export"

# La primera vez descarga el archivo; después usa la copia en caché
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
with open(hf_hub_download(repo_id=REPO_ID, filename="config.json"), encoding="utf-8") as f:
    config = json.load(f)
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")
x = torch.tensor((casas_nuevas - media) / desvio)

modelo = torch.export.load(hf_hub_download(repo_id=REPO_ID, filename="model.pt2")).module()
with torch.no_grad():
    precios = modelo(x).numpy()

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
