"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model_completo.pt + config.json. El modelo vive en el Hub; la inferencia la hace este código.

Ojo: usa pickle, solo con repos de confianza. La clase RegresionLineal tiene que
poder importarse: por eso se importa modelo.py (está en este repo).

Uso:
    python predict_hub.py tu-usuario/casas-pytorch-exportar-completo
"""

import json
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from modelo import RegresionLineal  # noqa: F401  (pickle la necesita)

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-pytorch-exportar-completo"

# La primera vez descarga el archivo; después usa la copia en caché
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
with open(hf_hub_download(repo_id=REPO_ID, filename="config.json"), encoding="utf-8") as f:
    config = json.load(f)
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")
x = torch.tensor((casas_nuevas - media) / desvio)

modelo = torch.load(hf_hub_download(repo_id=REPO_ID, filename="model_completo.pt"), weights_only=False)
modelo.eval()
with torch.no_grad():
    precios = modelo(x).numpy()

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
