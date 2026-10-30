"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.pkl. El modelo vive en el Hub; la inferencia la hace este código.

Ojo: cargar un pickle puede ejecutar código. Solo con repos de confianza.

Uso:
    python predict_hub.py tu-usuario/casas-sklearn-exportar-pickle
"""

import pickle
import sys

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-sklearn-exportar-pickle"

# La primera vez descarga el archivo; después usa la copia en caché
with open(hf_hub_download(repo_id=REPO_ID, filename="model.pkl"), "rb") as f:
    modelo = pickle.load(f)
casas_nuevas = pd.DataFrame({"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]})
precios = modelo.predict(casas_nuevas)

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
