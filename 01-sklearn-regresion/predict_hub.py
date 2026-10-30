"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC (o en el servidor donde corra).

El modelo vive en el Hub; la inferencia la hace este código.

Uso:
    python predict_hub.py tu-usuario/regresion-casas-sklearn
"""

import sys

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/regresion-casas-sklearn"

# 1. Bajar el archivo del modelo (la primera vez lo descarga; después usa la copia en caché)
ruta = hf_hub_download(repo_id=REPO_ID, filename="model.joblib")
print("Modelo descargado en:", ruta)

# 2. Cargarlo y predecir, igual que con el archivo local
modelo = joblib.load(ruta)
casas_nuevas = pd.DataFrame({"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]})
for casa, precio in zip(casas_nuevas.to_dict("records"), modelo.predict(casas_nuevas)):
    print(f"{casa} -> {precio:.1f} miles de US$")
