"""Carga model.pkl y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pickle

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
casas_nuevas = pd.DataFrame({"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]})

# Ojo: cargar un pickle puede ejecutar código. Solo abrir archivos de confianza.
with open(CARPETA / "model.pkl", "rb") as f:
    modelo = pickle.load(f)
precios = modelo.predict(casas_nuevas)

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
