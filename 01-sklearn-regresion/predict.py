"""Carga model.joblib y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python predict.py
"""

from pathlib import Path

import joblib
import pandas as pd

CARPETA = Path(__file__).parent

modelo = joblib.load(CARPETA / "model.joblib")

# Las columnas tienen que llamarse igual que en el entrenamiento
casas_nuevas = pd.DataFrame(
    {"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]}
)

for casa, precio in zip(casas_nuevas.to_dict("records"), modelo.predict(casas_nuevas)):
    print(f"{casa} -> {precio:.1f} miles de US$")
