"""Carga model.onnx y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
casas_nuevas = pd.DataFrame({"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]})

# Se carga con onnxruntime: no hace falta scikit-learn
sesion = ort.InferenceSession(CARPETA / "model.onnx")
entrada = sesion.get_inputs()[0].name
precios = sesion.run(None, {entrada: casas_nuevas.to_numpy(dtype="float32")})[0]

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
