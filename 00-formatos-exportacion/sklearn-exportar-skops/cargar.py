"""Carga model.skops y predice el precio de casas nuevas. No vuelve a entrenar.

Uso:
    python cargar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
casas_nuevas = pd.DataFrame({"m2": [50, 80, 110], "balcon": [0, 1, 1], "ascensor": [0, 0, 1]})

# skops no carga tipos desconocidos sin permiso: primero se revisan
desconocidos = sio.get_untrusted_types(file=CARPETA / "model.skops")
print("Tipos a revisar antes de cargar:", desconocidos)
modelo = sio.load(CARPETA / "model.skops", trusted=desconocidos)
precios = modelo.predict(casas_nuevas)

for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
