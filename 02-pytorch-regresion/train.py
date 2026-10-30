"""Entrena una regresión lineal con PyTorch y la exporta a model.safetensors + config.json.

Mismo problema que el ejemplo 01: precio de una casa a partir de m², balcón y ascensor.

Diferencia importante con scikit-learn: PyTorch guarda solo los pesos (state_dict).
La clase del modelo y la normalización de las entradas no viajan en el archivo de pesos,
así que la normalización se guarda aparte, en config.json.

Uso:
    python train.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from modelo import RegresionLineal

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
torch.manual_seed(42)


def generar_casas(n=200, semilla=42):
    """Dataset sintético de la Clase 10 (misma semilla, mismos datos)."""
    rng = np.random.default_rng(semilla)
    m2 = rng.integers(35, 121, n)
    balcon = rng.integers(0, 2, n)
    ascensor = rng.integers(0, 2, n)
    precio = 1.5 * m2 + 8 * balcon + 12 * ascensor + 20 + rng.normal(0, 4, n)
    return pd.DataFrame({"m2": m2, "balcon": balcon, "ascensor": ascensor, "precio": precio.round(1)})


# 1. Datos
casas = generar_casas()
X = casas[ENTRADAS].to_numpy(dtype="float32")
y = casas["precio"].to_numpy(dtype="float32").reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización: se calcula SOLO con los datos de entrenamiento y se guarda para predecir
media = X_train.mean(axis=0)
desvio = X_train.std(axis=0)
X_train_t = torch.tensor((X_train - media) / desvio)
y_train_t = torch.tensor(y_train)

# 2. Modelo, función de pérdida y optimizador
modelo = RegresionLineal(n_entradas=len(ENTRADAS))
perdida_fn = torch.nn.MSELoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.1)

# 3. El ciclo de entrenamiento
for epoca in range(500):
    prediccion = modelo(X_train_t)             # el modelo predice
    perdida = perdida_fn(prediccion, y_train_t)  # la pérdida mide el error
    optimizador.zero_grad()
    perdida.backward()                         # se calculan los gradientes
    optimizador.step()                         # el optimizador corrige pesos y bias
    if epoca % 100 == 0:
        print(f"época {epoca:3d} · pérdida (MSE) {perdida.item():.2f}")

# 4. Evaluación con datos que el modelo no vio
modelo.eval()
with torch.no_grad():
    pred_test = modelo(torch.tensor((X_test - media) / desvio)).numpy()
print(f"MAE en test: {mean_absolute_error(y_test, pred_test):.2f} miles de US$")
print(f"R² en test:  {r2_score(y_test, pred_test):.3f}")

# 5. Exportación: pesos en safetensors, todo lo demás en config.json
save_file(modelo.state_dict(), CARPETA / "model.safetensors")
config = {
    "arquitectura": "RegresionLineal (torch.nn.Linear)",
    "entradas": ENTRADAS,
    "salida": "precio (miles de US$)",
    "normalizacion": {"media": media.tolist(), "desvio": desvio.tolist()},
}
(CARPETA / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("Modelo guardado en model.safetensors + config.json")
