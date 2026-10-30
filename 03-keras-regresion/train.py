"""Entrena una regresión lineal con Keras y la exporta a model.keras.

Mismo problema que los ejemplos 01 y 02: precio de una casa a partir de m², balcón y ascensor.

Diferencia con PyTorch: en Keras la normalización es una capa del modelo (Normalization),
así que viaja adentro de model.keras y no hace falta guardarla aparte.

Uso:
    python train.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # como en la Clase 10; también anda con tensorflow

from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
keras.utils.set_random_seed(42)


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
y = casas["precio"].to_numpy(dtype="float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Modelo: capa de normalización + una neurona lineal (pesos y bias)
normalizacion = keras.layers.Normalization()
normalizacion.adapt(X_train)  # aprende media y desvío solo de los datos de entrenamiento

modelo = keras.Sequential([
    keras.Input(shape=(len(ENTRADAS),)),
    normalizacion,
    keras.layers.Dense(1),
])

# 3. Pérdida y optimizador se definen en compile; el ciclo lo hace fit
modelo.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1), loss="mse")
modelo.fit(X_train, y_train, epochs=200, verbose=0)

# 4. Evaluación con datos que el modelo no vio
pred_test = modelo.predict(X_test, verbose=0).ravel()
print(f"MAE en test: {mean_absolute_error(y_test, pred_test):.2f} miles de US$")
print(f"R² en test:  {r2_score(y_test, pred_test):.3f}")

# 5. Exportación: arquitectura + pesos + normalización, todo en un archivo
modelo.save(CARPETA / "model.keras")
print("Modelo guardado en model.keras")
