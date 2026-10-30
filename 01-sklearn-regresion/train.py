"""Entrena una regresión lineal con scikit-learn y la exporta a model.joblib.

Es el mismo problema de la Clase 10: predecir el precio de una casa (en miles de US$)
a partir de sus m², si tiene balcón y si tiene ascensor.

Uso:
    python train.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]


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
X = casas[ENTRADAS]
y = casas["precio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Algoritmo + datos de entrenamiento -> modelo (los pesos y el bias)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 3. Evaluación con datos que el modelo no vio
predicciones = modelo.predict(X_test)
print(f"MAE en test: {mean_absolute_error(y_test, predicciones):.2f} miles de US$")
print(f"R² en test:  {r2_score(y_test, predicciones):.3f}")
print("Pesos (coef_):", dict(zip(ENTRADAS, modelo.coef_.round(3).tolist())))
print(f"Bias (intercept_): {modelo.intercept_:.3f}")

# 4. Exportación: se guarda el modelo entrenado, no el algoritmo
joblib.dump(modelo, CARPETA / "model.joblib")
print("Modelo guardado en model.joblib")
