"""Entrena una regresión lineal con scikit-learn, la evalúa y la guarda en model.joblib.

Es el mismo problema de la Clase 10: predecir el precio de una casa (en miles de US$)
a partir de sus m², si tiene balcón y si tiene ascensor. Las carpetas sklearn-exportar-*
parten de este model.joblib.

Uso:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]

# 1. Datos
casas = pd.read_csv(CARPETA / "casas.csv")
X = casas[ENTRADAS]
y = casas["precio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entrenamiento
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 3. Evaluación con datos que el modelo no vio
predicciones = modelo.predict(X_test)
print(f"MAE en test: {mean_absolute_error(y_test, predicciones):.2f} miles de US$")
print(f"R² en test:  {r2_score(y_test, predicciones):.3f}")
print("Pesos (coef_):", dict(zip(ENTRADAS, modelo.coef_.round(3).tolist())))
print(f"Bias (intercept_): {modelo.intercept_:.3f}")

# 4. Guardado en el formato habitual de scikit-learn
joblib.dump(modelo, CARPETA / "model.joblib")
print("Modelo guardado en model.joblib")
