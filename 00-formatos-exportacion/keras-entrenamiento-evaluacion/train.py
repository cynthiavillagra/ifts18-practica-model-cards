"""Entrena una regresión lineal con Keras, la evalúa y la guarda en model.keras.

Las carpetas keras-exportar-* parten de este model.keras. Se usa el backend tensorflow
porque SavedModel y TFLite son formatos de TensorFlow.

Uso:
    python train.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # SavedModel y TFLite son formatos de TensorFlow

from pathlib import Path

import keras
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

CARPETA = Path(__file__).parent
ENTRADAS = ["m2", "balcon", "ascensor"]
keras.utils.set_random_seed(42)

# 1. Datos
casas = pd.read_csv(CARPETA / "casas.csv")
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

# 5. Guardado: arquitectura + pesos + normalización, todo en un archivo
modelo.save(CARPETA / "model.keras")
print("Modelo guardado en model.keras")
