---
language:
  - es
library_name: keras
pipeline_tag: tabular-regression
tags:
  - machine-learning
  - regression
  - linear-regression
  - educational-example
license: mit
---

# Regresión lineal con Keras: precio de una casa

## Descripción del modelo

Este modelo es un ejemplo pedagógico del Seminario de Actualización (IFTS N.º 18). Muestra cómo se entrena una regresión lineal con Keras y cómo se exporta en el formato nativo `.keras`.

Predice el precio de una casa, en miles de US$, a partir de su superficie y de si tiene balcón y ascensor. Resuelve el mismo problema que los ejemplos de scikit-learn y PyTorch.

## Objetivo

Predecir una variable numérica continua: el precio estimado de una casa.

## Datos de entrenamiento

- **Dataset:** sintético, generado en `train.py`. Es el mismo de la Clase 10, con la misma semilla, y no son casas reales.
- **Cantidad de registros:** 200. Se usan 160 para entrenar y 40 para evaluar.
- **Variables de entrada (X):** `m2` (de 35 a 120), `balcon` (0 o 1), `ascensor` (0 o 1).
- **Variable objetivo (y):** `precio`, en miles de US$.

## Entrenamiento

- **Tipo de aprendizaje:** supervisado.
- **Problema:** regresión.
- **Arquitectura:** `Sequential` con una capa `Normalization` y una capa `Dense(1)`, es decir, tres pesos y un bias.
- **Función de pérdida:** error cuadrático medio (`mse`).
- **Optimizador:** `SGD`, con tasa de aprendizaje 0,1.
- **Épocas:** 200.
- **Biblioteca:** Keras 3, con backend PyTorch. También funciona con TensorFlow.

La capa `Normalization` aprende la media y el desvío de los datos de entrenamiento con `adapt()`. Como es una capa del modelo, **se guarda adentro de `model.keras`** y no hace falta normalizar a mano al predecir. En el ejemplo de PyTorch, en cambio, la normalización se guarda aparte.

## Evaluación

Se evaluó sobre 40 casas que el modelo no vio durante el entrenamiento.

| Métrica | Resultado |
|---|---|
| MAE | 3,10 miles de US$ |
| R² | 0,989 |

## Archivos

| Archivo | Qué es |
|---|---|
| `model.keras` | arquitectura, pesos, normalización y configuración de entrenamiento, en un solo archivo |
| `train.py` | el código que lo entrenó |
| `predict.py` | cómo cargarlo y usarlo |
| `requirements.txt` | las dependencias |
| `README.md` | esta Model Card |

## Cómo utilizar el modelo

```python
import os
os.environ["KERAS_BACKEND"] = "torch"   # o "tensorflow"

import keras
import numpy as np

modelo = keras.saving.load_model("model.keras")
print(modelo.predict(np.array([[80, 1, 0]], dtype="float32")))  # [[148.6]]
```

## Limitaciones

- Tiene fines exclusivamente educativos. Los datos son sintéticos y no representan precios reales.
- Solo conoce casas de 35 a 120 m².
- Las columnas tienen que ir en este orden: `m2`, `balcon`, `ascensor`. El modelo recibe un arreglo, no una tabla con nombres de columnas.
- Necesita Keras 3. El formato `.keras` no se abre con versiones viejas de TensorFlow/Keras 2.

## Autoría

Modelo desarrollado con fines educativos para el IFTS N.º 18. Docente: Cynthia M. Villagra.
