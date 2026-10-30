---
language:
  - es
library_name: sklearn
pipeline_tag: tabular-regression
tags:
  - machine-learning
  - regression
  - linear-regression
  - educational-example
license: mit
---

# Regresión lineal con scikit-learn: precio de una casa

## Descripción del modelo

Este modelo es un ejemplo pedagógico. Muestra cómo funciona el aprendizaje automático supervisado y qué se publica cuando se comparte un modelo entrenado.

Predice el precio de una casa, en miles de US$, a partir de su superficie y de si tiene balcón y ascensor.

## Objetivo

Predecir una variable numérica continua: el precio estimado de una casa.

## Datos de entrenamiento

- **Dataset:** sintético, generado en `train.py`. Es el mismo de la Clase 10, con la misma semilla, y no son casas reales.
- **Cantidad de registros:** 200. Se usan 160 para entrenar y 40 para evaluar.
- **Variables de entrada (X):** `m2` (de 35 a 120), `balcon` (0 o 1), `ascensor` (0 o 1).
- **Variable objetivo (y):** `precio`, en miles de US$.
- **Cómo se generaron:** `precio = 1.5·m2 + 8·balcon + 12·ascensor + 20 + ruido`.

## Entrenamiento

- **Tipo de aprendizaje:** supervisado.
- **Problema:** regresión.
- **Algoritmo:** regresión lineal (`sklearn.linear_model.LinearRegression`), por mínimos cuadrados.
- **Biblioteca:** scikit-learn.

**Parámetros aprendidos.** Son los valores que guarda `model.joblib`:

| Parámetro | Valor | Se lee como… |
|---|---|---|
| peso de `m2` | 1,503 | cada m² suma unos US$ 1.500 |
| peso de `balcon` | 8,719 | el balcón suma unos US$ 8.700 |
| peso de `ascensor` | 11,988 | el ascensor suma unos US$ 12.000 |
| bias (`intercept_`) | 19,557 | precio base |

Los pesos aprendidos quedan muy cerca de los que se usaron para generar los datos.

## Evaluación

Se evaluó sobre 40 casas que el modelo no vio durante el entrenamiento.

| Métrica | Resultado |
|---|---|
| MAE | 3,08 miles de US$ |
| R² | 0,989 |

## Archivos

| Archivo | Qué es |
|---|---|
| `model.joblib` | el modelo entrenado, con sus pesos y su bias |
| `train.py` | el código que lo entrenó, para reproducirlo |
| `predict.py` | cómo cargarlo y usarlo |
| `requirements.txt` | las dependencias |
| `README.md` | esta Model Card |

## Cómo utilizar el modelo

```python
import joblib
import pandas as pd

modelo = joblib.load("model.joblib")

casa = pd.DataFrame({"m2": [80], "balcon": [1], "ascensor": [0]})
print(modelo.predict(casa))  # [148.5] miles de US$
```

Para descargarlo desde el Hub:

```python
from huggingface_hub import hf_hub_download
import joblib

ruta = hf_hub_download(repo_id="tu-usuario/nombre-del-repo", filename="model.joblib")
modelo = joblib.load(ruta)
```

## Limitaciones

- Tiene fines exclusivamente educativos. Los datos son sintéticos y no representan precios reales.
- Solo conoce casas de 35 a 120 m². Fuera de ese rango, extrapola una recta y nada garantiza que acierte.
- Supone que cada característica suma un monto fijo. No capta interacciones, como que el ascensor valga más en un edificio alto.
- `joblib` usa pickle: **solo carguen archivos `.joblib` de fuentes en las que confíen**, porque al cargarlos se puede ejecutar código.
- La versión de scikit-learn con la que se carga debería ser la misma con la que se entrenó.

## Autoría

Modelo desarrollado con fines educativos para el IFTS N.º 18. Docente: Cynthia M. Villagra.
