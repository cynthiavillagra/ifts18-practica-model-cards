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

# Keras: entrenamiento y evaluación

Entrena una regresión lineal con `casas.csv`, la evalúa con datos que no vio y la guarda en `model.keras`.

## Cómo correrlo

```bash
python train.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo. Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-keras-entrenamiento-evaluacion
```

## Datos

`casas.csv` es el dataset sintético de la Clase 10: 200 casas con `m2`, `balcon`, `ascensor` y `precio` (miles de US$). Se usan 160 para entrenar y 40 para evaluar.

## Formatos de exportación

Estas carpetas parten del modelo que guarda este `train.py`:

- `keras-exportar-weights-h5`: .weights.h5
- `keras-exportar-savedmodel-tflite`: SavedModel y TFLite
