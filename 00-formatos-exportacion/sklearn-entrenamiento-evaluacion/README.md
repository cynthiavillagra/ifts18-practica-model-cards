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

# scikit-learn: entrenamiento y evaluación

Entrena una regresión lineal con `casas.csv`, la evalúa con datos que no vio y la guarda en `model.joblib`.

## Cómo correrlo

```bash
python train.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo. Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-sklearn-entrenamiento-evaluacion
```

## Datos

`casas.csv` es el dataset sintético de la Clase 10: 200 casas con `m2`, `balcon`, `ascensor` y `precio` (miles de US$). Se usan 160 para entrenar y 40 para evaluar.

## Formatos de exportación

Estas carpetas parten del modelo que guarda este `train.py`:

- `sklearn-exportar-pickle`: pickle
- `sklearn-exportar-skops`: skops
- `sklearn-exportar-onnx`: ONNX
