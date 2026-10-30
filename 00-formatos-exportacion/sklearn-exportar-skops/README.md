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
  - skops
license: mit
---

# scikit-learn: exportar en formato skops

Exporta el modelo entrenado en `sklearn-entrenamiento-evaluacion` al formato **skops** (`model.skops`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../sklearn-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo. Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-sklearn-exportar-skops
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | el objeto entero, sin usar pickle |
| **¿Cómo se carga?** | `skops.io.load`, con scikit-learn instalado. Hay que aceptar los tipos que no conoce. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Más seguro que pickle: no crea objetos de tipos que no aceptaste. |
| **¿Para qué caso?** | compartir un modelo de scikit-learn en Hugging Face Hub |
