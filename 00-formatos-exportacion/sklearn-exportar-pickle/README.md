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
  - pickle
license: mit
---

# scikit-learn: exportar en formato pickle

Exporta el modelo entrenado en `sklearn-entrenamiento-evaluacion` al formato **pickle** (`model.pkl`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../sklearn-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo. Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-sklearn-exportar-pickle
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | el objeto entero de Python (pesos, bias y nombres de columnas) |
| **¿Cómo se carga?** | `pickle.load`, con scikit-learn instalado. No hace falta reescribir nada. |
| **¿Es seguro abrirlo si viene de un desconocido?** | **No.** Al cargar un pickle se puede ejecutar cualquier código. |
| **¿Para qué caso?** | seguir usándolo en Python con la misma versión de scikit-learn, entre personas de confianza |
