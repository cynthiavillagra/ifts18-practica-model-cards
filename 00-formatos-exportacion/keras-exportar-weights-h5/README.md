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
  - h5
license: mit
---

# Keras: exportar en formato .weights.h5

Exporta el modelo entrenado en `keras-entrenamiento-evaluacion` al formato **.weights.h5** (`model.weights.h5`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../keras-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/keras-exportar-weights-h5`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-keras-exportar-weights-h5
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | **solo los pesos** (incluida la media y el desvío de la capa Normalization) |
| **¿Cómo se carga?** | Hay que volver a armar la misma arquitectura y después `load_weights`. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Sí: solo tiene números. |
| **¿Para qué caso?** | seguir entrenando, o guardar puntos intermedios del entrenamiento |
