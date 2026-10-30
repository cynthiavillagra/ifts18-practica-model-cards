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
  - tensorflow
  - tflite
license: mit
---

# Keras: exportar en formato SavedModel y TFLite

Exporta el modelo entrenado en `keras-entrenamiento-evaluacion` al formato **SavedModel y TFLite** (`model_savedmodel/` y `model.tflite`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../keras-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/keras-exportar-savedmodel-tflite`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-keras-exportar-savedmodel-tflite
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | el grafo de TensorFlow con los pesos. TFLite es una versión chica y optimizada |
| **¿Cómo se carga?** | SavedModel con `tf.saved_model.load`, TFLite con `tf.lite.Interpreter`. Ninguno necesita Keras. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Sí: no ejecutan código de Python. |
| **¿Para qué caso?** | SavedModel para servir el modelo en un servidor, TFLite para un celular |
