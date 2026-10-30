---
language:
  - es
library_name: onnx
pipeline_tag: tabular-regression
tags:
  - machine-learning
  - regression
  - linear-regression
  - educational-example
  - onnx
license: mit
---

# scikit-learn: exportar en formato ONNX

Exporta el modelo entrenado en `sklearn-entrenamiento-evaluacion` al formato **ONNX** (`model.onnx`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../sklearn-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/sklearn-exportar-onnx`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-sklearn-exportar-onnx
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | un grafo de operaciones (multiplicar, sumar) con los pesos adentro |
| **¿Cómo se carga?** | `onnxruntime`, **sin** scikit-learn. También desde otros lenguajes. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Sí: el archivo no ejecuta código de Python. |
| **¿Para qué caso?** | una app que no usa Python, o predecir más rápido |
