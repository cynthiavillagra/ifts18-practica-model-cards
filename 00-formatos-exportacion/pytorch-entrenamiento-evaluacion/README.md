---
language:
  - es
library_name: pytorch
pipeline_tag: tabular-regression
tags:
  - machine-learning
  - regression
  - linear-regression
  - educational-example
license: mit
---

# PyTorch: entrenamiento y evaluación

Entrena una regresión lineal con `casas.csv`, la evalúa con datos que no vio y la guarda en `model.safetensors` + `config.json` (con `modelo.py`).

## Cómo correrlo

```bash
python train.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/pytorch-entrenamiento-evaluacion`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-pytorch-entrenamiento-evaluacion
```

## Datos

`casas.csv` es el dataset sintético de la Clase 10: 200 casas con `m2`, `balcon`, `ascensor` y `precio` (miles de US$). Se usan 160 para entrenar y 40 para evaluar.

## Formatos de exportación

Estas carpetas parten del modelo que guarda este `train.py`:

- `pytorch-exportar-completo`: modelo completo
- `pytorch-exportar-torchscript`: TorchScript
- `pytorch-exportar-torch-export`: torch.export
