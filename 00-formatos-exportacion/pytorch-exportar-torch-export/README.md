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
  - torch-export
license: mit
---

# PyTorch: exportar en formato torch.export

Exporta el modelo entrenado en `pytorch-entrenamiento-evaluacion` al formato **torch.export** (`model.pt2`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../pytorch-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/pytorch-exportar-torch-export`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-pytorch-exportar-torch-export
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | el grafo de operaciones y los pesos |
| **¿Cómo se carga?** | `torch.export.load(...).module()`, sin la clase. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Sí, no usa pickle. |
| **¿Para qué caso?** | llevar el modelo a producción o a otros dispositivos |
