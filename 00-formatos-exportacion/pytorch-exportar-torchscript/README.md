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
  - torchscript
license: mit
---

# PyTorch: exportar en formato TorchScript

Exporta el modelo entrenado en `pytorch-entrenamiento-evaluacion` al formato **TorchScript** (`model_ts.pt`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../pytorch-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo (desde la raíz: `python publicar_en_hub.py 00-formatos-exportacion/pytorch-exportar-torchscript`). Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-pytorch-exportar-torchscript
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | los pesos y el código del `forward` |
| **¿Cómo se carga?** | `torch.jit.load`, sin la clase. |
| **¿Es seguro abrirlo si viene de un desconocido?** | Sí, no usa pickle. Está *deprecated*: PyTorch recomienda torch.export. |
| **¿Para qué caso?** | proyectos viejos o C++. Para algo nuevo, torch.export |
