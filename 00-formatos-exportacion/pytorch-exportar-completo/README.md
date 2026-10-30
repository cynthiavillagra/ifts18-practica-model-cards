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
  - pytorch-full-model
license: mit
---

# PyTorch: exportar en formato modelo completo

Exporta el modelo entrenado en `pytorch-entrenamiento-evaluacion` al formato **modelo completo** (`model_completo.pt`) y lo vuelve a cargar para predecir.

## Cómo correrlo

```bash
python ../pytorch-entrenamiento-evaluacion/train.py   # si todavía no está entrenado
python exportar.py
python cargar.py
```

## Usar el modelo desde el Hub

Esta carpeta se puede subir tal cual como repositorio de modelo. Después, desde cualquier PC:

```bash
python predict_hub.py tu-usuario/casas-pytorch-exportar-completo
```

## Qué guarda el formato

| | |
|---|---|
| **¿Qué guarda?** | el objeto entero (clase + pesos), con pickle |
| **¿Cómo se carga?** | `torch.load(..., weights_only=False)`. La clase `RegresionLineal` tiene que poder importarse. |
| **¿Es seguro abrirlo si viene de un desconocido?** | **No.** Usa pickle: por eso PyTorch obliga a poner `weights_only=False`. |
| **¿Para qué caso?** | guardar y retomar rápido en tu propia compu |
