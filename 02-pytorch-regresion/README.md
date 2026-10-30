---
language:
  - es
library_name: pytorch
pipeline_tag: tabular-regression
tags:
  - machine-learning
  - regression
  - linear-regression
  - safetensors
  - educational-example
license: mit
---

# Regresión lineal con PyTorch: precio de una casa

## Descripción del modelo

Este modelo es un ejemplo pedagógico del Seminario de Actualización (IFTS N.º 18). Muestra cómo se entrena una regresión lineal con el ciclo explícito de PyTorch y cómo se publican los pesos en formato **safetensors**.

Predice el precio de una casa, en miles de US$, a partir de su superficie y de si tiene balcón y ascensor. Resuelve el mismo problema que el ejemplo de scikit-learn.

## Objetivo

Predecir una variable numérica continua: el precio estimado de una casa.

## Datos de entrenamiento

- **Dataset:** sintético, generado en `train.py`. Es el mismo de la Clase 10, con la misma semilla, y no son casas reales.
- **Cantidad de registros:** 200. Se usan 160 para entrenar y 40 para evaluar.
- **Variables de entrada (X):** `m2` (de 35 a 120), `balcon` (0 o 1), `ascensor` (0 o 1).
- **Variable objetivo (y):** `precio`, en miles de US$.
- **Preprocesamiento:** las entradas se normalizan restando la media y dividiendo por el desvío, ambos calculados con los datos de entrenamiento. Esos valores están en `config.json`.

## Entrenamiento

- **Tipo de aprendizaje:** supervisado.
- **Problema:** regresión.
- **Arquitectura:** una capa `torch.nn.Linear(3, 1)`, es decir, tres pesos y un bias. Está definida en `modelo.py`.
- **Función de pérdida:** error cuadrático medio (`MSELoss`).
- **Optimizador:** descenso por gradiente (`SGD`), con tasa de aprendizaje 0,1.
- **Épocas:** 500. La pérdida se estabiliza antes de la época 100.
- **Biblioteca:** PyTorch.

**Parámetros aprendidos.** Son los que guarda `model.safetensors`:

| Parámetro | Valor |
|---|---|
| `lineal.weight` | [35,157 · 4,351 · 5,992] |
| `lineal.bias` | 147,769 |

**Por qué los pesos no se parecen a los de scikit-learn.** Están en la escala de las entradas **normalizadas**. Si se divide cada peso por el desvío de su variable, se vuelve a la escala original. Por ejemplo, 35,157 / 23,398 ≈ 1,503 para `m2`, que es el mismo peso que encontró scikit-learn.

## Evaluación

Se evaluó sobre 40 casas que el modelo no vio durante el entrenamiento.

| Métrica | Resultado |
|---|---|
| MAE | 3,08 miles de US$ |
| R² | 0,989 |

## Archivos

| Archivo | Qué es |
|---|---|
| `model.safetensors` | **solo los pesos** (`state_dict`) |
| `config.json` | las entradas, la salida y la normalización que hay que aplicar antes de predecir |
| `modelo.py` | la clase del modelo: PyTorch no la guarda en el archivo de pesos |
| `train.py` | el código que lo entrenó |
| `predict.py` | cómo cargarlo y usarlo |
| `requirements.txt` | las dependencias |
| `README.md` | esta Model Card |

## Cómo utilizar el modelo

```python
import json
import numpy as np
import torch
from safetensors.torch import load_file
from modelo import RegresionLineal

config = json.load(open("config.json"))
media = np.array(config["normalizacion"]["media"], dtype="float32")
desvio = np.array(config["normalizacion"]["desvio"], dtype="float32")

modelo = RegresionLineal(n_entradas=3)              # 1. el modelo vacío
modelo.load_state_dict(load_file("model.safetensors"))  # 2. se le cargan los pesos
modelo.eval()

casa = np.array([[80, 1, 0]], dtype="float32")
with torch.no_grad():
    print(modelo(torch.tensor((casa - media) / desvio)))  # tensor([[148.5]])
```

## Limitaciones

- Tiene fines exclusivamente educativos. Los datos son sintéticos y no representan precios reales.
- Solo conoce casas de 35 a 120 m².
- **Si se olvida la normalización, las predicciones dan cualquier cosa.** Los pesos solos no alcanzan: hace falta `config.json`.
- Para cargar los pesos hace falta la misma arquitectura (`modelo.py`). Si cambia la clase, `load_state_dict` falla.
- A diferencia de pickle o joblib, **safetensors no ejecuta código al cargarse**: solo contiene números.

## Autoría

Modelo desarrollado con fines educativos para el IFTS N.º 18. Docente: Cynthia M. Villagra.
