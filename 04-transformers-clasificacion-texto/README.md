---
language:
  - es
library_name: transformers
pipeline_tag: text-classification
tags:
  - machine-learning
  - text-classification
  - bert
  - educational-example
license: mit
---

# Clasificador de comentarios con Transformers: ¿positivo o negativo?

## Descripción del modelo

Este modelo es un ejemplo pedagógico del Seminario de Actualización (IFTS N.º 18). Muestra **todas las piezas** que se publican cuando se comparte un modelo de Hugging Face Transformers: la arquitectura, los pesos y el tokenizer.

Clasifica comentarios cortos sobre departamentos como `positivo` o `negativo`.

Es un BERT diminuto **entrenado desde cero** con un vocabulario propio y muy chico. En un caso real no se entrena desde cero: se hace *fine-tuning* de un modelo preentrenado del Hub. Igual, los archivos que se publican son los mismos.

## Objetivo

Clasificar un texto en una de dos clases. Es un problema de clasificación binaria, como el de la Clase 11, pero con texto como entrada.

## Datos de entrenamiento

- **Dataset:** sintético, generado en `train.py` a partir de 5 plantillas de frases y 16 adjetivos (8 positivos y 8 negativos).
- **Cantidad de registros:** 368 comentarios. Se usan 294 para entrenar y 74 para evaluar.
- **Entrada (X):** el texto del comentario, por ejemplo "el departamento es muy luminoso".
- **Salida (y):** la etiqueta, `negativo` (0) o `positivo` (1).

## Entrenamiento

- **Tipo de aprendizaje:** supervisado.
- **Problema:** clasificación de texto (binaria).
- **Arquitectura:** `BertForSequenceClassification` con 2 capas, 2 cabezas de atención, dimensión 32 y unos 20.500 parámetros.
- **Tokenizer:** a nivel palabra (`WordLevel`), con 37 tokens. Pasa todo a minúsculas y saca las tildes.
- **Función de pérdida:** entropía cruzada. La calcula el modelo cuando recibe `labels`.
- **Optimizador:** `AdamW`, con tasa de aprendizaje 0,001.
- **Épocas:** 15, en lotes de 16.
- **Biblioteca:** Hugging Face Transformers, con PyTorch.

## Evaluación

Se evaluó sobre 74 comentarios que el modelo no vio durante el entrenamiento.

| Métrica | Resultado |
|---|---|
| Accuracy | 1,000 |

Que acierte el 100 % **no quiere decir que el modelo sea bueno**. Los comentarios de test usan las mismas plantillas y los mismos adjetivos que los de entrenamiento: es un examen con las mismas preguntas que la práctica. Ver *Limitaciones*.

## Archivos

| Archivo | Qué es |
|---|---|
| `config.json` | la arquitectura: capas, dimensiones, etiquetas (`id2label`) |
| `model.safetensors` | los pesos |
| `tokenizer.json` | el vocabulario y cómo se corta el texto en tokens |
| `tokenizer_config.json` | la configuración del tokenizer y sus tokens especiales |
| `train.py` | el código que lo entrenó |
| `predict.py` | cómo cargarlo y usarlo |
| `requirements.txt` | las dependencias |
| `README.md` | esta Model Card |

**Modelo y tokenizer van juntos.** Si el modelo se publica sin su tokenizer, no se puede usar: los números que entiende el modelo son posiciones del vocabulario de ese tokenizer y de ningún otro.

## Cómo utilizar el modelo

Desde esta carpeta:

```python
from transformers import pipeline

clasificador = pipeline("text-classification", model=".")
print(clasificador("el departamento es muy luminoso"))
# [{'label': 'positivo', 'score': 0.99...}]
```

Una vez publicado en el Hub, se cambia la carpeta por el nombre del repo:

```python
clasificador = pipeline("text-classification", model="tu-usuario/nombre-del-repo")
```

Funciona con Transformers 4 (la versión de Colab) y con Transformers 5.

## Limitaciones

- Tiene fines exclusivamente educativos.
- **Solo conoce 37 palabras.** Cualquier otra palabra se convierte en `[UNK]` (desconocida). Por ejemplo, "tiene una vista increíble" queda como `[UNK] [UNK] [UNK] [UNK]`. **El modelo igual responde, y con seguridad alta:** dice `negativo (1.00)`. Un modelo puede equivocarse con total confianza cuando le llega algo distinto de lo que vio al entrenar.
- No entiende negaciones ("no es luminoso") ni ironía. Las frases de entrenamiento no las tienen.
- El dataset sale de plantillas, así que el accuracy de test es optimista.
- Para un uso real, conviene partir de un modelo preentrenado en español y hacerle fine-tuning con comentarios reales.

## Autoría

Modelo desarrollado con fines educativos para el IFTS N.º 18. Docente: Cynthia M. Villagra.
