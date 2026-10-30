# Práctica: publicar un modelo con su Model Card

Seminario de Actualización · Tec. Sup. en Ciencia de Datos e IA · IFTS N.º 18
Docente: Cynthia M. Villagra · 2026

Este repo tiene **cuatro ejemplos** de un modelo entrenado, listos para publicar en Hugging Face Hub. Hay uno por cada biblioteca que usamos en la cursada. Cada carpeta es un repositorio de modelo completo: tiene el modelo exportado, el código para reproducirlo y su Model Card.

| Carpeta | Biblioteca | Problema | Qué se publica |
|---|---|---|---|
| [`01-sklearn-regresion`](01-sklearn-regresion) | scikit-learn | precio de una casa (regresión) | `model.joblib` |
| [`02-pytorch-regresion`](02-pytorch-regresion) | PyTorch | precio de una casa (regresión) | `model.safetensors` + `config.json` + `modelo.py` |
| [`03-keras-regresion`](03-keras-regresion) | Keras | precio de una casa (regresión) | `model.keras` |
| [`04-transformers-clasificacion-texto`](04-transformers-clasificacion-texto) | Transformers | comentario positivo o negativo (clasificación) | `model.safetensors` + `config.json` + archivos del tokenizer |

Los tres primeros resuelven **el mismo problema con los mismos datos de la Clase 10**. Así se puede comparar qué guarda cada biblioteca al exportar.

## Tres productos distintos

En cada carpeta hay tres cosas que conviene no confundir:

1. **El código de entrenamiento** (`train.py`). Es el procedimiento: toma los datos y un algoritmo, y aprende los parámetros.
2. **El modelo entrenado** (`model.joblib`, `model.safetensors`, `model.keras`). Es el resultado del aprendizaje: los parámetros ya ajustados, que permiten predecir.
3. **La Model Card** (`README.md`). Es la documentación: qué es el modelo, cómo se obtuvo, cómo se usa y cuáles son sus limitaciones.

Para que otra persona **use** el modelo, no necesita volver a entrenarlo. Necesita el modelo entrenado y la forma de cargarlo, que está en `predict.py`. El `train.py` se publica para que el resultado se pueda **reproducir**.

## Qué cambia entre bibliotecas al exportar

| | ¿Qué guarda el archivo? | ¿Qué hay que guardar aparte? |
|---|---|---|
| **scikit-learn** | el objeto entero: pesos, bias y nombres de las columnas | nada |
| **PyTorch** | **solo los pesos** (`state_dict`) | la clase del modelo (`modelo.py`) y la normalización de las entradas (`config.json`) |
| **Keras** | arquitectura, pesos y la capa de normalización | nada |
| **Transformers** | `save_pretrained` guarda la arquitectura (`config.json`) y los pesos | el **tokenizer**, que se guarda con su propio `save_pretrained` |

## Cómo correr un ejemplo

En cada carpeta, con el entorno virtual activado:

```bash
cd 01-sklearn-regresion
python -m venv .venv
# Windows: .venv\Scripts\activate     ·     Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python predict.py     # usa el modelo que ya está entrenado
python train.py       # opcional: lo vuelve a entrenar y pisa el archivo del modelo
```

**Atención.** Primero se activa el entorno y después se instala el `requirements.txt`. Lo aprendimos entre todos en la Clase 9.

En Google Colab no hace falta el entorno virtual: alcanza con `!pip install -r requirements.txt`.

## Cómo publicarlo en Hugging Face Hub

Cada **carpeta** se publica como un repositorio de modelo distinto. Lo que se sube es el contenido de la carpeta, no el repo entero.

### Opción A: desde la web

1. Entrar a <https://huggingface.co/new>.
2. Elegir el nombre del repositorio, por ejemplo `regresion-casas-sklearn`, y si va a ser público o privado.
3. En **Files and versions → Add file → Upload files**, arrastrar **todos** los archivos de la carpeta, incluido el `README.md`.
4. Hacer el commit. La pestaña **Model card** muestra el `README.md`, y los metadatos del encabezado YAML aparecen como etiquetas.

### Opción B: desde Python

```python
from huggingface_hub import HfApi

api = HfApi()  # antes: `hf auth login` en la terminal, con un token de escritura
api.create_repo("tu-usuario/regresion-casas-sklearn", exist_ok=True)
api.upload_folder(folder_path="01-sklearn-regresion", repo_id="tu-usuario/regresion-casas-sklearn")
```

**El token de Hugging Face no se escribe nunca en el código ni se sube a GitHub.** Se ingresa con `hf auth login` o se lee de una variable de entorno.

## La Model Card

La Model Card es el `README.md` del repositorio del modelo. Empieza con un bloque de **metadatos en YAML**, entre `---`, que el Hub usa para clasificar y buscar el modelo:

```yaml
---
language:
  - es
library_name: sklearn        # sklearn · pytorch · keras · transformers
pipeline_tag: tabular-regression
tags:
  - educational-example
license: mit
---
```

Después vienen estas secciones, que tienen todos los ejemplos:

| Sección | Qué documenta |
|---|---|
| Descripción del modelo | qué hace |
| Objetivo | para qué se creó |
| Datos de entrenamiento | qué dataset se usó, cuántos registros, qué entra (X) y qué sale (y) |
| Entrenamiento | tipo de aprendizaje, algoritmo o arquitectura, biblioteca |
| Evaluación | métrica y resultado, sobre datos que el modelo no vio |
| Cómo utilizar el modelo | código para cargarlo y predecir |
| Limitaciones | qué no puede hacer o dónde falla |
| Autoría | quién lo hizo y con qué fin |

Los valores de cada Model Card (pesos, métricas) son los que dio el entrenamiento real de ese modelo. Si se vuelve a correr `train.py` con otros datos, hay que actualizarlos.

## Documentación oficial

- Model Cards: <https://huggingface.co/docs/hub/model-cards>
- Subir modelos al Hub: <https://huggingface.co/docs/hub/models-uploading>
- Persistencia de modelos en scikit-learn: <https://scikit-learn.org/stable/model_persistence.html>
- Guardar y cargar modelos en PyTorch: <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
- Guardar y cargar modelos en Keras: <https://keras.io/guides/serialization_and_saving/>
- `save_pretrained` en Transformers: <https://huggingface.co/docs/transformers/main_classes/model>
