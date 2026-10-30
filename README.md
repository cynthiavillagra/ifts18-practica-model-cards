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

## Cómo preparar el entorno y correr un ejemplo

Hay dos formas de hacerlo.

**Un entorno por carpeta.** Cada ejemplo queda con sus propias dependencias, como si fuera un proyecto aparte. En Windows: doble clic en `crear_entornos.bat`. Crea un `.venv` en cada carpeta, instala su `requirements.txt` y prueba `predict.py`. Se puede volver a correr en cualquier PC.

**Un solo entorno para los cuatro.** Desde la carpeta raíz del repositorio, creá un entorno virtual e instalá las dependencias de los cuatro ejemplos:

```bash
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate.bat
# Linux/Mac:            source .venv/bin/activate
pip install -r requirements.txt
```

Con el entorno activado, ejecutá desde la raíz el ejemplo que quieras. Por ejemplo:

```bash
python 01-sklearn-regresion/predict.py  # usa el modelo que ya está entrenado
python 01-sklearn-regresion/train.py    # opcional: lo vuelve a entrenar
python 01-sklearn-regresion/predict_hub.py tu-usuario/regresion-casas-sklearn  # usa el modelo publicado en el Hub
```

Reemplazá `01-sklearn-regresion` por la carpeta de otro ejemplo para ejecutarlo. Primero se activa el entorno y después se instala el `requirements.txt`.

En Google Colab no hace falta el entorno virtual: alcanza con `!pip install -r requirements.txt`.

## Cómo publicarlo en Hugging Face Hub

Cada **carpeta** se publica como un repositorio de modelo distinto. Lo que se sube es el contenido de la carpeta, no el repo entero.

### Opción A: desde la web

1. Entrar a <https://huggingface.co/new>.
2. Elegir el nombre del repositorio, por ejemplo `regresion-casas-sklearn`, y si va a ser público o privado.
3. En **Files and versions → Add file → Upload files**, arrastrar **todos** los archivos de la carpeta, incluido el `README.md`.
4. Hacer el commit. La pestaña **Model card** muestra el `README.md`, y los metadatos del encabezado YAML aparecen como etiquetas.

### Opción B: con el script `publicar_en_hub.py`

Primero se inicia sesión una vez con un token de **escritura** (<https://huggingface.co/settings/tokens>). Después se corre el script desde la raíz del repo:

```bash
hf auth login
python publicar_en_hub.py                        # publica todas las carpetas en tu cuenta
python publicar_en_hub.py 01-sklearn-regresion   # o solo una
python publicar_en_hub.py 00-formatos-exportacion/sklearn-exportar-onnx  # o un formato
```

El script crea un repo de modelo por carpeta, por ejemplo `tu-usuario/regresion-casas-sklearn`. Las carpetas de `00-formatos-exportacion` se publican cada una aparte, por ejemplo `tu-usuario/casas-sklearn-exportar-onnx`. Cada carpeta se sube con `upload_folder`. **Ojo: `upload_folder` no lee el `.gitignore`.** Por eso el script indica en `ignore_patterns` qué no se sube: el `.venv`, `__pycache__` y `.env`. Si no lo hiciera, el entorno virtual entero terminaría en el Hub.

En la opción web pasa lo mismo: hay que arrastrar los archivos **sin** la carpeta `.venv`.

**El token de Hugging Face no se escribe nunca en el código ni se sube a GitHub.** Lo guarda `hf auth login` en tu PC.

## Inferir con el modelo publicado

El modelo queda **guardado** en el Hub, pero la inferencia **corre en tu código**: en tu PC, en Colab o en el servidor donde esté tu app (Render, Streamlit Cloud). La primera vez se descarga; después se usa la copia en caché.

Cada carpeta tiene un `predict_hub.py` que hace exactamente eso:

```bash
cd 01-sklearn-regresion
python predict_hub.py tu-usuario/regresion-casas-sklearn
```

| Biblioteca | Qué baja del Hub | Qué tiene que estar en tu código |
|---|---|---|
| scikit-learn | `model.joblib` (con `hf_hub_download`) | nada más |
| PyTorch | `model.safetensors` y `config.json` | la clase del modelo (`modelo.py`) |
| Keras | `model.keras` (con `hf_hub_download`) | nada más |
| Transformers | todo: `pipeline(model="tu-usuario/repo")` lo baja solo | nada más |

**Esto no es la API de Hugging Face.** `InferenceClient`, el que usamos en la Clase 6 con `cardiffnlp/twitter-xlm-roberta-base-sentiment`, hace correr el modelo en los servidores de Hugging Face. Eso solo funciona con los modelos que Hugging Face decide mantener cargados, en general los populares, y no con un modelo nuestro recién subido. Para que otra app consuma nuestro modelo por API, se arma una app (por ejemplo un Space de Gradio o una app en Render) que lo carga con `predict_hub.py`, y esa app se consulta con `gradio_client`.

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
