# Formatos de exportación

Ejemplos de los formatos de la Actividad de la Clase 10, con el mismo problema: predecir el precio de una casa a partir de `m2`, `balcon` y `ascensor`.

Por cada biblioteca hay una carpeta de **entrenamiento y evaluación**, con el dataset (`casas.csv`), y una carpeta por **formato de exportación**. Cada carpeta de exportación tiene `exportar.py`, que toma el modelo ya entrenado y lo guarda en ese formato, y `cargar.py`, que lo vuelve a cargar y predice.

| Biblioteca | Entrenamiento y evaluación | Formatos de exportación |
|---|---|---|
| scikit-learn | [`sklearn-entrenamiento-evaluacion`](sklearn-entrenamiento-evaluacion) → `model.joblib` | [pickle](sklearn-exportar-pickle) · [skops](sklearn-exportar-skops) · [ONNX](sklearn-exportar-onnx) |
| PyTorch | [`pytorch-entrenamiento-evaluacion`](pytorch-entrenamiento-evaluacion) → `model.safetensors` + `config.json` | [modelo completo](pytorch-exportar-completo) · [TorchScript](pytorch-exportar-torchscript) · [torch.export](pytorch-exportar-torch-export) |
| Keras | [`keras-entrenamiento-evaluacion`](keras-entrenamiento-evaluacion) → `model.keras` | [.weights.h5](keras-exportar-weights-h5) · [SavedModel y TFLite](keras-exportar-savedmodel-tflite) |

## Cómo correrlo

Primero el entrenamiento, después cualquier formato. Por ejemplo, para ONNX:

```bash
cd sklearn-entrenamiento-evaluacion
python train.py
cd ../sklearn-exportar-onnx
pip install -r requirements.txt
python exportar.py
python cargar.py
```

Todos los formatos de una misma biblioteca predicen el mismo precio para las tres casas de ejemplo.

## Subirlas al Hub

Cada carpeta se puede subir tal cual como repositorio de modelo: tiene todo lo que hace falta para cargar el modelo, su Model Card y un `predict_hub.py` que lo baja del Hub y predice. Se sube desde la web, como se explica en el README principal. Después, desde la raíz del repo:

```bash
python 00-formatos-exportacion/sklearn-exportar-onnx/predict_hub.py tu-usuario/casas-sklearn-exportar-onnx
```

En estos ejemplos Keras usa el backend `tensorflow` (en `03-keras-regresion` usa `torch`), porque SavedModel y TFLite son formatos de TensorFlow.

## Comparación

| Formato | ¿Qué guarda? | ¿Hace falta reescribir el modelo para cargarlo? | ¿Seguro si viene de un desconocido? |
|---|---|---|---|
| pickle | el objeto entero de Python | no | **no**: puede ejecutar código |
| skops | el objeto entero, sin pickle | no | más seguro: pide aceptar los tipos |
| ONNX | grafo de operaciones + pesos | no, y no necesita scikit-learn | sí |
| modelo completo (PyTorch) | el objeto entero, con pickle | no, pero la clase tiene que poder importarse | **no**: usa pickle |
| TorchScript | pesos + código del `forward` | no | sí (está *deprecated*) |
| torch.export | grafo de operaciones + pesos | no | sí |
| .weights.h5 | solo los pesos | **sí**: hay que armar la misma arquitectura | sí |
| SavedModel | grafo de TensorFlow + pesos | no, y no necesita Keras | sí |
| TFLite | grafo optimizado y chico | no | sí |

Documentación oficial:

- scikit-learn: https://scikit-learn.org/stable/model_persistence.html
- PyTorch: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Keras: https://keras.io/guides/serialization_and_saving/
- ONNX: https://onnx.ai/onnx/intro/
- Hugging Face, sobre la seguridad de pickle: https://huggingface.co/docs/hub/security-pickle
