"""Descarga el modelo desde Hugging Face Hub y predice EN ESTA PC.

Baja model.onnx. El modelo vive en el Hub; la inferencia la hace este código.

No hace falta scikit-learn: alcanza con onnxruntime.

Uso:
    python predict_hub.py tu-usuario/casas-sklearn-exportar-onnx
"""

import sys
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/casas-sklearn-exportar-onnx"

# La primera vez descarga el archivo; después usa la copia en caché
sesion = ort.InferenceSession(hf_hub_download(repo_id=REPO_ID, filename="model.onnx"))
casas_nuevas = np.array([[50, 0, 0], [80, 1, 0], [110, 1, 1]], dtype="float32")
precios = sesion.run(None, {sesion.get_inputs()[0].name: casas_nuevas})[0]

ENTRADAS = ["m2", "balcon", "ascensor"]
for casa, precio in zip([[50, 0, 0], [80, 1, 0], [110, 1, 1]], np.ravel(precios)):
    print(f"{dict(zip(ENTRADAS, casa))} -> {precio:.1f} miles de US$")
