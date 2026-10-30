"""Descarga el modelo y el tokenizer desde Hugging Face Hub y clasifica EN ESTA PC
(o en el servidor donde corra).

Con Transformers alcanza con el nombre del repo: pipeline() baja config.json,
model.safetensors y el tokenizer.

Uso:
    python predict_hub.py tu-usuario/clasificador-comentarios-transformers
"""

import sys

from transformers import pipeline

REPO_ID = sys.argv[1] if len(sys.argv) > 1 else "TU-USUARIO/clasificador-comentarios-transformers"

clasificador = pipeline("text-classification", model=REPO_ID)

comentarios = [
    "el departamento es muy luminoso",
    "me pareció un lugar ruidoso",
    "es amplio pero sobre todo tranquilo",
]
for comentario, resultado in zip(comentarios, clasificador(comentarios)):
    print(f"{comentario!r} -> {resultado['label']} ({resultado['score']:.2f})")
