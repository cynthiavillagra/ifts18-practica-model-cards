"""Carga el modelo y el tokenizer de esta carpeta y clasifica comentarios nuevos.

Uso:
    python predict.py

Cuando el modelo esté publicado en Hugging Face Hub, se carga igual cambiando la carpeta
por el nombre del repo:
    pipeline("text-classification", model="tu-usuario/nombre-del-repo")
"""

from pathlib import Path

from transformers import pipeline

CARPETA = Path(__file__).parent

clasificador = pipeline("text-classification", model=str(CARPETA))

comentarios = [
    "el departamento es muy luminoso",
    "me pareció un lugar ruidoso",
    "es amplio pero sobre todo tranquilo",
    "la verdad es húmedo y oscuro",
    "tiene una vista increíble",  # palabras que el tokenizer nunca vio: van a [UNK]
]

for comentario, resultado in zip(comentarios, clasificador(comentarios)):
    print(f"{comentario!r} -> {resultado['label']} ({resultado['score']:.2f})")

# Para ver cómo corta el tokenizer un texto:
tokens = clasificador.tokenizer.tokenize("tiene una vista increíble")
print("Tokens de 'tiene una vista increíble':", tokens)
