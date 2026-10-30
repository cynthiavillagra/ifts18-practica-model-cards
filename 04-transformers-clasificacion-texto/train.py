"""Entrena un clasificador de texto con Hugging Face Transformers y lo exporta.

Problema: clasificar comentarios sobre departamentos en "positivo" o "negativo".

Es un BERT diminuto (2 capas, 32 dimensiones) entrenado DESDE CERO, con un tokenizer propio
de vocabulario cerrado. Así se ven todas las piezas que publica un modelo de Transformers:

    config.json              la arquitectura (cuántas capas, dimensiones, etiquetas)
    model.safetensors        los pesos
    tokenizer.json           el vocabulario y cómo se corta el texto en tokens
    tokenizer_config.json    la configuración del tokenizer y sus tokens especiales

En un caso real no se entrena desde cero: se hace fine-tuning de un modelo preentrenado
(por ejemplo, uno en español del Hub). Las piezas exportadas son las mismas.

Uso:
    python train.py
"""

import json
import random
from itertools import product
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

CARPETA = Path(__file__).parent
ETIQUETAS = ["negativo", "positivo"]
random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Datos: comentarios armados con plantillas (dataset sintético y chico)
# ---------------------------------------------------------------------------
PLANTILLAS = [
    "el departamento es muy {a}",
    "me pareció un lugar {a}",
    "la verdad es {a} y {b}",
    "un depto {a}, bastante {b}",
    "es {a} pero sobre todo {b}",
]
POSITIVOS = ["luminoso", "amplio", "tranquilo", "cómodo", "lindo", "seguro", "moderno", "cálido"]
NEGATIVOS = ["oscuro", "chico", "ruidoso", "húmedo", "feo", "inseguro", "viejo", "frío"]


def armar_ejemplos(adjetivos, etiqueta):
    ejemplos = []
    for plantilla, a, b in product(PLANTILLAS, adjetivos, adjetivos):
        if a != b:
            ejemplos.append((plantilla.format(a=a, b=b), etiqueta))
    return ejemplos


datos = armar_ejemplos(POSITIVOS, 1) + armar_ejemplos(NEGATIVOS, 0)
datos = list(dict.fromkeys(datos))  # sin repetidos
random.shuffle(datos)
corte = int(len(datos) * 0.8)
train, test = datos[:corte], datos[corte:]
print(f"{len(train)} comentarios de entrenamiento, {len(test)} de test")

# ---------------------------------------------------------------------------
# 2. Tokenizer: aprende el vocabulario de los textos de entrenamiento
# ---------------------------------------------------------------------------
tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.train_from_iterator(
    [texto for texto, _ in train],
    trainers.WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]),
)
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
    model_max_length=32,
)
print(f"Vocabulario: {tokenizer.vocab_size} tokens")

# ---------------------------------------------------------------------------
# 3. Modelo: un BERT diminuto con una cabeza de clasificación de 2 etiquetas
# ---------------------------------------------------------------------------
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    max_position_embeddings=32,
    num_labels=len(ETIQUETAS),
    id2label=dict(enumerate(ETIQUETAS)),
    label2id={e: i for i, e in enumerate(ETIQUETAS)},
    pad_token_id=tokenizer.pad_token_id,
)
modelo = BertForSequenceClassification(config)


def a_tensores(lote):
    textos, etiquetas = zip(*lote)
    entrada = tokenizer(list(textos), padding=True, return_tensors="pt", return_token_type_ids=False)
    return entrada, torch.tensor(etiquetas)


# ---------------------------------------------------------------------------
# 4. Entrenamiento: el mismo ciclo de siempre (predice, pérdida, optimizador)
# ---------------------------------------------------------------------------
optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
modelo.train()
for epoca in range(15):
    random.shuffle(train)
    perdida_total = 0.0
    for i in range(0, len(train), 16):
        entrada, etiquetas = a_tensores(train[i:i + 16])
        salida = modelo(**entrada, labels=etiquetas)  # la pérdida (entropía cruzada) viene en la salida
        optimizador.zero_grad()
        salida.loss.backward()
        optimizador.step()
        perdida_total += salida.loss.item()
    print(f"época {epoca:2d} · pérdida {perdida_total / (len(train) / 16):.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluación con comentarios que el modelo no vio
# ---------------------------------------------------------------------------
modelo.eval()
entrada, etiquetas = a_tensores(test)
with torch.no_grad():
    predichas = modelo(**entrada).logits.argmax(dim=-1)
print(f"Accuracy en test: {accuracy_score(etiquetas, predichas):.3f}")

# ---------------------------------------------------------------------------
# 6. Exportación: modelo (config + pesos) y tokenizer, en la misma carpeta
# ---------------------------------------------------------------------------
modelo.save_pretrained(CARPETA)      # config.json + model.safetensors
tokenizer.save_pretrained(CARPETA)   # tokenizer.json + tokenizer_config.json

# Compatibilidad: Transformers 5 anota una clase de tokenizer que la versión 4 (la de Colab)
# no conoce. PreTrainedTokenizerFast la entienden las dos.
ruta_config_tok = CARPETA / "tokenizer_config.json"
config_tok = json.loads(ruta_config_tok.read_text(encoding="utf-8"))
config_tok["tokenizer_class"] = "PreTrainedTokenizerFast"
config_tok.pop("backend", None)
ruta_config_tok.write_text(json.dumps(config_tok, indent=2, ensure_ascii=False), encoding="utf-8")
print("Modelo y tokenizer guardados en la carpeta")
