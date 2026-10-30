"""Arquitectura del modelo. La usan train.py y predict.py.

PyTorch no guarda la clase dentro del archivo de pesos: para cargar model.safetensors
hay que volver a crear el mismo modelo vacío y después cargarle los pesos.
"""

import torch


class RegresionLineal(torch.nn.Module):
    def __init__(self, n_entradas=3):
        super().__init__()
        self.lineal = torch.nn.Linear(n_entradas, 1)  # pesos (w) y bias (b)

    def forward(self, x):
        return self.lineal(x)
