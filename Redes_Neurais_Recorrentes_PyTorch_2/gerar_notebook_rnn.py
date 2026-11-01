import json
import pathlib
import sys

pasta = pathlib.Path(sys.argv[1])
codigo = (pasta / "rnn_alura_completo.py").read_text(encoding="utf-8")

secoes = [
    ("Bibliotecas e configuracao", "import os"),
    ("Normalizacao e leitura de dados", "def normalizar"),
    ("Conversao para tensores", "def nome2tensor"),
    ("Arquitetura RNN", "class RNN"),
    ("Treinamento", "# Treinamento demonstrativo"),
    ("Previsoes", "def prever"),
]

marcadores = [codigo.index(texto) for _, texto in secoes]

celulas = []

for i, (titulo, _) in enumerate(secoes):
    inicio = marcadores[i]
    fim = marcadores[i + 1] if i + 1 < len(marcadores) else len(codigo)

    celulas.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {titulo}\n"]
    })

    celulas.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": codigo[inicio:fim].splitlines(keepends=True)
    })

notebook = {
    "cells": celulas,
    "metadata": {
        "kernelspec": {
            "display_name": "RNN Alura - Python 3.11",
            "language": "python",
            "name": "rnn-alura311"
        },
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

destino = pasta / "RNN_Alura_Exercicio_Completo.ipynb"
destino.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Notebook criado:", destino)