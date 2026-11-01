import json
from pathlib import Path

pasta = Path.cwd()
destino = pasta / "Balanceamento_de_Amostras.ipynb"

def codigo(texto):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": texto.splitlines(keepends=True)
    }

def texto(conteudo):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": conteudo.splitlines(keepends=True)
    }

celulas = [
    texto(
        "# Balanceamento de Amostras\n\n"
        "Exercicio da Alura: selecionar 50 sobrenomes "
        "de cada uma das 18 nacionalidades."
    ),
    codigo(
        "import os\n"
        "import string\n"
        "import unicodedata\n"
        "import numpy as np\n"
        "import torch\n\n"
        "arquivos = sorted(\n"
        "    arq for arq in os.listdir('data/names')\n"
        "    if arq.endswith('.txt')\n"
        ")\n"
        "categorias = [os.path.splitext(arq)[0] for arq in arquivos]\n\n"
        "def normalizar(nome):\n"
        "    nome = unicodedata.normalize('NFKD', nome)\n"
        "    nome = ''.join(c for c in nome if not unicodedata.combining(c))\n"
        "    return ''.join(c for c in nome if c in string.ascii_letters + \" .'-\")\n\n"
        "dados = []\n"
        "rotulos = []\n"
        "for arquivo, categoria in zip(arquivos, categorias):\n"
        "    with open(os.path.join('data/names', arquivo), encoding='utf-8-sig') as f:\n"
        "        nomes = [normalizar(linha.strip()) for linha in f]\n"
        "    nomes = [nome for nome in nomes if nome]\n"
        "    dados.append(nomes)\n"
        "    rotulos.append([categoria] * len(nomes))\n\n"
        "print('Categorias:', len(categorias))\n"
        "print('Total de sobrenomes:', sum(map(len, dados)))\n"
    ),
    texto(
        "## Conversao para tensores\n\n"
        "Cada letra recebe uma representacao One-hot. "
        "Cada nacionalidade recebe um indice inteiro."
    ),
    codigo(
        "dicionario = string.ascii_letters + \" .'-\"\n\n"
        "def name2tensor(nome):\n"
        "    nome = normalizar(nome)\n"
        "    tensor = torch.zeros(len(nome), len(dicionario))\n"
        "    for k, letra in enumerate(nome):\n"
        "        tensor[k, dicionario.index(letra)] = 1\n"
        "    return tensor\n\n"
        "def label2tensor(rotulo):\n"
        "    return torch.LongTensor([categorias.index(rotulo)])\n\n"
        "print('Exemplo:', dados[0][0])\n"
        "print('Tensor:', name2tensor(dados[0][0]).shape)\n"
        "print('Rotulo:', label2tensor(rotulos[0][0]))\n"
    ),
    texto(
        "## Amostragem balanceada\n\n"
        "A cada chamada selecionamos 50 indices por categoria. "
        "A selecao e feita com reposicao, preservando as duplicatas."
    ),
    codigo(
        "def amostrar(num_amostras=50):\n"
        "    dados_amostra = []\n"
        "    rotulos_amostra = []\n\n"
        "    for indice, categoria in enumerate(categorias):\n"
        "        num_nomes = len(dados[indice])\n"
        "        if num_nomes == 0:\n"
        "            raise ValueError(f'Categoria vazia: {categoria}')\n\n"
        "        indices = np.random.choice(\n"
        "            range(num_nomes), size=num_amostras, replace=True\n"
        "        )\n\n"
        "        for posicao in indices:\n"
        "            dados_amostra.append(name2tensor(dados[indice][posicao]))\n"
        "            rotulos_amostra.append(label2tensor(categoria))\n\n"
        "    return dados_amostra, rotulos_amostra\n\n"
        "dados_tns, rotulos_tns = amostrar(50)\n"
        "print('Total de amostras:', len(dados_tns))\n"
        "print('Total de rotulos:', len(rotulos_tns))\n"
        "print('Primeiro tensor:', dados_tns[0].shape)\n"
        "print('Primeiro rotulo:', rotulos_tns[0])\n"
    ),
    texto(
        "## Verificacao do balanceamento\n\n"
        "Confirmamos que cada nacionalidade aparece exatamente 50 vezes."
    ),
    codigo(
        "from collections import Counter\n\n"
        "contagem = Counter(int(rotulo.item()) for rotulo in rotulos_tns)\n"
        "for indice, quantidade in sorted(contagem.items()):\n"
        "    print(f'{categorias[indice]}: {quantidade}')\n\n"
        "assert len(dados_tns) == 50 * len(categorias)\n"
        "assert len(rotulos_tns) == len(dados_tns)\n"
        "assert all(quantidade == 50 for quantidade in contagem.values())\n"
        "print('BALANCEAMENTO VALIDADO')\n"
    )
]

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

destino.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Notebook criado:", destino)