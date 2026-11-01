import ast
import json
import os
import random
import re
import string
import sys
import time
import traceback
import unicodedata
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ORIGINAL = ROOT / "Classificação de Sequências (1).ipynb"

OUT.mkdir(exist_ok=True)
LOG = OUT / "diagnostico.txt"

def registrar(texto):
    print(texto, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(str(texto) + "\n")

LOG.write_text(
    "DIAGNOSTICO RNN ALURA\n",
    encoding="utf-8"
)

def celula(codigo):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": codigo.splitlines(keepends=True)
    }

def markdown(texto):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": texto.splitlines(keepends=True)
    }

def salvar_notebook(caminho, cells, metadata=None):
    notebook = {
        "cells": cells,
        "metadata": metadata or {
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
    caminho.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

registrar("1. ANALISE DO NOTEBOOK ORIGINAL")

with ORIGINAL.open(encoding="utf-8") as f:
    nb = json.load(f)

corrigidas = []
alteracoes = 0
problemas = 0

for indice, cell in enumerate(nb["cells"]):
    nova = dict(cell)

    if cell.get("cell_type") != "code":
        corrigidas.append(nova)
        continue

    source = cell.get("source", [])
    codigo = "".join(source) if isinstance(source, list) else source
    anterior = codigo

    # Corrige apenas padroes conhecidos da aula.
    codigo, n = re.subn(
        r"\.decode\(\s*['\"]utf-8['\"]\s*\)",
        "",
        codigo
    )
    alteracoes += n

    codigo, n = re.subn(
        r"nn\.LogSoftmax\(\s*\)",
        "nn.LogSoftmax(dim=1)",
        codigo
    )
    alteracoes += n

    if codigo != anterior:
        registrar(f"Celula {indice}: correcoes conhecidas aplicadas.")

    try:
        ast.parse(codigo)
    except SyntaxError as erro:
        problemas += 1
        registrar(
            f"Celula {indice}: sintaxe pendente - "
            f"linha {erro.lineno}: {erro.msg}"
        )

    nova["source"] = codigo.splitlines(keepends=True)
    nova["execution_count"] = None
    nova["outputs"] = []
    corrigidas.append(nova)

registrar(f"Celulas analisadas: {len(nb['cells'])}")
registrar(f"Substituicoes controladas: {alteracoes}")
registrar(f"Celulas com sintaxe pendente: {problemas}")

original_corrigido = OUT / "Classificacao_Original_Corrigido.ipynb"

salvar_notebook(
    original_corrigido,
    corrigidas,
    nb.get("metadata")
)

registrar("Notebook original corrigido salvo.")

registrar("2. VALIDACAO DO AMBIENTE")

import numpy as np
import torch
from torch import nn

registrar(f"Python: {sys.version.split()[0]}")
registrar(f"PyTorch: {torch.__version__}")
registrar(f"NumPy: {np.__version__}")

torch.manual_seed(42)
random.seed(42)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

registrar(f"Dispositivo: {DEVICE}")

# Testa a inicializacao real do otimizador.
teste = nn.Linear(3, 2).to(DEVICE)
teste_adam = torch.optim.Adam(teste.parameters(), lr=0.001)
registrar("Otimizador Adam: OK")

registrar("3. CARREGAMENTO DOS SOBRENOMES")

PASTA = ROOT / "data" / "names"
ALFABETO = string.ascii_letters + " .'-"
ARQUIVOS = sorted(PASTA.glob("*.txt"))

if len(ARQUIVOS) < 2:
    raise RuntimeError(
        "Sao necessarios pelo menos dois arquivos TXT."
    )

def normalizar(nome):
    decomposicao = unicodedata.normalize("NFKD", nome)
    sem_acentos = "".join(
        c for c in decomposicao
        if not unicodedata.combining(c)
    )
    return "".join(c for c in sem_acentos if c in ALFABETO)

categorias = []
amostras = []

for arquivo in ARQUIVOS:
    nomes = arquivo.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    indice = len(categorias)
    categorias.append(arquivo.stem)

    for nome in nomes:
        nome = normalizar(nome.strip())
        if nome:
            amostras.append((nome, indice))

if not amostras:
    raise RuntimeError("Nenhum sobrenome valido encontrado.")

registrar(f"Categorias: {len(categorias)}")
registrar(f"Total de sobrenomes: {len(amostras)}")

def nome_tensor(nome):
    nome = normalizar(nome)

    if not nome:
        raise ValueError("Sobrenome vazio.")

    tensor = torch.zeros(
        len(nome), len(ALFABETO),
        device=DEVICE
    )

    for i, letra in enumerate(nome):
        tensor[i, ALFABETO.index(letra)] = 1

    return tensor

class ClassificadorRNN(nn.Module):
    def __init__(self, entrada, oculto, saida):
        super().__init__()
        self.oculto = oculto
        self.rnn = nn.RNNCell(entrada, oculto)
        self.linear = nn.Linear(oculto, saida)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, entrada):
        hidden = torch.zeros(
            1, self.oculto,
            device=entrada.device
        )

        for caractere in entrada:
            hidden = self.rnn(
                caractere.unsqueeze(0),
                hidden
            )

        return self.softmax(self.linear(hidden))

modelo = ClassificadorRNN(
    len(ALFABETO),
    64,
    len(categorias)
).to(DEVICE)

saida_teste = modelo(nome_tensor("Araujo"))

assert saida_teste.shape == (1, len(categorias))
registrar("Teste de dimensoes da RNN: OK")

registrar("4. SEPARACAO TREINO E TESTE")

# Divisao por categoria para preservar representatividade.
por_categoria = {}

for nome, indice in amostras:
    por_categoria.setdefault(indice, []).append(nome)

treino = []
teste = []

for indice, nomes in por_categoria.items():
    random.shuffle(nomes)

    if len(nomes) < 2:
        registrar(
            f"Categoria {categorias[indice]} possui "
            "menos de 2 amostras: sem teste independente."
        )
        treino.extend((nome, indice) for nome in nomes)
        continue

    corte = max(1, int(len(nomes) * 0.2))
    corte = min(corte, len(nomes) - 1)

    teste.extend((nome, indice) for nome in nomes[:corte])
    treino.extend((nome, indice) for nome in nomes[corte:])

registrar(f"Treinamento: {len(treino)} amostras")
registrar(f"Teste: {len(teste)} amostras")

criterio = nn.NLLLoss()
otimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=0.001
)

EPOCAS = 3
LIMITE = 300

registrar(
    "5. TREINAMENTO: 3 epocas, ate 300 amostras por epoca"
)

historico = []
inicio = time.time()

for epoca in range(EPOCAS):
    modelo.train()

    lote = random.sample(
        treino,
        min(LIMITE, len(treino))
    )

    perdas = []
    acertos = 0

    for nome, indice in lote:
        entrada = nome_tensor(nome)
        alvo = torch.tensor(
            [indice],
            dtype=torch.long,
            device=DEVICE
        )

        otimizador.zero_grad()
        saida = modelo(entrada)
        perda = criterio(saida, alvo)
        perda.backward()
        otimizador.step()

        perdas.append(perda.item())
        acertos += int(
            saida.argmax(dim=1).item() == indice
        )

    perda_media = sum(perdas) / len(perdas)
    acuracia = acertos / len(lote)

    historico.append({
        "epoca": epoca + 1,
        "perda": perda_media,
        "acuracia_treino": acuracia
    })

    registrar(
        f"Epoca {epoca + 1}/{EPOCAS}: "
        f"perda={perda_media:.4f}, "
        f"acuracia_treino={acuracia:.2%}"
    )

registrar(
    f"Tempo de treinamento: {time.time() - inicio:.1f}s"
)

registrar("6. AVALIACAO")

modelo.eval()
acertos = 0

with torch.no_grad():
    for nome, indice in teste:
        saida = modelo(nome_tensor(nome))
        acertos += int(
            saida.argmax(dim=1).item() == indice
        )

acuracia_teste = (
    acertos / len(teste) if teste else None
)

if acuracia_teste is not None:
    registrar(f"Acuracia no teste: {acuracia_teste:.2%}")
else:
    registrar("Teste independente indisponivel.")

def prever(nome, top=3):
    modelo.eval()

    with torch.no_grad():
        saida = modelo(nome_tensor(nome))
        probabilidades = saida.exp()
        valores, indices = probabilidades.topk(
            min(top, len(categorias)),
            dim=1
        )

    return [
        {
            "categoria": categorias[indice],
            "probabilidade": valor
        }
        for valor, indice in zip(
            valores[0].tolist(),
            indices[0].tolist()
        )
    ]

previsoes = {}

for sobrenome in ["Araujo", "Cabral", "Schmidt"]:
    previsoes[sobrenome] = prever(sobrenome)
    registrar(f"Previsao {sobrenome}: {previsoes[sobrenome]}")

registrar("7. SALVANDO RESULTADOS")

torch.save(
    {
        "pesos": modelo.state_dict(),
        "categorias": categorias,
        "alfabeto": ALFABETO,
        "oculto": 64,
        "entrada": len(ALFABETO),
        "saida": len(categorias),
        "epocas": EPOCAS
    },
    OUT / "modelo_rnn.pt"
)

resultados = {
    "categorias": categorias,
    "historico": historico,
    "acuracia_teste": acuracia_teste,
    "previsoes": previsoes,
    "treino": len(treino),
    "teste": len(teste)
}

(OUT / "resultados.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# Cria notebook com resultados reais do treinamento.
cells = [
    markdown(
        "# RNN Alura - Treinamento validado\n\n"
        "Este notebook documenta os resultados da execucao "
        "real realizada pelo PowerShell.\n\n"
        "O notebook original corrigido esta salvo separadamente.\n"
    ),
    celula(
        "import json\n"
        "from pathlib import Path\n"
        "resultados = json.loads("
        "Path('RNN_Alura_Validado/resultados.json').read_text("
        "encoding='utf-8'))\n"
        "print('Categorias:', len(resultados['categorias']))\n"
        "print('Acuracia de teste:', resultados['acuracia_teste'])\n"
    ),
    celula(
        "import matplotlib.pyplot as plt\n"
        "historico = resultados['historico']\n"
        "plt.plot([x['epoca'] for x in historico], "
        "[x['perda'] for x in historico], marker='o')\n"
        "plt.xlabel('Epoca')\n"
        "plt.ylabel('Perda media')\n"
        "plt.title('Treinamento RNN')\n"
        "plt.grid(True)\n"
        "plt.show()\n"
    ),
    celula(
        "for nome, previsoes in resultados['previsoes'].items():\n"
        "    print('\\n', nome)\n"
        "    for item in previsoes:\n"
        "        print(item['categoria'], "
        "f\"{item['probabilidade']:.2%}\")\n"
    )
]

salvar_notebook(
    OUT / "RNN_Alura_Resultados.ipynb",
    cells
)

registrar("Modelo salvo e notebook de resultados criado.")
registrar("AUTOMACAO CONCLUIDA.")