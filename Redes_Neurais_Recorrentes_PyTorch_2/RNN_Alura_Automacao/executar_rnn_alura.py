import ast
import copy
import json
import math
import random
import re
import string
import sys
import time
import traceback
import unicodedata
from collections import Counter
from pathlib import Path

import nbformat
import numpy as np
import torch
from torch import nn

ROOT = Path.cwd()
OUT = ROOT / "RNN_Alura_Automacao"
ORIGINAL = ROOT / "Classificação de Sequências (1).ipynb"
CORRIGIDO = OUT / "Classificacao_Sequencias_Corrigido.ipynb"
RELATORIO = OUT / "relatorio_validacao.json"
DIAGNOSTICO = OUT / "diagnostico_python.txt"
MODELO = OUT / "modelo_rnn_treinado.pt"
RESULTADOS = OUT / "resultados_treinamento.json"
NOTEBOOK_RESULTADOS = OUT / "RNN_Alura_Execucao_Validada.ipynb"

SEMENTE = 42
EPOCAS = 3
AMOSTRAS_POR_CLASSE = 50
HIDDEN = 256

random.seed(SEMENTE)
np.random.seed(SEMENTE)
torch.manual_seed(SEMENTE)

def registrar(mensagem):
    print(mensagem, flush=True)
    with DIAGNOSTICO.open("a", encoding="utf-8") as f:
        f.write(str(mensagem) + "\n")

def celula_codigo(source):
    return nbformat.v4.new_code_cell(source)

def celula_markdown(source):
    return nbformat.v4.new_markdown_cell(source)

def salvar_json(caminho, objeto):
    caminho.write_text(
        json.dumps(objeto, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def analisar_original():
    registrar("ANALISANDO NOTEBOOK ORIGINAL")

    nb = nbformat.read(str(ORIGINAL), as_version=4)
    nbformat.validate(nb)

    copia = copy.deepcopy(nb)
    problemas = []
    correcoes = []

    for indice, celula in enumerate(copia.cells):
        if celula.cell_type != "code":
            continue

        fonte = celula.source
        linhas = fonte.splitlines()

        # Reconhece comandos exclusivos do Jupyter.
        magicas = any(
            linha.lstrip().startswith(("%", "!", "?"))
            for linha in linhas
        )

        # Corrige somente a chamada sem dimensao,
        # quando reconhecida com seguranca.
        padrao = r"\bnn\.LogSoftmax\(\s*\)"
        novo = re.sub(
            padrao,
            "nn.LogSoftmax(dim=1)",
            fonte
        )

        if novo != fonte:
            correcoes.append({
                "celula": indice + 1,
                "tipo": "LogSoftmax sem dim"
            })
            fonte = novo

        # Detecta usos de decode, mas nao substitui
        # indiscriminadamente: bytes e str exigem
        # tratamentos diferentes.
        if ".decode(" in fonte:
            problemas.append({
                "celula": indice + 1,
                "tipo": "decode requer verificacao de tipo"
            })

        if not magicas:
            try:
                ast.parse(fonte)
            except SyntaxError as erro:
                problemas.append({
                    "celula": indice + 1,
                    "tipo": "sintaxe",
                    "linha": erro.lineno,
                    "mensagem": erro.msg
                })
        else:
            problemas.append({
                "celula": indice + 1,
                "tipo": "sintaxe especial Jupyter"
            })

        celula.source = fonte

    nbformat.write(copia, str(CORRIGIDO))

    relatorio = {
        "celulas_originais": len(nb.cells),
        "correcoes_seguras": correcoes,
        "pendencias": problemas,
        "original_modificado": False,
        "notebook_corrigido": str(CORRIGIDO),
        "execucao_integral_original": False
    }

    salvar_json(RELATORIO, relatorio)

    registrar(
        f"Celulas analisadas: {len(nb.cells)}; "
        f"correcoes: {len(correcoes)}; "
        f"pendencias: {len(problemas)}"
    )

    return relatorio

def normalizar(nome):
    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(
        letra for letra in nome
        if not unicodedata.combining(letra)
    )
    return "".join(
        letra for letra in nome
        if letra in DICIONARIO
    )

DICIONARIO = string.ascii_letters + " .'-"

def carregar_dados():
    pasta = ROOT / "data" / "names"
    arquivos = sorted(pasta.glob("*.txt"))

    if len(arquivos) < 2:
        raise ValueError("Numero insuficiente de categorias.")

    categorias = []
    grupos = {}

    for arquivo in arquivos:
        categoria = arquivo.stem
        nomes = []

        with arquivo.open(
            "r", encoding="utf-8-sig"
        ) as f:
            for linha in f:
                nome = normalizar(linha.strip())
                if nome:
                    nomes.append(nome)

        if len(nomes) < 2:
            raise ValueError(
                f"Categoria com menos de 2 nomes: {categoria}"
            )

        categorias.append(categoria)
        grupos[categoria] = nomes

    return categorias, grupos

def dividir_dados(categorias, grupos):
    treino = {}
    teste = {}

    gerador = np.random.default_rng(SEMENTE)

    for categoria in categorias:
        nomes = grupos[categoria].copy()
        indices = gerador.permutation(len(nomes))

        corte = max(
            1,
            min(len(nomes) - 1, int(len(nomes) * 0.8))
        )

        treino[categoria] = [
            nomes[i] for i in indices[:corte]
        ]

        teste[categoria] = [
            nomes[i] for i in indices[corte:]
        ]

    return treino, teste

def nome_tensor(nome):
    tensor = torch.zeros(
        len(nome),
        len(DICIONARIO),
        dtype=torch.float32
    )

    for i, letra in enumerate(nome):
        tensor[i, DICIONARIO.index(letra)] = 1.0

    return tensor

class RNN(nn.Module):
    def __init__(self, entrada, hidden, saida):
        super().__init__()
        self.hidden = hidden
        self.recorrente = nn.RNNCell(entrada, hidden)
        self.saida = nn.Linear(hidden, saida)
        self.ativacao = nn.LogSoftmax(dim=1)

    def forward(self, nome):
        h = torch.zeros(
            1,
            self.hidden,
            device=nome.device,
            dtype=nome.dtype
        )

        for letra in nome:
            h = self.recorrente(
                letra.unsqueeze(0),
                h
            )

        return self.ativacao(self.saida(h))

def amostrar_treino(categorias, treino, quantidade):
    exemplos = []

    for indice, categoria in enumerate(categorias):
        nomes = treino[categoria]

        escolhidos = np.random.choice(
            len(nomes),
            size=quantidade,
            replace=True
        )

        for posicao in escolhidos:
            exemplos.append(
                (nomes[int(posicao)], indice)
            )

    random.shuffle(exemplos)
    return exemplos

def executar_epoca(
    modelo, exemplos, criterio,
    device, otimizador=None
):
    treinando = otimizador is not None

    modelo.train(treinando)

    perdas = []
    acertos = 0

    for nome, indice in exemplos:
        x = nome_tensor(nome).to(device)
        y = torch.tensor(
            [indice],
            dtype=torch.long,
            device=device
        )

        if treinando:
            otimizador.zero_grad()

        with torch.set_grad_enabled(treinando):
            saida = modelo(x)
            perda = criterio(saida, y)

            if treinando:
                perda.backward()
                otimizador.step()

        perdas.append(float(perda.item()))
        acertos += int(
            saida.argmax(dim=1).item() == indice
        )

    return {
        "loss_media": float(np.mean(perdas)),
        "loss_desvio": float(np.std(perdas)),
        "acuracia": acertos / len(exemplos),
        "amostras": len(exemplos)
    }

def prever(modelo, nome, categorias, device):
    normalizado = normalizar(nome)

    if not normalizado:
        raise ValueError("Sobrenome sem caracteres validos.")

    modelo.eval()

    with torch.no_grad():
        saida = modelo(
            nome_tensor(normalizado).to(device)
        )

        probabilidades = saida.exp().squeeze(0)
        valores, indices = torch.topk(
            probabilidades,
            k=min(3, len(categorias))
        )

    return [
        {
            "categoria": categorias[int(i)],
            "probabilidade": round(float(v), 6)
        }
        for v, i in zip(
            valores.cpu().tolist(),
            indices.cpu().tolist()
        )
    ]

def criar_notebook_resultados(resultados):
    nb = nbformat.v4.new_notebook()

    nb.metadata.kernelspec = {
        "display_name": "RNN Alura - Python 3.11",
        "language": "python",
        "name": "rnn-alura311"
    }

    nb.cells = [
        celula_markdown(
            "# RNN Alura - Execucao validada\n\n"
            "Resultados de um treinamento supervisionado "
            "independente, baseado na arquitetura da aula.\n\n"
            "O notebook original foi preservado."
        ),
        celula_codigo(
            "import json\n"
            "from pathlib import Path\n"
            "p = Path('RNN_Alura_Automacao')\n"
            "resultados = json.loads("
            "(p / 'resultados_treinamento.json')"
            ".read_text(encoding='utf-8'))\n"
            "print(json.dumps(resultados, "
            "ensure_ascii=False, indent=2))"
        ),
        celula_markdown(
            "## Observacoes\n\n"
            "- Entrada One-hot de 56 caracteres.\n"
            "- RNNCell com 256 unidades.\n"
            "- Saida LogSoftmax com 18 classes "
            "(se o conjunto tiver 18 categorias).\n"
            "- NLLLoss e Adam.\n"
            "- Separacao treino/teste por categoria.\n"
            "- Treinamento limitado a tres epocas.\n"
            "- Acuracia medida em dados separados.\n"
            "- O desempenho nao equivale a um "
            "treinamento completo do curso."
        )
    ]

    nbformat.write(nb, str(NOTEBOOK_RESULTADOS))

def main():
    inicio = time.perf_counter()

    relatorio = analisar_original()

    categorias, grupos = carregar_dados()
    treino, teste = dividir_dados(categorias, grupos)

    total = sum(map(len, grupos.values()))

    registrar(
        f"Dados: {total} nomes; "
        f"{len(categorias)} categorias"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    registrar(f"Dispositivo: {device}")
    registrar(
        f"Treinamento: {EPOCAS} epocas; "
        f"{AMOSTRAS_POR_CLASSE} exemplos por categoria "
        "em cada epoca."
    )

    modelo = RNN(
        len(DICIONARIO),
        HIDDEN,
        len(categorias)
    ).to(device)

    criterio = nn.NLLLoss().to(device)

    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=5e-5,
        weight_decay=1e-7
    )

    # Teste estrutural antes do treinamento.
    amostra = nome_tensor("Araujo").to(device)

    with torch.no_grad():
        saida = modelo(amostra)

    assert saida.shape == (1, len(categorias))

    assert torch.allclose(
        saida.exp().sum(dim=1),
        torch.ones(1, device=device),
        atol=1e-5
    )

    registrar("Teste estrutural da RNN: APROVADO")

    historico = []

    for epoca in range(1, EPOCAS + 1):
        exemplos = amostrar_treino(
            categorias,
            treino,
            AMOSTRAS_POR_CLASSE
        )

        metricas = executar_epoca(
            modelo,
            exemplos,
            criterio,
            device,
            otimizador
        )

        metricas["epoca"] = epoca
        historico.append(metricas)

        registrar(
            f"Epoca {epoca}/{EPOCAS} | "
            f"Loss {metricas['loss_media']:.4f} | "
            f"Acc {metricas['acuracia']:.2%}"
        )

    # Avaliacao somente depois do treinamento.
    exemplos_teste = []

    for indice, categoria in enumerate(categorias):
        for nome in teste[categoria]:
            exemplos_teste.append((nome, indice))

    metricas_teste = executar_epoca(
        modelo,
        exemplos_teste,
        criterio,
        device
    )

    registrar(
        f"Teste separado | "
        f"Loss {metricas_teste['loss_media']:.4f} | "
        f"Acc {metricas_teste['acuracia']:.2%}"
    )

    previsoes = {}

    for sobrenome in (
        "Araujo",
        "Cabral",
        "Schmidt"
    ):
        previsoes[sobrenome] = prever(
            modelo,
            sobrenome,
            categorias,
            device
        )

    # Salva somente apos treinamento e avaliacao.
    torch.save({
        "model_state_dict": modelo.state_dict(),
        "categorias": categorias,
        "dicionario": DICIONARIO,
        "hidden_size": HIDDEN,
        "epocas": EPOCAS,
        "treinado": True,
        "semente": SEMENTE
    }, MODELO)

    # Testa carregamento do checkpoint salvo.
    checkpoint = torch.load(
        MODELO,
        map_location=device,
        weights_only=True
    )

    modelo_recarregado = RNN(
        len(checkpoint["dicionario"]),
        checkpoint["hidden_size"],
        len(checkpoint["categorias"])
    ).to(device)

    modelo_recarregado.load_state_dict(
        checkpoint["model_state_dict"]
    )

    modelo_recarregado.eval()

    with torch.no_grad():
        saida_recarregada = modelo_recarregado(
            nome_tensor("Araujo").to(device)
        )

    assert torch.isfinite(saida_recarregada).all()

    registrar("Checkpoint recarregado: APROVADO")

    resultados = {
        "status": "executado_e_validado",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "categorias": categorias,
        "total_nomes": total,
        "total_treino": sum(map(len, treino.values())),
        "total_teste": len(exemplos_teste),
        "epocas": EPOCAS,
        "amostras_por_classe_por_epoca": AMOSTRAS_POR_CLASSE,
        "historico_treino": historico,
        "metricas_teste": metricas_teste,
        "previsoes_top3": previsoes,
        "tempo_segundos": round(
            time.perf_counter() - inicio, 2
        ),
        "notebook_original_executado_integralmente": False,
        "notebook_original_preservado": True,
        "pendencias_notebook_original": relatorio["pendencias"],
        "adaptacoes": [
            "Implementacao supervisionada independente",
            "Entrada dimensionada pelo dicionario real",
            "Amostragem balanceada com reposicao",
            "Separacao treino/teste antes da amostragem",
            "Limite de tres epocas",
            "Sem executar celulas arbitrarias do original"
        ]
    }

    salvar_json(RESULTADOS, resultados)
    criar_notebook_resultados(resultados)

    registrar("MODELO TREINADO E VALIDADO")
    registrar(f"Modelo: {MODELO}")
    registrar(f"Resultados: {RESULTADOS}")
    registrar(f"Notebook: {NOTEBOOK_RESULTADOS}")
    registrar("AUTOMACAO CONCLUIDA")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        registrar("ERRO NA AUTOMACAO")
        registrar(traceback.format_exc())
        raise