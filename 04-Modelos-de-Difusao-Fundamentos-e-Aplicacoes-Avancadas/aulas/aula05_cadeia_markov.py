# ==========================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 05 - ESTRUTURA MARKOVIANA
# ==========================================================

from pathlib import Path
import csv
import json

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
QUADROS = RESULTADOS / "aula05_quadros"

RESULTADOS.mkdir(parents=True, exist_ok=True)
QUADROS.mkdir(parents=True, exist_ok=True)

GIF = RESULTADOS / "aula05_cadeia_markov.gif"
PNG = RESULTADOS / "aula05_grafo_markov.png"
CSV = RESULTADOS / "aula05_historico_markov.csv"
JSON = RESULTADOS / "aula05_parametros_markov.json"

np.random.seed(42)

print("=" * 65)
print("AULA 05 - ESTRUTURA MARKOVIANA")
print("=" * 65)


# ==========================================================
# 2 - DEFINIR ESTADOS
# ==========================================================

print("\n1 - DEFININDO ESTADOS")

states = [
    "Casa",
    "Trabalho",
    "Mercado",
    "Parque"
]

for i, state in enumerate(states):
    print(f"{i} - {state}")


# ==========================================================
# 3 - DEFINIR MATRIZ DE TRANSICAO
# ==========================================================

print("\n2 - CRIANDO MATRIZ DE TRANSICAO")

transition_matrix = np.array([
    [0.1, 0.6, 0.2, 0.1],
    [0.4, 0.2, 0.3, 0.1],
    [0.5, 0.1, 0.1, 0.3],
    [0.3, 0.1, 0.2, 0.4]
])

print("\nMATRIZ:")
print(transition_matrix)

# Cada linha deve somar 1.0
assert transition_matrix.shape == (4, 4)

assert np.allclose(
    transition_matrix.sum(axis=1),
    1.0
)

assert np.all(
    transition_matrix >= 0
)

print("\nMatriz de probabilidades validada: OK")


# ==========================================================
# 4 - CRIAR GRAFO DIRECIONADO
# ==========================================================

print("\n3 - CRIANDO GRAFO")

G = nx.DiGraph()

for i, state_from in enumerate(states):

    for j, state_to in enumerate(states):

        G.add_edge(
            state_from,
            state_to,
            weight=float(
                transition_matrix[i, j]
            )
        )

edge_labels = {
    (sf, st): f"{transition_matrix[i, j]:.1f}"
    for i, sf in enumerate(states)
    for j, st in enumerate(states)
}

pos = {
    "Casa": (0, 0),
    "Trabalho": (2, 1),
    "Mercado": (1, -1),
    "Parque": (-1, 1)
}

print("Nos:", G.number_of_nodes())
print("Transicoes:", G.number_of_edges())

assert G.number_of_nodes() == 4
assert G.number_of_edges() == 16


# ==========================================================
# 5 - FUNCAO PARA PROXIMO ESTADO
# ==========================================================

print("\n4 - CRIANDO FUNCAO NEXT_STATE")


def next_state(current_state):

    indice = states.index(
        current_state
    )

    probs = transition_matrix[indice]

    return np.random.choice(
        states,
        p=probs
    )


print("Funcao next_state: OK")


# ==========================================================
# 6 - SIMULAR CADEIA DE MARKOV
# ==========================================================

print("\n5 - SIMULANDO DESLOCAMENTOS")

current = "Casa"
steps = 10

history = [current]

print("Passo 0:", current)

for i in range(steps):

    current = next_state(current)

    history.append(current)

    print(
        f"Passo {i + 1}: {current}"
    )

assert len(history) == 11

assert all(
    state in states
    for state in history
)

print("\nHistorico de 10 transicoes: OK")


# ==========================================================
# 7 - FUNCAO DE VISUALIZACAO DO GRAFO
# ==========================================================

def desenhar_grafo(passo, estado_atual):

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    # Desenhar todos os nos
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="lightgray",
        node_size=2300,
        edgecolors="black"
    )

    # Destacar o estado atual
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[estado_atual],
        ax=ax,
        node_color="orange",
        node_size=2300,
        edgecolors="black"
    )

    # Desenhar as transicoes direcionadas
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=18,
        arrowstyle="-|>",
        node_size=2300,
        alpha=0.45,
        connectionstyle="arc3,rad=0.12"
    )

    # Nomes dos estados
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=11,
        font_weight="bold"
    )

    # Probabilidades das transicoes
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax,
        font_size=7,
        font_color="blue",
        rotate=False,
        label_pos=0.35
    )

    ax.set_title(
        f"Passo {passo}: atualmente em {estado_atual}",
        fontsize=14
    )

    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 2)

    ax.axis("off")

    fig.tight_layout()

    return fig


# ==========================================================
# 8 - CRIAR QUADROS E GIF
# ==========================================================

print("\n6 - CRIANDO ANIMACAO")

filenames = []

for i, state in enumerate(history):

    fig = desenhar_grafo(
        i,
        state
    )

    fname = (
        QUADROS
        / f"markov_step_{i:02d}.png"
    )

    fig.savefig(
        fname,
        dpi=110,
        bbox_inches="tight"
    )

    plt.close(fig)

    filenames.append(fname)

    print(
        f"Quadro {i + 1}/11: OK"
    )

print("\n7 - SALVANDO GIF")

frames = [
    imageio.imread(filename)
    for filename in filenames
]

# As imagens devem ter dimensoes iguais
assert len({
    frame.shape for frame in frames
}) == 1

imageio.mimsave(
    GIF,
    frames,
    format="GIF",
    duration=0.8,
    loop=0
)

print("GIF salvo:", GIF)


# ==========================================================
# 9 - SALVAR IMAGEM FINAL
# ==========================================================

import shutil

shutil.copyfile(
    filenames[-1],
    PNG
)

print("Imagem PNG salva:", PNG)


# ==========================================================
# 10 - SALVAR HISTORICO CSV
# ==========================================================

print("\n8 - SALVANDO HISTORICO")

with CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as arquivo:

    writer = csv.writer(arquivo)

    writer.writerow([
        "passo",
        "estado",
        "estado_anterior",
        "probabilidade_transicao"
    ])

    for i, state in enumerate(history):

        if i == 0:

            anterior = ""
            probabilidade = ""

        else:

            anterior = history[i - 1]

            probabilidade = float(
                transition_matrix[
                    states.index(anterior),
                    states.index(state)
                ]
            )

        writer.writerow([
            i,
            state,
            anterior,
            probabilidade
        ])

print("CSV salvo: OK")


# ==========================================================
# 11 - SALVAR PARAMETROS JSON
# ==========================================================

dados_json = {
    "aula": 5,
    "tema": "Estrutura Markoviana",
    "seed": 42,
    "estado_inicial": "Casa",
    "total_transicoes": steps,
    "total_estados_historico": len(history),
    "estados": states,
    "matriz_transicao": transition_matrix.tolist(),
    "historico": history
}

with JSON.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        dados_json,
        arquivo,
        indent=4,
        ensure_ascii=False
    )

print("JSON salvo: OK")


# ==========================================================
# 12 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert len(states) == 4
assert transition_matrix.shape == (4, 4)

assert np.allclose(
    transition_matrix.sum(axis=1),
    1.0
)

assert G.number_of_nodes() == 4
assert G.number_of_edges() == 16

assert len(history) == 11
assert len(filenames) == 11

for arquivo in [GIF, PNG, CSV, JSON]:

    assert arquivo.exists()
    assert arquivo.stat().st_size > 0

print("NumPy: OK")
print("NetworkX: OK")
print("Imageio: OK")
print("Quatro estados: OK")
print("Matriz de transicao: OK")
print("Grafo direcionado: OK")
print("Dez transicoes simuladas: OK")
print("Onze quadros gerados: OK")
print("Animacao GIF: OK")
print("Imagem PNG: OK")
print("Historico CSV: OK")
print("Parametros JSON: OK")

print("\nAULA 05 EXECUTADA COM SUCESSO!")
