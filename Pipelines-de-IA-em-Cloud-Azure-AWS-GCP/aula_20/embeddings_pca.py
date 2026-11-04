import json
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


BASE = Path(__file__).resolve().parent

MODELO = "nomic-embed-text:latest"
URL = "http://127.0.0.1:11434/api/embed"

FRASES = [
    {
        "id": "A1",
        "categoria": "Animais",
        "texto": "O cachorro está brincando no parque."
    },
    {
        "id": "A2",
        "categoria": "Animais",
        "texto": "O cão corre feliz na grama."
    },
    {
        "id": "T1",
        "categoria": "Tecnologia",
        "texto": "O computador está processando dados."
    },
    {
        "id": "T2",
        "categoria": "Tecnologia",
        "texto": "O laptop foi usado para escrever o relatório."
    },
    {
        "id": "C1",
        "categoria": "Café",
        "texto": "Adoro tomar café pela manhã."
    },
    {
        "id": "C2",
        "categoria": "Café",
        "texto": "Bebo café todo dia antes de trabalhar."
    }
]


def gerar_embeddings(frases):
    payload = {
        "model": MODELO,
        "input": [item["texto"] for item in frases]
    }

    requisicao = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(requisicao, timeout=180) as resposta:
        dados = json.loads(resposta.read().decode("utf-8"))

    vetores = np.asarray(dados["embeddings"], dtype=np.float64)

    if vetores.shape[0] != len(frases):
        raise ValueError("Quantidade de embeddings incorreta.")

    if not np.isfinite(vetores).all():
        raise ValueError("Embeddings contem valores invalidos.")

    return vetores


def main():
    print("=== AULA 20 - EMBEDDINGS E PCA ===")

    vetores = gerar_embeddings(FRASES)

    print("\n=== EMBEDDINGS ===")
    print("Frases:", len(FRASES))
    print("Dimensoes:", vetores.shape[1])
    print("Matriz:", vetores.shape)

    similaridades = cosine_similarity(vetores)

    print("\n=== SIMILARIDADE ENTRE FRASES DO MESMO ASSUNTO ===")

    pares = [
        (0, 1, "Animais"),
        (2, 3, "Tecnologia"),
        (4, 5, "Cafe")
    ]

    for i, j, categoria in pares:
        print(
            f"{categoria}: "
            f"{similaridades[i, j]:.6f}"
        )

    print("\n=== VIZINHO MAIS PROXIMO ===")

    vizinhos = []

    for i, frase in enumerate(FRASES):
        scores = similaridades[i].copy()
        scores[i] = -np.inf

        j = int(np.argmax(scores))

        resultado = {
            "frase": frase["id"],
            "vizinho": FRASES[j]["id"],
            "mesma_categoria": (
                frase["categoria"] == FRASES[j]["categoria"]
            ),
            "similaridade": round(float(scores[j]), 6)
        }

        vizinhos.append(resultado)

        print(
            f"{frase['id']} -> {FRASES[j]['id']} "
            f"| score={scores[j]:.6f} "
            f"| mesma categoria={resultado['mesma_categoria']}"
        )

    pca = PCA(n_components=2)
    pontos = pca.fit_transform(vetores)

    variancia = pca.explained_variance_ratio_

    print("\n=== PCA ===")
    print("Dimensoes originais:", vetores.shape[1])
    print("Dimensoes reduzidas:", pontos.shape[1])
    print(
        "Variancia explicada:",
        f"{variancia.sum() * 100:.2f}%"
    )

    categorias = {
        "Animais": "tab:blue",
        "Tecnologia": "tab:orange",
        "Café": "tab:green"
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for categoria, cor in categorias.items():
        indices = [
            i for i, item in enumerate(FRASES)
            if item["categoria"] == categoria
        ]

        ax.scatter(
            pontos[indices, 0],
            pontos[indices, 1],
            label=categoria,
            color=cor,
            s=110
        )

        for i in indices:
            ax.annotate(
                FRASES[i]["id"],
                (pontos[i, 0], pontos[i, 1]),
                xytext=(7, 7),
                textcoords="offset points"
            )

    ax.set_title("Aula 20 - Embeddings projetados com PCA")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    imagem = BASE / "embeddings_pca.png"
    fig.savefig(imagem, dpi=160)
    plt.close(fig)

    relatorio = {
        "aula": 20,
        "modelo": MODELO,
        "frases": FRASES,
        "dimensoes": int(vetores.shape[1]),
        "similaridade_cosseno": similaridades.tolist(),
        "vizinhos": vizinhos,
        "pca": {
            "pontos": pontos.tolist(),
            "variancia_explicada": variancia.tolist(),
            "variancia_total": float(variancia.sum())
        }
    }

    arquivo = BASE / "resultado_embeddings.json"

    arquivo.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n=== RESULTADOS ===")
    print("Imagem:", imagem)
    print("Relatorio:", arquivo)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()