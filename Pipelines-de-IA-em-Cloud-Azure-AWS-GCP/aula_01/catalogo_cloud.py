import json
from pathlib import Path

SERVICOS = {
    "AWS": {
        "machine_learning": "Amazon SageMaker",
        "ia_generativa": "Amazon Bedrock",
        "agentes": "Amazon Bedrock"
    },
    "Google Cloud": {
        "machine_learning": "Vertex AI",
        "ia_generativa": "Vertex AI / Gemini",
        "agentes": "Vertex AI"
    },
    "Azure": {
        "machine_learning": "Azure Machine Learning",
        "ia_generativa": "Azure OpenAI",
        "agentes": "Servicos de IA do Azure"
    }
}

def recomendar_servicos(tipo_aplicacao):
    recomendacoes = {}

    for provedor, servicos in SERVICOS.items():
        if tipo_aplicacao in servicos:
            recomendacoes[provedor] = servicos[tipo_aplicacao]

    return recomendacoes


def salvar_resultado(tipo_aplicacao, recomendacoes):
    resultado = {
        "tipo_aplicacao": tipo_aplicacao,
        "recomendacoes": recomendacoes
    }

    destino = Path(__file__).parent / "resultado.json"

    with destino.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=4, ensure_ascii=False)

    return destino


def main():
    print("=== LABORATORIO DE SERVICOS CLOUD PARA IA ===")
    print("1 - Machine Learning")
    print("2 - IA Generativa")
    print("3 - Agentes de IA")

    opcoes = {
        "1": "machine_learning",
        "2": "ia_generativa",
        "3": "agentes"
    }

    escolha = input("Escolha uma opcao: ").strip()

    if escolha not in opcoes:
        print("Opcao invalida.")
        return

    tipo = opcoes[escolha]
    recomendacoes = recomendar_servicos(tipo)

    print("\nServicos relacionados:")

    for provedor, servico in recomendacoes.items():
        print(f"{provedor}: {servico}")

    destino = salvar_resultado(tipo, recomendacoes)
    print(f"\nResultado salvo em: {destino}")


if __name__ == "__main__":
    main()
