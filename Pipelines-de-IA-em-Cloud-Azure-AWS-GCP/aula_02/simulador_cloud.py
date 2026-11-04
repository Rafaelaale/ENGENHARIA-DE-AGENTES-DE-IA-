from decimal import Decimal

CUSTOS_EXEMPLO = {
    "AWS Bedrock": Decimal("0.002"),
    "Google Vertex AI": Decimal("0.0015"),
    "Azure AI": Decimal("0.0025")
}


def calcular_custo(requisicoes, custo_unitario):
    return Decimal(requisicoes) * custo_unitario


def verificar_orcamento(custo_total, orcamento):
    if custo_total > orcamento:
        return "ALERTA: custo acima do orcamento."
    return "Custo dentro do orcamento."


def main():
    print("=== SIMULADOR DIDATICO DE CUSTOS CLOUD ===")
    print("Valores ficticios em USD por requisicao.")

    provedores = list(CUSTOS_EXEMPLO.keys())

    for indice, nome in enumerate(provedores, start=1):
        print(f"{indice} - {nome}")

    try:
        opcao = int(input("Escolha o provedor: "))
        if opcao < 1 or opcao > len(provedores):
            print("Provedor invalido.")
            return

        requisicoes = int(input("Quantidade de requisicoes: "))
        orcamento = Decimal(input("Orcamento em USD: "))

        if requisicoes < 0 or orcamento < 0:
            print("Os valores nao podem ser negativos.")
            return

    except (ValueError, ArithmeticError):
        print("Entrada invalida.")
        return

    provedor = provedores[opcao - 1]
    custo_unitario = CUSTOS_EXEMPLO[provedor]
    custo_total = calcular_custo(requisicoes, custo_unitario)

    print(f"\nProvedor: {provedor}")
    print(f"Requisicoes: {requisicoes}")
    print(f"Custo estimado: USD {custo_total:.4f}")
    print(verificar_orcamento(custo_total, orcamento))


if __name__ == "__main__":
    main()
