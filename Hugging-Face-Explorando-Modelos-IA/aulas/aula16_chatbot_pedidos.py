# ==========================================================
# HUGGING FACE - AULA 16
# CRIANDO O CHATBOT PARA CONSULTA DE PEDIDOS
# ==========================================================

from pathlib import Path
import json
import sys

import pandas as pd


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("AULA 16 - CHATBOT DE PEDIDOS DO E-COMMERCE")
print("=" * 65)


# ==========================================================
# 2 - CRIAR DICIONARIO DE PEDIDOS
# ==========================================================

print("\n1 - CRIANDO BASE DE PEDIDOS")

dados_pedidos = {
    "numero_pedido": [
        "12345",
        "67890",
        "11121",
        "22232"
    ],
    "status": [
        "Shipped",
        "Processing",
        "Delivered",
        "Cancelled"
    ]
}


# ==========================================================
# 3 - CONVERTER PARA DATAFRAME PANDAS
# ==========================================================

df_status_pedidos = pd.DataFrame(
    dados_pedidos
)

# Manter os numeros como texto
df_status_pedidos["numero_pedido"] = (
    df_status_pedidos["numero_pedido"]
    .astype(str)
)

print("\nDATAFRAME DE PEDIDOS:")
print(
    df_status_pedidos.to_string(
        index=False
    )
)

assert len(df_status_pedidos) == 4

assert list(df_status_pedidos.columns) == [
    "numero_pedido",
    "status"
]

arquivo_pedidos = (
    DADOS / "aula16_status_pedidos.csv"
)

df_status_pedidos.to_csv(
    arquivo_pedidos,
    index=False,
    encoding="utf-8-sig"
)

print("\nBase de pedidos salva:")
print(arquivo_pedidos)


# ==========================================================
# 4 - FUNCAO VERIFICAR STATUS DO PEDIDO
# ==========================================================

print("\n2 - CRIANDO FUNCAO DE CONSULTA")

def verificar_status_pedido(numero_pedido):

    numero_pedido = str(numero_pedido).strip()

    # Filtrar o DataFrame pelo numero
    pedido = df_status_pedidos[
        df_status_pedidos["numero_pedido"]
        == numero_pedido
    ]

    # Tratar pedido inexistente
    if pedido.empty:

        return (
            "Order number not found. "
            "Please check and try again."
        )

    # Recuperar status correspondente
    status = pedido["status"].iloc[0]

    return (
        f"The status of your order "
        f"{numero_pedido} is: {status}"
    )


# ==========================================================
# 5 - TESTAR TODOS OS PEDIDOS
# ==========================================================

print("\n3 - TESTANDO CONSULTAS")

testes = [
    ("12345", "Shipped"),
    ("67890", "Processing"),
    ("11121", "Delivered"),
    ("22232", "Cancelled")
]

registros = []

for numero, status_esperado in testes:

    resposta = verificar_status_pedido(
        numero
    )

    print("\nPedido:", numero)
    print("Resposta:", resposta)

    assert status_esperado in resposta

    registros.append({
        "numero_pedido": numero,
        "status_esperado": status_esperado,
        "resposta": resposta,
        "correto": True
    })


# ==========================================================
# 6 - TESTAR PEDIDO INEXISTENTE
# ==========================================================

print("\n4 - TESTANDO PEDIDO INEXISTENTE")

numero_inexistente = "99999"

resposta_inexistente = (
    verificar_status_pedido(
        numero_inexistente
    )
)

mensagem_esperada = (
    "Order number not found. "
    "Please check and try again."
)

assert resposta_inexistente == mensagem_esperada

print("Pedido:", numero_inexistente)
print("Resposta:", resposta_inexistente)

registros.append({
    "numero_pedido": numero_inexistente,
    "status_esperado": "Not found",
    "resposta": resposta_inexistente,
    "correto": True
})


# ==========================================================
# 7 - DEFINIR PALAVRAS-CHAVE
# ==========================================================

print("\n5 - DEFININDO PALAVRAS-CHAVE")

palavras_chave_status = [
    "order",
    "order status",
    "status of my order",
    "check my order",
    "track my order",
    "order update"
]

for palavra in palavras_chave_status:
    print("-", palavra)

assert len(palavras_chave_status) == 6


# ==========================================================
# 8 - SALVAR RESULTADOS DOS TESTES
# ==========================================================

arquivo_resultados = (
    RESULTADOS
    / "aula16_testes_status_pedidos.json"
)

with arquivo_resultados.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        {
            "total_pedidos": len(df_status_pedidos),
            "palavras_chave": palavras_chave_status,
            "testes": registros
        },
        arquivo,
        ensure_ascii=False,
        indent=4
    )

print("\nResultados salvos:")
print(arquivo_resultados)


# ==========================================================
# 9 - PREPARAR HISTORICO DO CHAT
# ==========================================================

print("\n6 - INICIANDO LOOP DE CONVERSA")

ids_historico_chat = None

print("Digite uma mensagem em ingles.")
print("Para encerrar: exit, quit ou stop.")
print(
    "A integracao da consulta de pedidos "
    "ao chat sera feita na proxima aula."
)


# ==========================================================
# 10 - LOOP DE CONVERSA
# ==========================================================

while True:

    try:
        input_usuario = input(
            "\nYou: "
        ).strip()

    except (EOFError, KeyboardInterrupt):
        print("\nBot: Goodbye!")
        break

    # Mostrar entradas enviadas automaticamente
    if not sys.stdin.isatty():
        print(input_usuario)

    # Verificar comando de encerramento
    if input_usuario.lower() in [
        "exit",
        "quit",
        "stop"
    ]:

        print("Bot: Goodbye!")
        break

    if not input_usuario:
        print("Bot: Please enter a message.")
        continue

    # A transcricao termina antes de implementar
    # as respostas do chatbot neste loop.
    print(
        "Bot: Message received. "
        "Order status integration "
        "will be added in the next lesson."
    )


# ==========================================================
# 11 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert arquivo_pedidos.exists()
assert arquivo_resultados.exists()

assert len(registros) == 5

assert all(
    registro["correto"]
    for registro in registros
)

print("DataFrame Pandas: OK")
print("Quatro pedidos cadastrados: OK")
print("Consulta por numero: OK")
print("Pedido inexistente tratado: OK")
print("Seis palavras-chave: OK")
print("Loop de conversa: OK")
print("Encerramento do chat: OK")
print("CSV salvo: OK")
print("JSON salvo: OK")

print("\nTESTES DE CONSULTA: 5/5")
print("AULA 16 EXECUTADA COM SUCESSO!")
