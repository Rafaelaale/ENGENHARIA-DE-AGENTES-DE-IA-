# ==========================================================
# AULA 17 - CHATBOT INTEGRADO AO E-COMMERCE
# DIALOGPT + CONSULTA DE PEDIDOS COM PANDAS
# ==========================================================

from pathlib import Path
import argparse
import json
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path(__file__).resolve().parent.parent
ARQUIVO_PEDIDOS = BASE / "dados" / "aula16_status_pedidos.csv"
ARQUIVO_RESULTADO = BASE / "resultados" / "aula17_historico_chatbot_pedidos.json"

MODELO = "microsoft/DialoGPT-medium"

print("=" * 60)
print("AULA 17 - CHATBOT DE PEDIDOS")
print("=" * 60)

# ----------------------------------------------------------
# 1 - DATAFRAME DA AULA 16
# ----------------------------------------------------------

dados = pd.read_csv(
    ARQUIVO_PEDIDOS,
    dtype={"numero_pedido": str}
)

print("\n1 - PEDIDOS CARREGADOS")
print(dados.to_string(index=False))


def verificar_status_pedido(numero_pedido):

    pedido = dados.loc[
        dados["numero_pedido"] == str(numero_pedido).strip(),
        "status"
    ]

    if pedido.empty:
        return "Order number not found. Please check and try again."

    return (
        f"The status of your order {numero_pedido} "
        f"is: {pedido.iloc[0]}"
    )


# ----------------------------------------------------------
# 2 - PALAVRAS-CHAVE
# ----------------------------------------------------------

palavras_chave_status = [
    "order",
    "order status",
    "status of my order",
    "check my order",
    "track my order",
    "order update"
]

# ----------------------------------------------------------
# 3 - CARREGAR DIALOGPT
# ----------------------------------------------------------

print("\n2 - CARREGANDO DIALOGPT")

torch.set_num_threads(min(4, torch.get_num_threads()))

tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModelForCausalLM.from_pretrained(MODELO)

model.eval()

print("Modelo carregado: OK")

# ----------------------------------------------------------
# 4 - CONVERSACAO COM ESTADO
# ----------------------------------------------------------

ids_historico_chat = None
aguardando_numero = False
historico = []


def responder(mensagem):

    global ids_historico_chat
    global aguardando_numero

    mensagem = mensagem.strip()

    if mensagem.casefold() in ("exit", "quit", "stop"):
        return "Goodbye!", True

    if not mensagem:
        return "Please enter a message.", False

    if aguardando_numero:

        aguardando_numero = False
        resposta = verificar_status_pedido(mensagem)
        return resposta, False

    if any(
        palavra in mensagem.casefold()
        for palavra in palavras_chave_status
    ):

        aguardando_numero = True

        return (
            "Could you please enter your order number?",
            False
        )

    # Conversa normal com IA
    entrada = tokenizer.encode(
        mensagem + tokenizer.eos_token,
        return_tensors="pt"
    )

    if ids_historico_chat is not None:

        entrada = torch.cat(
            [ids_historico_chat, entrada],
            dim=-1
        )

    if entrada.shape[-1] > 850:
        entrada = entrada[:, -850:]

    tamanho = entrada.shape[-1]

    with torch.inference_mode():

        ids_historico_chat = model.generate(
            entrada,
            attention_mask=torch.ones_like(entrada),
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.75
        )

    resposta = tokenizer.decode(
        ids_historico_chat[0, tamanho:],
        skip_special_tokens=True
    ).strip()

    return resposta or "Could you repeat that?", False


# ----------------------------------------------------------
# 5 - TESTE AUTOMATICO OU CONVERSA MANUAL
# ----------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--demo", action="store_true")
args = parser.parse_args()

if args.demo:

    mensagens = [
        "Hello!",
        "I would like to check the status of my order",
        "12345",
        "Check my order",
        "99999",
        "stop"
    ]

else:

    mensagens = None


def registrar(mensagem, resposta):

    historico.append({
        "usuario": mensagem,
        "bot": resposta
    })


print("\n3 - INICIANDO CONVERSA")

if args.demo:

    for mensagem in mensagens:

        print("\nYou:", mensagem)

        resposta, encerrar = responder(mensagem)

        print("Bot:", resposta)

        registrar(mensagem, resposta)

        if encerrar:
            break

else:

    while True:

        try:
            mensagem = input("\nYou: ")

        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

        resposta, encerrar = responder(mensagem)

        print("Bot:", resposta)

        registrar(mensagem, resposta)

        if encerrar:
            break


# ----------------------------------------------------------
# 6 - SALVAR HISTORICO
# ----------------------------------------------------------

ARQUIVO_RESULTADO.parent.mkdir(
    parents=True,
    exist_ok=True
)

with ARQUIVO_RESULTADO.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        {
            "modelo": MODELO,
            "historico": historico
        },
        arquivo,
        ensure_ascii=False,
        indent=4
    )

# ----------------------------------------------------------
# 7 - VALIDACAO
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDACAO FINAL")
print("=" * 60)

assert ARQUIVO_RESULTADO.exists()

assert verificar_status_pedido("12345") == (
    "The status of your order 12345 is: Shipped"
)

assert verificar_status_pedido("99999") == (
    "Order number not found. Please check and try again."
)

if args.demo:

    respostas = [item["bot"] for item in historico]

    assert "The status of your order 12345 is: Shipped" in respostas
    assert "Order number not found. Please check and try again." in respostas
    assert respostas[-1] == "Goodbye!"

print("Modelo DialoGPT: OK")
print("DataFrame Pandas: OK")
print("Deteccao de pedidos: OK")
print("Consulta do pedido 12345: OK")
print("Pedido inexistente: OK")
print("Encerramento: OK")
print("Historico JSON: OK")

print("\nAULA 17 EXECUTADA COM SUCESSO!")
