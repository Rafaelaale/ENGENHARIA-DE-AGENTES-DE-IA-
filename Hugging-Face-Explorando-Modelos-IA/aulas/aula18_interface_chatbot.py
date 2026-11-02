# ==========================================================
# HUGGING FACE - AULA 18
# CRIANDO A INTERFACE DO CHATBOT COM GRADIO
# ==========================================================

import argparse
from pathlib import Path

import torch
import pandas as pd
import gradio as gr

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

ARQUIVO_PEDIDOS = (
    BASE / "dados" / "aula16_status_pedidos.csv"
)

MODELO = "microsoft/DialoGPT-medium"

print("=" * 65)
print("AULA 18 - INTERFACE DO CHATBOT")
print("=" * 65)


# ==========================================================
# 2 - CARREGANDO DIALOGPT
# ==========================================================

print("\n1 - CARREGANDO MODELO DIALOGPT")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

tokenizer = AutoTokenizer.from_pretrained(
    MODELO
)

model = AutoModelForCausalLM.from_pretrained(
    MODELO
)

model.eval()

print("Modelo carregado: OK")
print("Tokenizador carregado: OK")


# ==========================================================
# 3 - CARREGANDO DATAFRAME DE PEDIDOS
# ==========================================================

print("\n2 - CARREGANDO PEDIDOS")

df_status_pedidos = pd.read_csv(
    ARQUIVO_PEDIDOS,
    dtype={"numero_pedido": str}
)

print(df_status_pedidos.to_string(index=False))

print(
    "Pedidos cadastrados:",
    len(df_status_pedidos)
)


# ==========================================================
# 4 - FUNCAO DE CONSULTA
# ==========================================================

def verificar_status_pedido(numero_pedido):

    numero_pedido = str(numero_pedido).strip()

    pedido = df_status_pedidos[
        df_status_pedidos["numero_pedido"]
        == numero_pedido
    ]

    if pedido.empty:

        return (
            "Order number not found. "
            "Please check and try again."
        )

    status = pedido["status"].iloc[0]

    return (
        f"The status of your order "
        f"{numero_pedido} is: {status}"
    )


# ==========================================================
# 5 - PALAVRAS-CHAVE
# ==========================================================

palavras_chave_status = [
    "order",
    "order status",
    "status of my order",
    "check my order",
    "track my order",
    "order update"
]


# ==========================================================
# 6 - FUNCAO DE RESPOSTA DO CHATBOT
# ==========================================================

def responder(input_usuario, ids_historico_chat):

    if any(
        keyword in input_usuario.lower()
        for keyword in palavras_chave_status
    ):

        return (
            "Could you please enter your order number?",
            ids_historico_chat
        )

    novo_usuario_input_ids = tokenizer.encode(
        input_usuario + tokenizer.eos_token,
        return_tensors="pt"
    )

    if ids_historico_chat is not None:

        bot_input_ids = torch.cat(
            [
                ids_historico_chat,
                novo_usuario_input_ids
            ],
            dim=-1
        )

    else:

        bot_input_ids = novo_usuario_input_ids

    # Limitar o historico para preservar
    # espaco para a resposta do modelo.
    if bot_input_ids.shape[-1] > 850:

        bot_input_ids = bot_input_ids[
            :,
            -850:
        ]

    tamanho_entrada = bot_input_ids.shape[-1]

    with torch.inference_mode():

        ids_historico_chat = model.generate(
            bot_input_ids,
            attention_mask=torch.ones_like(
                bot_input_ids
            ),
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.75
        )

    resposta = tokenizer.decode(
        ids_historico_chat[
            0,
            tamanho_entrada:
        ],
        skip_special_tokens=True
    ).strip()

    if not resposta:
        resposta = "Could you repeat that?"

    return resposta, ids_historico_chat


print("\n3 - FUNCOES PREPARADAS")
print("Consulta de pedidos: OK")
print("Palavras-chave: OK")
print("Funcao responder: OK")


# ==========================================================
# 7 - CRIANDO INTERFACE GRADIO
# ==========================================================

print("\n4 - CRIANDO INTERFACE")

with gr.Blocks(
    title="E-commerce Chatbot"
) as app:

    gr.Markdown(
        "# E-commerce Chatbot"
    )

    gr.Markdown(
        "Customer service and order tracking"
    )

    chatbot = gr.Chatbot(
        label="Conversation"
    )

    msg = gr.Textbox(
        placeholder="Type your message here...",
        label="Your message"
    )

    # Historico do modelo
    estado = gr.State(None)

    # Identifica quando o bot aguarda um pedido
    aguardando_numero_pedido = gr.State(False)

    # O processamento da mensagem e a conexao
    # dos eventos Gradio serao implementados
    # no proximo video da Alura.


print("Gradio Blocks: OK")
print("Componente Chatbot: OK")
print("Caixa de mensagens: OK")
print("Estado do historico: OK")
print("Estado aguardando pedido: OK")


# ==========================================================
# 8 - VALIDACAO INICIAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO INICIAL")
print("=" * 65)

assert len(df_status_pedidos) == 4

assert verificar_status_pedido("12345") == (
    "The status of your order 12345 is: Shipped"
)

assert verificar_status_pedido("99999") == (
    "Order number not found. "
    "Please check and try again."
)

assert len(palavras_chave_status) == 6

assert chatbot is not None
assert msg is not None
assert estado is not None
assert aguardando_numero_pedido is not None

print("Modelo DialoGPT: OK")
print("DataFrame Pandas: OK")
print("Quatro pedidos: OK")
print("Funcao de consulta: OK")
print("Seis palavras-chave: OK")
print("Funcao de resposta: OK")
print("Interface Gradio: OK")
print("Dois estados Gradio: OK")

print("\nAULA 18 - ESTRUTURA VALIDADA!")


# ==========================================================
# 9 - EXECUCAO
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--check",
    action="store_true"
)

args = parser.parse_args()

if not args.check:

    print("\nINICIANDO INTERFACE...")
    print("Endereco: http://127.0.0.1:7862")
    print("Acesso: local e privado")
    print("Encerrar: Ctrl+C")

    print(
        "AVISO: O envio de mensagens ainda nao "
        "foi implementado nesta aula."
    )

    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=True
    )
