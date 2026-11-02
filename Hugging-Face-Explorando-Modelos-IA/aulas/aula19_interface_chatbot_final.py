# ==========================================================
# HUGGING FACE - AULA 19
# FINALIZANDO A INTERFACE DO CHATBOT
#
# DIALOGPT + PANDAS + GRADIO
# ==========================================================

from pathlib import Path
import argparse
import json

import gradio as gr
import pandas as pd
import torch

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

ARQUIVO_TESTES = (
    BASE / "resultados" / "aula19_validacao.json"
)

MODELO = "microsoft/DialoGPT-medium"

MENSAGEM_PEDIDO = (
    "Could you please enter your order number?"
)

print("=" * 65)
print("AULA 19 - INTERFACE FINAL DO CHATBOT")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR PEDIDOS
# ==========================================================

print("\n1 - CARREGANDO DATAFRAME DE PEDIDOS")

if not ARQUIVO_PEDIDOS.exists():
    raise FileNotFoundError(
        f"Arquivo nao encontrado: {ARQUIVO_PEDIDOS}"
    )

df_status_pedidos = pd.read_csv(
    ARQUIVO_PEDIDOS,
    dtype={"numero_pedido": str}
)

print(df_status_pedidos.to_string(index=False))

assert len(df_status_pedidos) == 4

assert {
    "numero_pedido",
    "status"
}.issubset(df_status_pedidos.columns)


# ==========================================================
# 3 - CARREGAR MODELO DIALOGPT
# ==========================================================

print("\n2 - CARREGANDO DIALOGPT")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

tokenizer = AutoTokenizer.from_pretrained(
    MODELO
)

model = AutoModelForCausalLM.from_pretrained(
    MODELO
)

model.to("cpu")
model.eval()

print("Modelo carregado: OK")
print("Tokenizador carregado: OK")


# ==========================================================
# 4 - CONSULTA DE PEDIDOS
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
# 6 - FUNCAO RESPONDER COM DIALOGPT
# ==========================================================

def responder(input_usuario, ids_historico_chat):

    mensagem = input_usuario.strip()

    if any(
        palavra in mensagem.lower()
        for palavra in palavras_chave_status
    ):

        return (
            MENSAGEM_PEDIDO,
            ids_historico_chat
        )

    # Transformar mensagem em tokens
    novo_usuario_input_ids = tokenizer.encode(
        mensagem + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Acrescentar historico da conversa
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

    # Preservar espaco para a resposta
    if bot_input_ids.shape[-1] > 850:

        bot_input_ids = bot_input_ids[
            :,
            -850:
        ]

    tamanho_entrada = bot_input_ids.shape[-1]

    # Gerar resposta com IA
    with torch.inference_mode():

        ids_historico_chat = model.generate(
            input_ids=bot_input_ids,
            attention_mask=torch.ones_like(
                bot_input_ids
            ),
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.75
        )

    novos_tokens = ids_historico_chat[
        0,
        tamanho_entrada:
    ]

    resposta = tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True
    ).strip()

    if not resposta:
        resposta = "Could you repeat that?"

    return resposta, ids_historico_chat


# ==========================================================
# 7 - PROCESSAR ENTRADA DO USUARIO
# ==========================================================

def processar_entrada(
    input_usuario,
    historico,
    ids_historico_chat,
    aguardando_numero_pedido
):

    mensagem = (input_usuario or "").strip()

    # Manter o historico visual do Gradio
    historico = list(historico or [])

    if not mensagem:

        return (
            historico,
            ids_historico_chat,
            aguardando_numero_pedido,
            ""
        )

    # Encerrar e reiniciar estado da conversa
    if mensagem.lower() in (
        "exit",
        "quit",
        "stop"
    ):

        historico.append(
            (mensagem, "Goodbye!")
        )

        return historico, None, False, ""

    # Se o chatbot pediu o numero do pedido
    if aguardando_numero_pedido:

        resposta = verificar_status_pedido(
            mensagem
        )

        aguardando_numero_pedido = False

    else:

        # Conversa normal com IA ou identificacao
        # de uma solicitacao de pedido
        resposta, ids_historico_chat = responder(
            mensagem,
            ids_historico_chat
        )

        if resposta == MENSAGEM_PEDIDO:

            aguardando_numero_pedido = True

    # Adicionar mensagens ao chatbot visual
    historico.append(
        (mensagem, resposta)
    )

    # Retornar:
    # 1 - Historico exibido na tela
    # 2 - Historico interno do DialoGPT
    # 3 - Estado aguardando numero do pedido
    # 4 - Caixa de texto vazia
    return (
        historico,
        ids_historico_chat,
        aguardando_numero_pedido,
        ""
    )


# ==========================================================
# 8 - TESTES AUTOMATICOS
# ==========================================================

print("\n3 - EXECUTANDO TESTES")

assert verificar_status_pedido("12345") == (
    "The status of your order 12345 is: Shipped"
)

assert verificar_status_pedido("67890") == (
    "The status of your order 67890 is: Processing"
)

assert verificar_status_pedido("11121") == (
    "The status of your order 11121 is: Delivered"
)

assert verificar_status_pedido("22232") == (
    "The status of your order 22232 is: Cancelled"
)

assert verificar_status_pedido("99999") == (
    "Order number not found. "
    "Please check and try again."
)

# Simular conversa sem abrir o navegador
historico = []
estado = None
aguardando = False

# Teste 1: identificar pedido
historico, estado, aguardando, caixa = (
    processar_entrada(
        "Check my order",
        historico,
        estado,
        aguardando
    )
)

assert aguardando is True
assert historico[-1][1] == MENSAGEM_PEDIDO
assert caixa == ""

# Teste 2: consultar pedido existente
historico, estado, aguardando, caixa = (
    processar_entrada(
        "12345",
        historico,
        estado,
        aguardando
    )
)

assert aguardando is False

assert historico[-1][1] == (
    "The status of your order 12345 is: Shipped"
)

# Teste 3: pedido inexistente
historico, estado, aguardando, caixa = (
    processar_entrada(
        "Track my order",
        historico,
        estado,
        aguardando
    )
)

assert aguardando is True

historico, estado, aguardando, caixa = (
    processar_entrada(
        "99999",
        historico,
        estado,
        aguardando
    )
)

assert historico[-1][1] == (
    "Order number not found. "
    "Please check and try again."
)

assert aguardando is False

# Teste 4: encerramento
historico, estado, aguardando, caixa = (
    processar_entrada(
        "stop",
        historico,
        estado,
        aguardando
    )
)

assert historico[-1][1] == "Goodbye!"
assert estado is None
assert aguardando is False

print("Quatro status de pedidos: OK")
print("Pedido inexistente: OK")
print("Deteccao de palavra-chave: OK")
print("Solicitacao de numero: OK")
print("Consulta pelo chatbot: OK")
print("Estado de espera: OK")
print("Historico visual: OK")
print("Encerramento: OK")

ARQUIVO_TESTES.parent.mkdir(
    parents=True,
    exist_ok=True
)

with ARQUIVO_TESTES.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        {
            "modelo": MODELO,
            "testes": "OK",
            "historico_teste": [
                {
                    "usuario": usuario,
                    "bot": bot
                }
                for usuario, bot in historico
            ]
        },
        arquivo,
        ensure_ascii=False,
        indent=4
    )

print("JSON de validacao salvo: OK")


# ==========================================================
# 9 - CRIANDO A INTERFACE GRADIO
# ==========================================================

print("\n4 - CRIANDO INTERFACE GRADIO")

with gr.Blocks(
    title="E-commerce Chatbot"
) as app:

    gr.Markdown(
        "# E-commerce Chatbot"
    )

    gr.Markdown(
        "Talk to the chatbot or check your order status."
    )

    chatbot = gr.Chatbot(
        label="Customer Support",
        height=420
    )

    msg = gr.Textbox(
        placeholder="Type your message here...",
        label="Your message",
        lines=1
    )

    # Historico de tokens do DialoGPT
    estado = gr.State(None)

    # Estado de espera pelo numero do pedido
    aguardando_numero_pedido = gr.State(False)

    # Conectar Enter ao processamento
    msg.submit(
        fn=processar_entrada,
        inputs=[
            msg,
            chatbot,
            estado,
            aguardando_numero_pedido
        ],
        outputs=[
            chatbot,
            estado,
            aguardando_numero_pedido,
            msg
        ]
    )

print("Gradio Blocks: OK")
print("Componente Chatbot: OK")
print("Textbox: OK")
print("Estados: OK")
print("Evento de envio: OK")


# ==========================================================
# 10 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert ARQUIVO_TESTES.exists()
assert app is not None
assert chatbot is not None
assert msg is not None

print("Modelo DialoGPT: OK")
print("DataFrame Pandas: OK")
print("Consultas de pedidos: OK")
print("Processamento de mensagens: OK")
print("Historico da conversa: OK")
print("Estados Gradio: OK")
print("Evento msg.submit: OK")
print("Interface criada: OK")

print("\nVALIDACAO AUTOMATICA CONCLUIDA!")


# ==========================================================
# 11 - INICIAR APLICACAO
# ==========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--check",
    action="store_true"
)

args = parser.parse_args()

if not args.check:

    print("\nINICIANDO CHATBOT INTERATIVO...")
    print("Endereco: http://127.0.0.1:7862")
    print("Acesso: local e privado")
    print("Para encerrar: Ctrl+C")

    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=True
    )
