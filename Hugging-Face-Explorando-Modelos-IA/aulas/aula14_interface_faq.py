# ==========================================================
# HUGGING FACE - AULA 14
# INTERFACE GRADIO PARA FAQ DO E-COMMERCE
# ==========================================================

from pathlib import Path
import json

import gradio as gr
import torch
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

ARQUIVO_CONTEXTOS = (
    BASE / "dados" / "aula13_contextos_faq.json"
)

MODELO = "deepset/roberta-base-squad2"

print("=" * 65)
print("AULA 14 - FAQ INTERATIVO COM GRADIO")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR CONTEXTOS DA AULA 13
# ==========================================================

print("\n1 - CARREGANDO PERGUNTAS DO FAQ")

if not ARQUIVO_CONTEXTOS.exists():
    raise FileNotFoundError(
        "Arquivo de contextos da Aula 13 "
        "nao encontrado."
    )

with ARQUIVO_CONTEXTOS.open(
    "r",
    encoding="utf-8"
) as arquivo:

    contextos = json.load(arquivo)

assert isinstance(contextos, dict)
assert len(contextos) == 9

print("Perguntas carregadas:", len(contextos))

for numero, pergunta in enumerate(
    contextos.keys(),
    start=1
):
    print(f"{numero}. {pergunta}")


# ==========================================================
# 3 - CARREGAR MODELO QUESTION ANSWERING
# ==========================================================

print("\n2 - CARREGANDO MODELO ROBERTA")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

QA_modelo = pipeline(
    "question-answering",
    model=MODELO,
    tokenizer=MODELO,
    device=-1
)

print("Modelo carregado: OK")


# ==========================================================
# 4 - FUNCAO RESPONDENDO_FAQ
# ==========================================================

def respondendo_faq(pergunta):

    if not pergunta:
        raise gr.Error(
            "Selecione uma pergunta."
        )

    if pergunta not in contextos:
        raise gr.Error(
            "Pergunta nao encontrada no FAQ."
        )

    contexto = contextos[pergunta]

    print("\nPERGUNTA SELECIONADA:")
    print(pergunta)

    with torch.inference_mode():

        resultado = QA_modelo(
            question=pergunta,
            context=contexto
        )

    resposta = resultado["answer"]

    print("\nRESPOSTA GERADA:")
    print(resposta)

    print(
        "Score:",
        f"{resultado['score']:.4f}"
    )

    return resposta


# ==========================================================
# 5 - CRIAR INTERFACE GRADIO
# ==========================================================

print("\n3 - CRIANDO INTERFACE")

perguntas = list(contextos.keys())

app = gr.Interface(

    fn=respondendo_faq,

    inputs=gr.Dropdown(
        choices=perguntas,
        value=perguntas[0],
        label="Select your question",
        interactive=True
    ),

    outputs=gr.Textbox(
        label="Answer",
        lines=3,
        interactive=False
    ),

    title="E-commerce FAQ",

    description=(
        "Select a question to get "
        "an answer from our FAQ."
    ),

    allow_flagging="never"
)

print("Interface Gradio criada: OK")
print("Dropdown de perguntas: OK")
print("Caixa de respostas: OK")


# ==========================================================
# 6 - VALIDACAO INICIAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO INICIAL")
print("=" * 65)

assert len(perguntas) == 9

assert all(
    isinstance(pergunta, str)
    and isinstance(contextos[pergunta], str)
    for pergunta in perguntas
)

print("Modelo RoBERTa: OK")
print("Contextos da Aula 13: OK")
print("Nove perguntas: OK")
print("Funcao respondendo_faq: OK")
print("Interface Gradio: OK")
print("Acesso: local e privado")


# ==========================================================
# 7 - INICIAR APLICACAO
# ==========================================================

if __name__ == "__main__":

    print("\nINICIANDO FAQ INTERATIVO...")
    print("Endereco: http://127.0.0.1:7861")
    print("Para encerrar: Ctrl+C")

    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        show_error=True
    )
