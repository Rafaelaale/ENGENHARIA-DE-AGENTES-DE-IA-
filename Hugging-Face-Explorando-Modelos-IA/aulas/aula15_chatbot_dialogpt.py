# ==========================================================
# HUGGING FACE - AULA 15
# CHATBOT COM MICROSOFT DIALOGPT-MEDIUM
# ==========================================================

from pathlib import Path
from datetime import datetime
import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "microsoft/DialoGPT-medium"

MAX_INTERACOES = 5
MAX_NOVOS_TOKENS = 60
MAX_HISTORICO_TOKENS = 900

ARQUIVO = (
    RESULTADOS
    / "aula15_historico_chatbot.json"
)

print("=" * 65)
print("AULA 15 - CHATBOT COM DIALOGPT")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR TOKENIZADOR E MODELO
# ==========================================================

print("\n1 - CARREGANDO MODELO")

print("Modelo:", MODELO)
print("Dispositivo: CPU")

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

print("\nMODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 3 - HISTORICO DA CONVERSA
# ==========================================================

chat_history_ids = None
historico_conversa = []

print("\n2 - INICIANDO CHATBOT")
print("Idioma recomendado: ingles")
print("Total de interacoes:", MAX_INTERACOES)
print("Digite 'sair' para encerrar antecipadamente.")


# ==========================================================
# 4 - CONVERSAR POR CINCO INTERACOES
# ==========================================================

for step in range(MAX_INTERACOES):

    print("\n" + "-" * 60)

    print(
        f"INTERACAO {step + 1}/{MAX_INTERACOES}"
    )

    try:
        mensagem_usuario = input(
            ">> User: "
        ).strip()

    except (KeyboardInterrupt, EOFError):
        print("\nConversa interrompida.")
        break

    if mensagem_usuario.lower() in (
        "sair",
        "exit",
        "quit"
    ):
        print("Encerrando conversa.")
        break

    if not mensagem_usuario:
        print("Mensagem vazia. Tente novamente.")
        continue

    # Transformar texto em tokens.
    # Acrescentar EOS ao final da mensagem.
    new_user_input_ids = tokenizer.encode(
        mensagem_usuario + tokenizer.eos_token,
        return_tensors="pt"
    )

    # Recuperar historico das mensagens anteriores.
    if chat_history_ids is not None:

        bot_input_ids = torch.cat(
            [
                chat_history_ids,
                new_user_input_ids
            ],
            dim=-1
        )

    else:

        bot_input_ids = new_user_input_ids

    # Preservar o limite de contexto do modelo.
    if bot_input_ids.shape[-1] > MAX_HISTORICO_TOKENS:

        bot_input_ids = bot_input_ids[
            :,
            -MAX_HISTORICO_TOKENS:
        ]

    tamanho_entrada = bot_input_ids.shape[-1]

    # Mascara de atencao para os tokens de entrada.
    attention_mask = torch.ones_like(
        bot_input_ids
    )

    # Gerar resposta do chatbot.
    with torch.inference_mode():

        chat_history_ids = model.generate(
            input_ids=bot_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NOVOS_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.75
        )

    # Extrair somente a resposta nova.
    novos_tokens = chat_history_ids[
        :,
        tamanho_entrada:
    ][0]

    resposta = tokenizer.decode(
        novos_tokens,
        skip_special_tokens=True
    ).strip()

    if not resposta:
        resposta = "[O modelo nao gerou texto nesta rodada]"

    print("\nDialoGPT:", resposta)

    # Armazenar o historico legivel.
    historico_conversa.append({

        "interacao": step + 1,

        "usuario": mensagem_usuario,

        "chatbot": resposta

    })


# ==========================================================
# 5 - SALVAR HISTORICO DA CONVERSA
# ==========================================================

print("\n" + "=" * 65)
print("3 - SALVANDO HISTORICO")
print("=" * 65)

dados = {

    "modelo": MODELO,

    "data_execucao": datetime.now().isoformat(),

    "total_interacoes": len(historico_conversa),

    "historico": historico_conversa

}

with ARQUIVO.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        dados,
        arquivo,
        ensure_ascii=False,
        indent=4
    )

print("\nArquivo salvo:")
print(ARQUIVO)


# ==========================================================
# 6 - VALIDACAO FINAL
# ==========================================================

assert ARQUIVO.exists()

assert len(historico_conversa) <= MAX_INTERACOES

assert all(
    isinstance(item["usuario"], str)
    and isinstance(item["chatbot"], str)
    for item in historico_conversa
)

with ARQUIVO.open(
    "r",
    encoding="utf-8"
) as arquivo:

    verificacao = json.load(arquivo)

assert verificacao["total_interacoes"] == len(
    historico_conversa
)

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Modelo DialoGPT-medium: OK")
print("Tokenizador: OK")
print("Geracao de texto: OK")
print("Historico de conversa: OK")
print("Limite de contexto: OK")
print("Arquivo JSON: OK")

print(
    "Interacoes realizadas:",
    len(historico_conversa)
)

print("\nAULA 15 EXECUTADA COM SUCESSO!")
