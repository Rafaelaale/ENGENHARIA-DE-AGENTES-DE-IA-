# ==========================================================
# HUGGING FACE - AULA 12
# SELECIONANDO O MODELO PARA QUESTION ANSWERING
# ==========================================================

from pathlib import Path
import json

import torch
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "deepset/roberta-base-squad2"

ARQUIVO_JSON = (
    RESULTADOS / "aula12_perguntas_respostas.json"
)

print("=" * 65)
print("AULA 12 - QUESTION ANSWERING")
print("SISTEMA DE PERGUNTAS E RESPOSTAS PARA E-COMMERCE")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR MODELO PRE-TREINADO
# ==========================================================

print("\n1 - CARREGANDO MODELO")

print("Modelo:", MODELO)
print("Tarefa: question-answering")
print("Idioma: ingles")
print("Dispositivo: CPU")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

QA_modelo = pipeline(
    "question-answering",
    model=MODELO,
    tokenizer=MODELO,
    device=-1
)

print("\nMODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 3 - CRIAR FUNCAO DE PERGUNTAS E RESPOSTAS
# ==========================================================

def obter_resposta(pergunta, contexto):

    resposta = QA_modelo(
        question=pergunta,
        context=contexto
    )

    return resposta


# ==========================================================
# 4 - PERGUNTA DA AULA
# ==========================================================

print("\n2 - PREPARANDO PERGUNTA")

pergunta = (
    "What are the payment options available?"
)

contexto = (
    "In our e-commerce, you can make payments "
    "using credit card, debit card, Apple Pay, "
    "or through bank transfers."
)

print("\nPERGUNTA:")
print(pergunta)

print("\nCONTEXTO:")
print(contexto)


# ==========================================================
# 5 - EXECUTAR O MODELO
# ==========================================================

print("\n3 - OBTENDO RESPOSTA")

resposta = obter_resposta(
    pergunta,
    contexto
)

print("\nRESULTADO COMPLETO:")
print(resposta)


# ==========================================================
# 6 - EXTRAIR INFORMACOES
# ==========================================================

score = float(resposta["score"])

inicio = int(resposta["start"])
fim = int(resposta["end"])

texto_resposta = resposta["answer"]

print("\n4 - RESPOSTA INTERPRETADA")

print("Pergunta:", pergunta)
print("Resposta:", texto_resposta)

print(
    "Score:",
    f"{score:.6f}"
)

print(
    "Score percentual:",
    f"{score * 100:.2f}%"
)

print("Inicio:", inicio)
print("Fim:", fim)


# ==========================================================
# 7 - VALIDAR O TRECHO DO CONTEXTO
# ==========================================================

print("\n5 - VALIDANDO RESPOSTA")

assert 0 <= score <= 1

assert 0 <= inicio <= fim <= len(contexto)

trecho_contexto = contexto[inicio:fim]

assert trecho_contexto == texto_resposta, (
    "O trecho extraido nao corresponde "
    "as posicoes retornadas."
)

resposta_esperada = (
    "credit card, debit card, Apple Pay, "
    "or through bank transfers"
)

corresponde = (
    texto_resposta.strip().casefold()
    == resposta_esperada.casefold()
)

print(
    "Resposta localizada no contexto: SIM"
)

print(
    "Correspondencia com a Alura:",
    "SIM" if corresponde else "NAO"
)

if not corresponde:

    print(
        "O modelo selecionou um trecho diferente "
        "do exemplo apresentado pela professora."
    )


# ==========================================================
# 8 - SALVAR RESULTADO JSON
# ==========================================================

print("\n6 - SALVANDO RESULTADOS")

resultado_final = {

    "modelo": MODELO,

    "tarefa": "question-answering",

    "idioma": "ingles",

    "pergunta": pergunta,

    "contexto": contexto,

    "resposta": texto_resposta,

    "score": score,

    "inicio": inicio,

    "fim": fim,

    "resposta_esperada": resposta_esperada,

    "corresponde_alura": corresponde
}

with ARQUIVO_JSON.open(
    "w",
    encoding="utf-8"
) as arquivo:

    json.dump(
        resultado_final,
        arquivo,
        ensure_ascii=False,
        indent=4
    )

print("Arquivo salvo:")
print(ARQUIVO_JSON)


# ==========================================================
# 9 - VALIDACAO FINAL
# ==========================================================

with ARQUIVO_JSON.open(
    "r",
    encoding="utf-8"
) as arquivo:

    verificacao = json.load(arquivo)

assert verificacao["resposta"] == texto_resposta
assert verificacao["score"] == score
assert ARQUIVO_JSON.exists()

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Modelo RoBERTa: OK")
print("Pipeline Question Answering: OK")
print("Funcao obter_resposta: OK")
print("Pergunta em ingles: OK")
print("Contexto utilizado: OK")
print("Resposta extraida do contexto: OK")
print("Score validado: OK")
print("Arquivo JSON: OK")

print("\nAULA 12 EXECUTADA COM SUCESSO!")
