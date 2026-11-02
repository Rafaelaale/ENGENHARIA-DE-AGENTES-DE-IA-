# ==========================================================
# HUGGING FACE - AULA 13
# APLICANDO QUESTION ANSWERING AOS DADOS DO FAQ
# ==========================================================

from pathlib import Path
import json

import pandas as pd
import torch
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "deepset/roberta-base-squad2"

print("=" * 65)
print("AULA 13 - FAQ INTELIGENTE COM QUESTION ANSWERING")
print("=" * 65)


# ==========================================================
# 2 - DICIONARIO DE PERGUNTAS E CONTEXTOS
# ==========================================================

contextos = {

    "How do I create an account?":
    "You can create an account by clicking on the 'Sign Up' button on our homepage. "
    "You will need to provide your email address, create a password, and fill in "
    "some basic information about yourself. An account will help you track your "
    "orders, manage your personal settings, and speed up future transactions.",

    "Which payment methods do you accept?":
    "We accept a wide range of payment methods including Visa, MasterCard, "
    "American Express, Discover, PayPal, and Apple Pay. You can also pay "
    "using store credit or gift cards issued by our company.",

    "How can I track my order?":
    "Once your order has shipped, you will receive an email with a tracking "
    "number. You can use this number on our website's tracking page to see "
    "the current status of your delivery.",

    "Do you offer international shipping?":
    "Yes, we offer international shipping to most countries. Shipping costs "
    "and delivery times vary depending on the destination. All applicable "
    "customs fees, taxes, and duties are the responsibility of the customer "
    "and are calculated at checkout.",

    "How long does delivery take?":
    "For standard shipping, deliveries typically take between 3 to 5 business "
    "days. For expedited shipping, expect your order to arrive within 1 to 2 "
    "business days. Delivery times may vary based on your location and "
    "the time of the year.",

    "What is your return policy?":
    "Our return policy allows you to return products within 30 days of "
    "receiving them. Items must be in their original condition and packaging. "
    "Some items, such as perishable goods, are not eligible for return.",

    "Can I change or cancel my order after it's been placed?":
    "You can change or cancel your order within 24 hours of placing it without "
    "any additional charge.To make changes or cancel your order, please "
    "contact our customer service immediately.",

    "What should I do if I receive a damaged item?":
    "If you receive a damaged item, please contact our customer service "
    "within 48 hours of delivery to report the damage. You will need to "
    "provide your order number, the description of the damage, and photographic "
    "evidence. We will arrange for a replacement or refund as appropriate.",

    "How do I reset my password?":
    "If you've forgotten your password, go to the login page and click "
    "on 'Forgot Password'. Enter your email address and we will send you "
    "a link to reset your password. For security purposes, this link "
    "will expire within 24 hours."
}

assert len(contextos) == 9

print("\n1 - CONTEXTOS CARREGADOS")
print("Total de perguntas:", len(contextos))

for numero, pergunta in enumerate(contextos, start=1):
    print(f"{numero}. {pergunta}")


# ==========================================================
# 3 - SALVAR CONTEXTOS PARA A PROXIMA AULA
# ==========================================================

arquivo_contextos = DADOS / "aula13_contextos_faq.json"

with arquivo_contextos.open("w", encoding="utf-8") as f:
    json.dump(
        contextos,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nContextos salvos:")
print(arquivo_contextos)


# ==========================================================
# 4 - CARREGAR MODELO DA AULA 12
# ==========================================================

print("\n2 - CARREGANDO MODELO QUESTION ANSWERING")

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
# 5 - FUNCAO RESPONDENDO_FAQ
# ==========================================================

def respondendo_faq(pergunta):

    if pergunta not in contextos:
        raise ValueError(
            "Pergunta nao cadastrada no FAQ."
        )

    contexto = contextos[pergunta]

    resultado = QA_modelo(
        question=pergunta,
        context=contexto
    )

    return resultado["answer"]


# ==========================================================
# 6 - TESTAR AS QUATRO PERGUNTAS DA ALURA
# ==========================================================

print("\n3 - TESTANDO AS PERGUNTAS DA PROFESSORA")

perguntas_teste = [
    "How do I create an account?",
    "Which payment methods do you accept?",
    "Do you offer international shipping?",
    "Can I change or cancel my order after it's been placed?"
]

respostas_esperadas = [
    "by clicking on the 'Sign Up' button on our homepage",
    "Visa, MasterCard, American Express, Discover, PayPal, and Apple Pay",
    "we offer international shipping to most countries",
    "You can change or cancel your order within 24 hours"
]

correspondencias = 0

for numero, (pergunta, esperado) in enumerate(
    zip(perguntas_teste, respostas_esperadas),
    start=1
):

    resposta = respondendo_faq(pergunta)

    corresponde = (
        resposta.strip().casefold()
        == esperado.strip().casefold()
    )

    if corresponde:
        correspondencias += 1

    print("\n" + "-" * 60)
    print("TESTE:", numero)
    print("PERGUNTA:", pergunta)
    print("RESPOSTA:", resposta)
    print("ESPERADO:", esperado)
    print(
        "CORRESPONDENCIA:",
        "SIM" if corresponde else "NAO"
    )

print("\nCorrespondencias nos quatro exemplos:")
print(f"{correspondencias}/4")


# ==========================================================
# 7 - PROCESSAR TODAS AS NOVE PERGUNTAS
# ==========================================================

print("\n4 - PROCESSANDO FAQ COMPLETO")

registros = []

for numero, (pergunta, contexto) in enumerate(
    contextos.items(),
    start=1
):

    resultado = QA_modelo(
        question=pergunta,
        context=contexto
    )

    resposta = resultado["answer"]
    score = float(resultado["score"])
    inicio = int(resultado["start"])
    fim = int(resultado["end"])

    assert 0 <= score <= 1
    assert 0 <= inicio <= fim <= len(contexto)
    assert contexto[inicio:fim] == resposta

    registros.append({
        "Pergunta": pergunta,
        "Resposta": resposta,
        "Score": score,
        "Inicio": inicio,
        "Fim": fim,
        "Contexto": contexto
    })

    print(
        f"\n[{numero}/{len(contextos)}] "
        f"{pergunta}"
    )

    print("Resposta:", resposta)
    print(f"Score: {score:.2%}")


# ==========================================================
# 8 - CRIAR DATAFRAME PANDAS
# ==========================================================

print("\n5 - DATAFRAME DO FAQ")

dados = pd.DataFrame(registros)

assert len(dados) == 9

with pd.option_context(
    "display.max_colwidth", 85,
    "display.width", 200
):
    print(
        dados[
            ["Pergunta", "Resposta", "Score"]
        ].to_string(index=False)
    )


# ==========================================================
# 9 - SALVAR CSV
# ==========================================================

arquivo_csv = (
    RESULTADOS / "aula13_faq_respostas.csv"
)

dados.to_csv(
    arquivo_csv,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV salvo:")
print(arquivo_csv)


# ==========================================================
# 10 - SALVAR JSON
# ==========================================================

arquivo_json = (
    RESULTADOS / "aula13_faq_respostas.json"
)

with arquivo_json.open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "modelo": MODELO,
            "total_perguntas": len(contextos),
            "correspondencias_exemplos": correspondencias,
            "resultados": registros
        },
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nJSON salvo:")
print(arquivo_json)


# ==========================================================
# 11 - VALIDACAO FINAL
# ==========================================================

verificacao = pd.read_csv(
    arquivo_csv
)

assert len(verificacao) == len(contextos)

assert verificacao["Resposta"].notna().all()

assert arquivo_contextos.exists()
assert arquivo_csv.exists()
assert arquivo_json.exists()

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("Modelo RoBERTa: OK")
print("Pipeline Question Answering: OK")
print("Dicionario de contextos: OK")
print("Funcao respondendo_faq: OK")
print("Perguntas cadastradas:", len(contextos))
print("Perguntas processadas:", len(registros))
print("Extracao validada: OK")
print("CSV: OK")
print("JSON: OK")
print(
    "Correspondencia com quatro exemplos:",
    f"{correspondencias}/4"
)

print("\nAULA 13 EXECUTADA COM SUCESSO!")
