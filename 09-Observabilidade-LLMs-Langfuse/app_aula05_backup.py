import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI

# CONFIGURACOES
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "bitext_customer_support.csv"

load_dotenv(BASE_DIR / ".env")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

# INICIALIZAR LANGFUSE
langfuse = get_client()

# INICIALIZAR OLLAMA
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# ======================================
# CARREGAR PERGUNTA DO DATASET
# ======================================

def get_customer_question():

    print("\nCarregando dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("O dataset esta vazio.")

    colunas = [
        "instruction",
        "category",
        "intent",
        "response"
    ]

    faltando = [
        coluna for coluna in colunas
        if coluna not in df.columns
    ]

    if faltando:
        raise ValueError(
            f"Colunas ausentes: {faltando}"
        )

    print(f"Total de registros: {len(df)}")
    print(f"Colunas: {list(df.columns)}")

    # Primeira linha do dataset
    registro = df.iloc[0]

    question = str(registro["instruction"])

    print("\nCategoria:", registro["category"])
    print("Intencao:", registro["intent"])

    return question


# ======================================
# PIPELINE INSTRUMENTADO
# ======================================

def ask_llm(question):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline"
    ) as root_span:

        # EVENTO: DATASET CARREGADO
        with root_span.start_as_current_observation(
            as_type="event",
            name="dataset-loaded"
        ):
            pass

        # GENERATION: CHAMADA AO OLLAMA
        completion = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente de "
                        "atendimento ao cliente. "
                        "Responda de forma clara, "
                        "objetiva e educada. "
                        "Nao invente dados de pedidos."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            name="ollama-model"
        )

        answer = completion.choices[0].message.content

        # ATUALIZAR SPAN RAIZ
        root_span.update(
            input={"question": question},
            output={"answer": answer}
        )

    # ENVIAR OBSERVACOES PENDENTES
    langfuse.flush()

    return answer


# ======================================
# EXECUCAO PRINCIPAL
# ======================================

if __name__ == "__main__":

    print("\n===== AULA 05 - LANGFUSE =====")

    print(f"Modelo: {OLLAMA_MODEL}")

    try:
        pergunta_dataset = get_customer_question()

        print("\nPergunta selecionada:")
        print(pergunta_dataset)

        resposta = ask_llm(pergunta_dataset)

        print("\n===== RESPOSTA DO OLLAMA =====")
        print(resposta)

        print("\nPipeline executado com sucesso.")
        print("Trace: customer-support-pipeline")

    except Exception as erro:
        print("\nERRO:")
        print(f"{type(erro).__name__}: {erro}")

    finally:
        langfuse.flush()
