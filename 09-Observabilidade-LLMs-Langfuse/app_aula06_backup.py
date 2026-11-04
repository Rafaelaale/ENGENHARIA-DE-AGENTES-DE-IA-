import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI


# =========================================
# CONFIGURACOES
# =========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "data" / "bitext_customer_support.csv"
)

load_dotenv(BASE_DIR / ".env")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# =========================================
# CARREGAMENTO DO DATASET
# =========================================

def get_customer_question():

    df = pd.read_csv(DATASET_PATH)

    if df.empty:
        raise ValueError("Dataset vazio.")

    if "instruction" not in df.columns:
        raise ValueError("Coluna instruction ausente.")

    if len(df) <= 4:
        raise ValueError("Dataset possui menos de 5 linhas.")

    # SELECIONAR REGISTRO DE INDICE 4
    question = str(df.iloc[4]["instruction"])

    print(f"\nTotal de registros: {len(df)}")
    print(f"Categoria: {df.iloc[4]['category']}")
    print(f"Intencao: {df.iloc[4]['intent']}")

    return question


# =========================================
# PIPELINE DE ATENDIMENTO AO CLIENTE
# =========================================

def run_customer_support_pipeline():

    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline"
    ) as root_span:

        # =================================
        # SPAN 1 - CARREGAR DATASET
        # =================================

        with root_span.start_as_current_observation(
            as_type="span",
            name="load_dataset"
        ) as dataset_span:

            question = get_customer_question()

            dataset_span.update(
                input={
                    "dataset_path": str(DATASET_PATH)
                },
                output={
                    "selected_question": question
                }
            )

        # =================================
        # EVENT - DATASET CARREGADO
        # =================================

        with root_span.start_as_current_observation(
            as_type="event",
            name="dataset-loaded"
        ):
            pass

        # =================================
        # SPAN 2 - PREPROCESSAMENTO
        # =================================

        with root_span.start_as_current_observation(
            as_type="span",
            name="preprocessing-question"
        ) as preprocessing_span:

            cleaned_question = question.strip()

            preprocessing_span.update(
                input={
                    "question": question
                },
                output={
                    "cleaned_question": cleaned_question
                },
                metadata={
                    "len_question": len(cleaned_question)
                }
            )

        # =================================
        # GENERATION - OLLAMA
        # =================================

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
                        "Nao invente informacoes "
                        "sobre pedidos."
                    )
                },
                {
                    "role": "user",
                    "content": cleaned_question
                }
            ],
            name="llm-generation"
        )

        answer = completion.choices[0].message.content

        # =================================
        # ATUALIZAR SPAN RAIZ
        # =================================

        root_span.update(
            input={
                "question": question
            },
            output={
                "answer": answer
            }
        )

    # ENVIAR OBSERVACOES
    langfuse.flush()

    return answer


# =========================================
# EXECUCAO PRINCIPAL
# =========================================

if __name__ == "__main__":

    print("\n===== ALURA - AULA 06 =====")
    print("PIPELINE COM SPANS")
    print(f"Modelo: {OLLAMA_MODEL}")

    try:
        resposta = run_customer_support_pipeline()

        print("\n===== RESPOSTA DO OLLAMA =====")
        print(resposta)

        print("\n===== OBSERVABILIDADE =====")
        print("Trace: customer-support-pipeline")
        print("Span: load_dataset")
        print("Event: dataset-loaded")
        print("Span: preprocessing-question")
        print("Generation: llm-generation")

        print("\nPipeline executado com sucesso.")

    except Exception as erro:
        print("\nERRO NA APLICACAO:")
        print(f"{type(erro).__name__}: {erro}")

        raise

    finally:
        langfuse.flush()