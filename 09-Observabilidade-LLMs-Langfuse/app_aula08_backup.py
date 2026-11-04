import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI

# CONFIGURACOES

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "data" / "bitext_customer_support.csv"
)

load_dotenv(BASE_DIR / ".env")

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

PROMPT_NAME = "customer_support_assistant"
PROMPT_LABEL = "production"

SESSION_ID = "chat-session-123"
USER_ID = "id-123"

PIPELINE_VERSION = "v2"
ENVIRONMENT = "develop"

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# CARREGAR PERGUNTA DO DATASET

def get_customer_question():

    df = pd.read_csv(DATASET_PATH)

    if len(df) <= 4:
        raise ValueError("Dataset insuficiente.")

    registro = df.iloc[4]

    question = str(registro["instruction"])

    print(f"\nTotal de registros: {len(df)}")
    print(f"Categoria: {registro['category']}")
    print(f"Intencao: {registro['intent']}")

    return question


# PIPELINE COM PROMPT EXTERNO

def run_customer_support_pipeline():

    with langfuse.start_as_current_observation(
        as_type="span",
        name="customer-support-pipeline"
    ) as root_span:

        with propagate_attributes(
            session_id=SESSION_ID,
            user_id=USER_ID,
            tags=[
                "customer-support",
                "prompt-load",
                OLLAMA_MODEL
            ],
            metadata={
                "environment": ENVIRONMENT,
                "version": PIPELINE_VERSION,
                "dataset_path": str(DATASET_PATH)
            }
        ):

            # SPAN 1 - DATASET
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

            # EVENT - DATASET CARREGADO
            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-loaded"
            ):
                pass

            # SPAN 2 - PREPROCESSAMENTO
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

            # SPAN 3 - CARREGAR PROMPT DO LANGFUSE
            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompt"
            ) as prompt_span:

                prompt_client = langfuse.get_prompt(
                    name=PROMPT_NAME,
                    label=PROMPT_LABEL,
                    type="text"
                )

                system_prompt = prompt_client.prompt

                prompt_span.update(
                    input={
                        "prompt_name": PROMPT_NAME,
                        "prompt_label": PROMPT_LABEL
                    },
                    output={
                        "PROMPT_NAME": PROMPT_NAME,
                        "PROMPT_LABEL": PROMPT_LABEL,
                        "prompt_version": prompt_client.version,
                        "system_prompt": system_prompt
                    }
                )

            print("\nPrompt carregado do LangFuse")
            print(f"Nome: {PROMPT_NAME}")
            print(f"Label: {PROMPT_LABEL}")
            print(f"Versao: {prompt_client.version}")

            # GENERATION - OLLAMA
            completion = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": cleaned_question
                    }
                ],
                name="llm-generation"
            )

            answer = completion.choices[0].message.content

            # ATUALIZAR SPAN RAIZ
            root_span.update(
                input={
                    "question": question
                },
                output={
                    "answer": answer
                },
                metadata={
                    "PROMPT_NAME": PROMPT_NAME,
                    "PROMPT_LABEL": PROMPT_LABEL,
                    "prompt_version": prompt_client.version,
                    "system_prompt": system_prompt
                }
            )

    langfuse.flush()

    return answer


# EXECUCAO

if __name__ == "__main__":

    print("\n===== ALURA - AULA 08 =====")
    print("VERSIONAMENTO DE PROMPTS")
    print(f"Modelo: {OLLAMA_MODEL}")

    try:
        if not langfuse.auth_check():
            raise RuntimeError(
                "Autenticacao LangFuse falhou. "
                "Confira credenciais e regiao."
            )

        resposta = run_customer_support_pipeline()

        print("\n===== RESPOSTA DO OLLAMA =====")
        print(resposta)

        print("\n===== OBSERVABILIDADE =====")
        print("Trace: customer-support-pipeline")
        print("Span: load_dataset")
        print("Event: dataset-loaded")
        print("Span: preprocessing-question")
        print("Span: load-prompt")
        print("Generation: llm-generation")

        print("\nPipeline executado com sucesso.")

    except Exception as erro:
        print("\nERRO NA AULA 08:")
        print(f"{type(erro).__name__}: {erro}")
        raise

    finally:
        langfuse.flush()