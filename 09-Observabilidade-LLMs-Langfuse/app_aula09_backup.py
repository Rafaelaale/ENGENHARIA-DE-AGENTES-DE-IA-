import os
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI
from langfuse.api.commons.errors.not_found_error import NotFoundError

# ============================================
# CONFIGURACOES
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "data" / "bitext_customer_support.csv"
)

load_dotenv(BASE_DIR / ".env", override=True)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:latest"
)

PROMPT_NAME = "customer_support_assistant"

PROMPT_LABELS = [
    "baseline",
    "improved"
]

SESSION_ID = "chat-session-123"
USER_ID = "id-123"

PIPELINE_VERSION = "experiment-v1"
ENVIRONMENT = "develop"

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# ============================================
# CONFIGURAR VERSOES DOS PROMPTS
# ============================================

def preparar_prompts():

    print("\nVerificando prompts no LangFuse...")

    if not langfuse.auth_check():
        raise RuntimeError(
            "Autenticacao LangFuse falhou."
        )

    # Recuperar prompt original da Aula 08
    production = langfuse.get_prompt(
        name=PROMPT_NAME,
        label="production",
        type="text"
    )

    # BASELINE - MESMO TEXTO DE PRODUCAO
    try:
        baseline = langfuse.get_prompt(
            name=PROMPT_NAME,
            label="baseline",
            type="text"
        )

    except NotFoundError:
        # Adicionar baseline a versao existente,
        # preservando os demais rotulos.
        labels = list(
            dict.fromkeys(
                [x for x in production.labels if x != "latest"] + [
                    "production",
                    "baseline"
                ]
            )
        )

        langfuse.update_prompt(
            name=PROMPT_NAME,
            version=production.version,
            new_labels=labels
        )

        baseline = langfuse.get_prompt(
            name=PROMPT_NAME,
            label="baseline",
            type="text"
        )

    # IMPROVED - PROMPT APRIMORADO
    improved_text = (
        "Voce e um assistente de atendimento "
        "ao cliente de alta qualidade. "
        "Siga estas regras: "
        "1. Comece com uma resposta direta. "
        "2. Use linguagem clara e objetiva. "
        "3. Evite respostas vagas. "
        "4. Apresente uma explicacao breve. "
        "5. Sugira proximos passos praticos. "
        "6. Se a pergunta estiver incompleta, "
        "solicite esclarecimentos. "
        "7. Nao invente dados de pedidos, "
        "contatos ou politicas da empresa."
    )

    try:
        improved = langfuse.get_prompt(
            name=PROMPT_NAME,
            label="improved",
            type="text"
        )

    except NotFoundError:
        langfuse.create_prompt(
            name=PROMPT_NAME,
            type="text",
            prompt=improved_text,
            labels=["improved"]
        )

        improved = langfuse.get_prompt(
            name=PROMPT_NAME,
            label="improved",
            type="text"
        )

    print(
        f"Baseline: versao {baseline.version}"
    )

    print(
        f"Improved: versao {improved.version}"
    )


# ============================================
# CARREGAMENTO DO DATASET
# ============================================

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


# ============================================
# EXECUTAR EXPERIMENTO
# ============================================

def run_prompt_experiment():

    results = []

    with langfuse.start_as_current_observation(
        as_type="span",
        name="prompt-experiment"
    ) as root_span:

        with propagate_attributes(
            session_id=SESSION_ID,
            user_id=USER_ID,
            tags=[
                "customer-support",
                "prompt-experiment",
                OLLAMA_MODEL
            ],
            metadata={
                "environment": ENVIRONMENT,
                "version": PIPELINE_VERSION,
                "dataset_path": str(DATASET_PATH)
            }
        ):

            # ====================================
            # SPAN - CARREGAR DATASET
            # ====================================

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

            # ====================================
            # EVENTO
            # ====================================

            with root_span.start_as_current_observation(
                as_type="event",
                name="dataset-loaded"
            ):
                pass

            # ====================================
            # PREPROCESSAMENTO
            # ====================================

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

            # ====================================
            # EXPERIMENTO COM DOIS PROMPTS
            # ====================================

            for label in PROMPT_LABELS:

                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"run-{label}"
                ) as experiment_span:

                    # CARREGAR PROMPT PELO LABEL
                    prompt_client = langfuse.get_prompt(
                        name=PROMPT_NAME,
                        label=label,
                        type="text"
                    )

                    system_prompt = prompt_client.prompt

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
                        name=f"llm-generation-{label}"
                    )

                    answer = (
                        completion.choices[0].message.content
                    )

                    experiment_span.update(
                        input={
                            "question": cleaned_question
                        },
                        output={
                            "answer": answer
                        },
                        metadata={
                            "PROMPT_LABEL": label,
                            "PROMPT_VERSION": prompt_client.version
                        }
                    )

                    results.append({
                        "label": label,
                        "version": prompt_client.version,
                        "answer": answer
                    })

            # ====================================
            # CONSOLIDAR COMPARACAO
            # ====================================

            root_span.update(
                input={
                    "question": question
                },
                output={
                    "results": results
                },
                metadata={
                    "comparison": results
                }
            )

    langfuse.flush()

    return results


# ============================================
# EXECUCAO PRINCIPAL
# ============================================

if __name__ == "__main__":

    print("\n===== ALURA - AULA 09 =====")
    print("EXPERIMENTOS COM PROMPTS")
    print(f"Modelo: {OLLAMA_MODEL}")

    try:
        preparar_prompts()

        results = run_prompt_experiment()

        print("\n===== COMPARACAO =====")

        for resultado in results:

            print("\n" + "=" * 45)

            print(
                f"PROMPT: {resultado['label']}"
            )

            print(
                f"VERSAO: {resultado['version']}"
            )

            print("\nRESPOSTA:")
            print(resultado["answer"])

        print("\n" + "=" * 45)

        print("Trace: prompt-experiment")
        print("Span: run-baseline")
        print("Span: run-improved")

        print("\nExperimento concluido.")

    except Exception as erro:
        print("\nERRO NO EXPERIMENTO:")
        print(f"{type(erro).__name__}: {erro}")
        raise

    finally:
        langfuse.flush()