import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI

from dotenv import load_dotenv
load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / 'data' / 'bitext_customer_support.csv'
PROMPT_FILES = {'baseline': 'prompts/customer_support_baseline.md', 'improved': 'prompts/customer_support_improved.md'}
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
SESSION_ID = 'prompt-session-123'
USER_ID = 'id-123'
langfuse = get_client()
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def load_prompt(path):
    arquivo = BASE_DIR / path
    if not arquivo.is_file():
        raise FileNotFoundError(f'Prompt nao encontrado: {arquivo}')
    with open(arquivo, 'r', encoding='utf-8') as file:
        conteudo = file.read()
    if not conteudo.strip():
        raise ValueError(f'Prompt vazio: {arquivo}')
    return conteudo

PROMPT_PATH = "prompts/customer_support_intent_classifier.md"


def run_evaluation_pipeline():
    with langfuse.start_as_current_observation(
        as_type="span",
        name="intent-evaluation-pipeline"
    ) as root_span:

        with propagate_attributes(
            session_id="intent-eval-session-123",
            user_id=USER_ID,
            tags=[
                "customer-support",
                "intent-evaluation",
                OLLAMA_MODEL
            ],
            metadata={
                "environment": "develop",
                "evaluation_type": "ground-truth"
            }
        ):

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)

                required_columns = {
                    "instruction",
                    "intent",
                    "category"
                }

                missing = required_columns - set(df.columns)

                if missing:
                    raise ValueError(
                        "Colunas ausentes no dataset: "
                        + ", ".join(sorted(missing))
                    )

                if len(df) <= 4:
                    raise ValueError(
                        "Dataset precisa conter pelo menos 5 linhas."
                    )

                row = df.iloc[4]

                question = str(row["instruction"])
                true_intent = str(row["intent"]).strip()
                category = str(row["category"])

                dataset_span.update(
                    input={
                        "dataset_path": str(DATASET_PATH)
                    },
                    output={
                        "selected_question": question,
                        "true_intent": true_intent
                    },
                    metadata={
                        "category": category
                    }
                )

                dataset_span.create_event(
                    name="dataset-loaded",
                    metadata={"rows": len(df)}
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="predict-intent"
            ) as predict_span:

                system_prompt = load_prompt(PROMPT_PATH)

                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name="llm-predict-intent",
                    model=OLLAMA_MODEL
                ) as generation:

                    completion = client.chat.completions.create(
                        model=OLLAMA_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": question
                            }
                        ]
                    )

                    predicted_intent = (
                        completion.choices[0].message.content or ""
                    ).strip()

                    generation.update(
                        input=question,
                        output=predicted_intent
                    )

                predict_span.update(
                    input={"question": question},
                    output={
                        "predicted_intent": predicted_intent
                    },
                    metadata={
                        "prompt_path": PROMPT_PATH
                    }
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="evaluation-intent"
            ) as eval_span:

                is_correct = predicted_intent == true_intent
                score = 1 if is_correct else 0

                eval_span.update(
                    input={
                        "predicted_intent": predicted_intent,
                        "true_intent": true_intent
                    },
                    output={
                        "is_correct": is_correct,
                        "score": score
                    }
                )

            root_span.update(
                input={
                    "question": question,
                    "true_intent": true_intent
                },
                output={
                    "predicted_intent": predicted_intent,
                    "is_correct": is_correct,
                    "score": score
                },
                metadata={
                    "prompt_path": PROMPT_PATH,
                    "category": category
                }
            )

    langfuse.flush()

    return {
        "question": question,
        "true_intent": true_intent,
        "predicted_intent": predicted_intent,
        "score": score
    }


if __name__ == "__main__":
    try:
        results = run_evaluation_pipeline()

        print("\n===== ALURA - GROUND TRUTH =====")
        print("Modelo:", OLLAMA_MODEL)
        print("Pergunta:", results["question"])
        print("Intencao esperada:", results["true_intent"])
        print("Intencao prevista:", results["predicted_intent"])
        print("Score:", results["score"])
        print("Trace: intent-evaluation-pipeline")
        print("\nAvaliacao concluida.")

    finally:
        langfuse.flush()
