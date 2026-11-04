import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=False)

DATASET_PATH = BASE_DIR / "data" / "bitext_customer_support.csv"

JUDGE_PROMPT_PATH = (
    "prompts/customer_support_answer_judge.md"
)

ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_MODEL_JUDGE = os.getenv(
    "OLLAMA_MODEL_JUDGE",
    OLLAMA_MODEL
)

USER_ID = "id-123"

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def load_prompt(path):
    arquivo = BASE_DIR / path

    if not arquivo.is_file():
        raise FileNotFoundError(
            f"Prompt nao encontrado: {arquivo}"
        )

    conteudo = arquivo.read_text(encoding="utf-8-sig")

    if not conteudo.strip():
        raise ValueError(
            f"Prompt vazio: {arquivo}"
        )

    return conteudo


def parse_judge_response(response):
    """
    Interpreta a resposta textual:
    SCORE: 1 a 5
    REASON: justificativa
    """

    match_score = re.search(
        r"(?im)^\s*SCORE\s*:\s*([1-5])\s*$",
        response
    )

    match_reason = re.search(
        r"(?im)^\s*REASON\s*:\s*(.+)",
        response
    )

    if not match_score:
        return None, "Formato SCORE invalido"

    score = int(match_score.group(1))

    reason = (
        match_reason.group(1).strip()
        if match_reason
        else "Justificativa nao informada"
    )

    return score, reason


def run_llm_judge_pipeline():

    with langfuse.start_as_current_observation(
        as_type="span",
        name="llm-judge-pipeline"
    ) as root_span:

        with propagate_attributes(
            session_id="llm-judge-session-123",
            user_id=USER_ID,
            tags=[
                "customer-support",
                "llm-as-judge",
                OLLAMA_MODEL
            ],
            metadata={
                "environment": "develop",
                "version": "v1",
                "dataset_path": str(DATASET_PATH),
                "judge_prompt_path": JUDGE_PROMPT_PATH,
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL
            }
        ):

            # 1. CARREGAMENTO DO DATASET

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)

                required = {
                    "instruction",
                    "response",
                    "category"
                }

                missing = required - set(df.columns)

                if missing:
                    raise ValueError(
                        f"Colunas ausentes: {sorted(missing)}"
                    )

                if len(df) < 5:
                    raise ValueError(
                        "Dataset precisa ter pelo menos 5 linhas."
                    )

                row = df.iloc[4]

                question = str(row["instruction"])
                expected_answer = str(row["response"])
                category = str(row["category"])

                dataset_span.update(
                    input={
                        "dataset_path": str(DATASET_PATH)
                    },
                    output={
                        "selected_question": question,
                        "expected_answer": expected_answer
                    },
                    metadata={
                        "category": category
                    }
                )

                dataset_span.create_event(
                    name="dataset-loaded",
                    metadata={
                        "rows": len(df)
                    }
                )

            # 2. CARREGAMENTO DOS PROMPTS

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts"
            ) as prompt_load_span:

                judge_prompt = load_prompt(
                    JUDGE_PROMPT_PATH
                )

                prompt_client = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL
                )

                system_prompt = prompt_client.prompt

                if not isinstance(system_prompt, str):
                    raise TypeError(
                        "Prompt do assistente nao e texto simples."
                    )

                prompt_load_span.update(
                    output={
                        "judge_prompt_path": JUDGE_PROMPT_PATH,
                        "answer_prompt_name": ANSWER_PROMPT_NAME
                    }
                )

            # 3. GERACAO DA RESPOSTA

            with root_span.start_as_current_observation(
                as_type="span",
                name="generate-answer"
            ) as generate_span:

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
                    ],
                    name="llm-generate-answer"
                )

                model_answer = (
                    completion.choices[0].message.content or ""
                ).strip()

                generate_span.update(
                    input={
                        "question": question
                    },
                    output={
                        "model_answer": model_answer
                    },
                    metadata={
                        "prompt_name": ANSWER_PROMPT_NAME
                    }
                )

            # 4. LLM COMO JUIZ

            with root_span.start_as_current_observation(
                as_type="span",
                name="llm-judge-evaluation"
            ) as judge_span:

                judge_user_message = f"""
Pergunta:
{question}

Resposta esperada:
{expected_answer}

Resposta gerada:
{model_answer}
"""

                completion = client.chat.completions.create(
                    model=OLLAMA_MODEL_JUDGE,
                    messages=[
                        {
                            "role": "system",
                            "content": judge_prompt
                        },
                        {
                            "role": "user",
                            "content": judge_user_message
                        }
                    ],
                    name="llm-judge"
                )

                judge_response = (
                    completion.choices[0].message.content or ""
                ).strip()

                score, reason = parse_judge_response(
                    judge_response
                )

                if score is None:
                    judge_span.update(
                        level="ERROR",
                        status_message=reason,
                        input=judge_user_message,
                        output=judge_response
                    )
                else:
                    judge_span.update(
                        input=judge_user_message,
                        output={
                            "judge_response": judge_response,
                            "score": score,
                            "reason": reason
                        },
                        metadata={
                            "judge_model": OLLAMA_MODEL_JUDGE
                        }
                    )

                df.loc[4, "score"] = (
                    score if score is not None else 0
                )

            # 5. ATUALIZACAO DO TRACE

            root_span.update(
                input={
                    "question": question,
                    "expected_answer": expected_answer
                },
                output={
                    "model_answer": model_answer,
                    "judge_response": judge_response,
                    "score": score,
                    "reason": reason
                },
                metadata={
                    "category": category,
                    "judge_model": OLLAMA_MODEL_JUDGE
                }
            )

    langfuse.flush()

    return {
        "question": question,
        "expected_answer": expected_answer,
        "model_answer": model_answer,
        "judge_response": judge_response,
        "score": score,
        "reason": reason
    }


if __name__ == "__main__":
    try:
        result = run_llm_judge_pipeline()

        print("\n===== ALURA - LLM AS JUDGE =====")
        print("Modelo gerador:", OLLAMA_MODEL)
        print("Modelo juiz:", OLLAMA_MODEL_JUDGE)

        print("\nPERGUNTA:")
        print(result["question"])

        print("\nRESPOSTA ESPERADA:")
        print(result["expected_answer"])

        print("\nRESPOSTA GERADA:")
        print(result["model_answer"])

        print("\nAVALIACAO DO JUIZ:")
        print(result["judge_response"])

        print("\nSCORE:", result["score"])
        print("JUSTIFICATIVA:", result["reason"])

        print("\nTrace: llm-judge-pipeline")
        print("Avaliacao concluida.")

    finally:
        langfuse.flush()
