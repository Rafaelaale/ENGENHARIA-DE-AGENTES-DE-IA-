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
JUDGE_PROMPT_PATH = BASE_DIR / "prompts" / "customer_support_answer_judge.md"

ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"

FALLBACK_PROMPT_NAME = "customer_support_assistant"
FALLBACK_PROMPT_LABEL = "fallback"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_MODEL_JUDGE = os.getenv("OLLAMA_MODEL_JUDGE", OLLAMA_MODEL)

N_EXAMPLES = 5
SCORE_THRESHOLD_FALLBACK = 2

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def extract_score(text):
    match = re.search(
        r"(?im)^\s*SCORE\s*:\s*([1-5])\b",
        text or ""
    )
    return int(match.group(1)) if match else None


def classify_severity(score):
    if score is None:
        return "review"
    if score <= SCORE_THRESHOLD_FALLBACK:
        return "critical"
    if score == 3:
        return "review"
    return "ok"


def generate_answer(system_prompt, question, name, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        name=name
    )

    return (completion.choices[0].message.content or "").strip()


def judge_answer(judge_prompt, question, expected_answer, answer, name):
    judge_input = (
        f"Pergunta: {question}\n\n"
        f"Resposta esperada: {expected_answer}\n\n"
        f"Resposta gerada: {answer}"
    )

    return generate_answer(
        system_prompt=judge_prompt,
        question=judge_input,
        name=name,
        model=OLLAMA_MODEL_JUDGE
    )


def run_batch_llm_judge_monitoring():
    results = []

    with langfuse.start_as_current_observation(
        as_type="span",
        name="batch-llm-judge-monitoring"
    ) as root_span:

        with propagate_attributes(
            session_id="customer-support-monitoring-session-123",
            user_id="id-123",
            tags=[
                "customer-support",
                "llm-as-judge-monitoring"
            ],
            metadata={
                "environment": "develop",
                "version": "v1",
                "answer_prompt_name": ANSWER_PROMPT_NAME,
                "answer_prompt_label": ANSWER_PROMPT_LABEL,
                "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                "fallback_prompt_label": FALLBACK_PROMPT_LABEL
            }
        ):

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-dataset"
            ) as dataset_span:

                df = pd.read_csv(DATASET_PATH)

                required = {"instruction", "response", "category"}
                missing = required - set(df.columns)

                if missing:
                    raise ValueError(
                        f"Colunas ausentes no dataset: {missing}"
                    )

                sample_df = df.head(N_EXAMPLES).copy()

                dataset_span.update(
                    input={"dataset_path": str(DATASET_PATH)},
                    output={"num_samples": len(sample_df)}
                )

                dataset_span.create_event(
                    name="dataset-loaded",
                    metadata={"num_samples": len(sample_df)}
                )

            with root_span.start_as_current_observation(
                as_type="span",
                name="load-prompts"
            ) as prompt_load_span:

                judge_prompt = JUDGE_PROMPT_PATH.read_text(
                    encoding="utf-8-sig"
                ).strip()

                answer_prompt = langfuse.get_prompt(
                    name=ANSWER_PROMPT_NAME,
                    label=ANSWER_PROMPT_LABEL
                )

                fallback_prompt = langfuse.get_prompt(
                    name=FALLBACK_PROMPT_NAME,
                    label=FALLBACK_PROMPT_LABEL
                )

                answer_system_prompt = answer_prompt.prompt
                fallback_system_prompt = fallback_prompt.prompt

                if not isinstance(answer_system_prompt, str):
                    raise TypeError("Prompt de producao nao e texto.")

                if not isinstance(fallback_system_prompt, str):
                    raise TypeError("Prompt de fallback nao e texto.")

                prompt_load_span.update(
                    output={
                        "answer_prompt_name": ANSWER_PROMPT_NAME,
                        "answer_prompt_label": ANSWER_PROMPT_LABEL,
                        "fallback_prompt_name": FALLBACK_PROMPT_NAME,
                        "fallback_prompt_label": FALLBACK_PROMPT_LABEL
                    }
                )

            for idx, row in sample_df.iterrows():

                question = str(row["instruction"])
                expected_answer = str(row["response"])
                category = str(row["category"])

                with root_span.start_as_current_observation(
                    as_type="span",
                    name=f"evaluation-{idx}"
                ) as evaluation_span:

                    model_answer = generate_answer(
                        answer_system_prompt,
                        question,
                        f"generate-answer-{idx}",
                        OLLAMA_MODEL
                    )

                    judge_score = judge_answer(
                        judge_prompt,
                        question,
                        expected_answer,
                        model_answer,
                        f"judge-answer-{idx}"
                    )

                    score_value = extract_score(judge_score)

                    with evaluation_span.start_as_current_observation(
                        as_type="span",
                        name="initial-calculate-severity"
                    ) as initial_severity_span:

                        severity = classify_severity(score_value)

                        initial_severity_span.update(
                            input={"score_value": score_value},
                            output={"severity": severity}
                        )

                    final_model_answer = model_answer
                    final_judge_answer = judge_score
                    final_score = score_value

                    used_fallback = False
                    mitigation_status = "original_kept"

                    fallback_answer = None
                    fallback_judge = None
                    fallback_score = None

                    if severity == "critical":

                        used_fallback = True
                        mitigation_status = "fallback_applied"

                        fallback_answer = generate_answer(
                            fallback_system_prompt,
                            question,
                            f"fallback-answer-{idx}",
                            OLLAMA_MODEL
                        )

                        fallback_judge = judge_answer(
                            judge_prompt,
                            question,
                            expected_answer,
                            fallback_answer,
                            f"fallback-judge-{idx}"
                        )

                        fallback_score = extract_score(fallback_judge)

                        # A comparacao final sera tratada na
                        # continuacao da aula.
                        # Nao substituimos a resposta original
                        # antes de validar o resultado do fallback.

                        mitigation_status = "fallback_evaluated"

                    evaluation_result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "model_answer": model_answer,
                        "judge_score": judge_score,
                        "numeric_score": score_value,
                        "category": category,
                        "severity": severity,
                        "used_fallback": used_fallback,
                        "mitigation_status": mitigation_status,
                        "fallback_answer": fallback_answer,
                        "fallback_judge": fallback_judge,
                        "fallback_score": fallback_score,
                        "final_model_answer": final_model_answer,
                        "final_judge_answer": final_judge_answer,
                        "final_score": final_score
                    }

                    evaluation_span.update(
                        input={
                            "question": question,
                            "expected_answer": expected_answer
                        },
                        output=evaluation_result,
                        metadata={
                            "example_index": int(idx),
                            "category": category
                        }
                    )

                    results.append(evaluation_result)

                    print(f"\nEXEMPLO {idx + 1}/{len(sample_df)}")
                    print("Score inicial:", score_value)
                    print("Severidade:", severity)
                    print("Fallback executado:", used_fallback)

                    if used_fallback:
                        print("Score fallback:", fallback_score)

            root_span.update(
                input={
                    "dataset_path": str(DATASET_PATH),
                    "num_examples": len(sample_df)
                },
                output={
                    "results": results,
                    "num_critical_cases": sum(
                        item["severity"] == "critical"
                        for item in results
                    ),
                    "num_fallback_attempts": sum(
                        item["used_fallback"]
                        for item in results
                    )
                }
            )

    langfuse.flush()
    return results


if __name__ == "__main__":
    try:
        results = run_batch_llm_judge_monitoring()

        print("\n===== ALURA - MONITORAMENTO E FALLBACK =====")
        print("Total de casos:", len(results))

        critical = [
            item for item in results
            if item["severity"] == "critical"
        ]

        print("Casos criticos:", len(critical))
        print("Fallbacks executados:", sum(
            item["used_fallback"] for item in results
        ))

        for item in critical:
            print("\nPergunta:", item["question"])
            print("Score inicial:", item["numeric_score"])
            print("Score fallback:", item["fallback_score"])
            print("Status:", item["mitigation_status"])

        print("\nTrace: batch-llm-judge-monitoring")
        print("Monitoramento concluido.")

    finally:
        langfuse.flush()
