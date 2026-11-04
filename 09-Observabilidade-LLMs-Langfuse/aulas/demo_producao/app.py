import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=False)

DATASET_PATH = BASE_DIR / "data" / "bitext_customer_support.csv"
JUDGE_PROMPT_PATH = BASE_DIR / "prompts" / "customer_support_answer_judge.md"

ANSWER_PROMPT_NAME = "customer_support_assistant"
ANSWER_PROMPT_LABEL = "production"
FALLBACK_PROMPT_NAME = "customer_support_assistant"
FALLBACK_PROMPT_LABEL = "fallback"

OPENAI_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OPENAI_MODEL_JUDGE = os.getenv("OLLAMA_MODEL_JUDGE", OPENAI_MODEL)

CRITICAL_SCORE_THRESHOLD = 2
N_EXAMPLES = int(os.getenv("N_EXAMPLES", "5"))
MAX_REVIEW_CASES_TO_PRINT = 2

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=180.0,
    max_retries=0
)


def load_prompt(path):
    return Path(path).read_text(encoding="utf-8-sig").strip()


def extract_judge_score(judge_output):
    match = re.search(
        r"(?im)^\s*SCORE\s*:\s*([1-5])\b",
        judge_output or ""
    )
    return int(match.group(1)) if match else None


def calculate_severity(score_value):
    if score_value is None:
        return "review"
    if score_value <= CRITICAL_SCORE_THRESHOLD:
        return "critical"
    if score_value == 3:
        return "review"
    return "ok"


def build_judge_input(question, expected_answer, generated_answer):
    return (
        f"Pergunta:\n{question}\n\n"
        f"Resposta esperada:\n{expected_answer}\n\n"
        f"Resposta gerada:\n{generated_answer}"
    )


def generate_answer(system_prompt, question, request_name, model=None):
    completion = client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        name=request_name,
        max_tokens=180 if model == OPENAI_MODEL_JUDGE else 350
    )

    answer = (completion.choices[0].message.content or "").strip()
    usage = completion.usage

    input_tokens = (
        getattr(usage, "prompt_tokens", 0) or 0
    ) if usage else 0

    output_tokens = (
        getattr(usage, "completion_tokens", 0) or 0
    ) if usage else 0

    return answer, input_tokens, output_tokens


def judge_answer(judge_prompt, question, expected_answer,
                 generated_answer, request_name):

    judge_input = build_judge_input(
        question, expected_answer, generated_answer
    )

    judge_output, input_tokens, output_tokens = generate_answer(
        judge_prompt,
        judge_input,
        request_name,
        OPENAI_MODEL_JUDGE
    )

    score_value = extract_judge_score(judge_output)
    severity = calculate_severity(score_value)

    return judge_output, score_value, severity, input_tokens, output_tokens


def evaluate_example(row, idx, answer_system_prompt,
                     fallback_system_prompt, judge_prompt, evaluation_span):

    question = str(row["instruction"])
    expected_answer = str(row["response"])
    category = str(row["category"])
    flags = str(row.get("flags", f"example-{idx}"))

    initial_answer, answer_in, answer_out = generate_answer(
        answer_system_prompt, question, f"generate-answer-{idx}"
    )

    (
        initial_judge_score,
        initial_score_value,
        initial_severity,
        judge_in,
        judge_out
    ) = judge_answer(
        judge_prompt, question, expected_answer,
        initial_answer, f"judge-answer-{idx}"
    )

    with evaluation_span.start_as_current_observation(
        as_type="span",
        name="initial-calculate-severity"
    ) as severity_span:
        severity_span.update(
            input={"score": initial_score_value},
            output={"severity": initial_severity}
        )

    final_answer = initial_answer
    final_judge_score = initial_judge_score
    final_score_value = initial_score_value

    used_fallback = False
    mitigation_status = "original_kept"

    fallback_in = 0
    fallback_out = 0
    fallback_judge_in = 0
    fallback_judge_out = 0

    if initial_severity == "critical":
        used_fallback = True

        fallback_answer, fallback_in, fallback_out = generate_answer(
            fallback_system_prompt,
            question,
            f"fallback-answer-{idx}"
        )

        (
            fallback_judge,
            fallback_score,
            _,
            fallback_judge_in,
            fallback_judge_out
        ) = judge_answer(
            judge_prompt,
            question,
            expected_answer,
            fallback_answer,
            f"fallback-judge-{idx}"
        )

        if fallback_score is not None:
            final_answer = fallback_answer
            final_judge_score = fallback_judge
            final_score_value = fallback_score
            mitigation_status = "fallback_applied"
        else:
            mitigation_status = "fallback_invalid_score"

        evaluation_span.create_event(
            name="fallback-applied",
            input={
                "initial_score": initial_score_value,
                "fallback_score": fallback_score
            },
            metadata={"mitigation_status": mitigation_status}
        )

    final_severity = calculate_severity(final_score_value)

    with evaluation_span.start_as_current_observation(
        as_type="span",
        name="final-calculate-severity"
    ) as final_span:
        final_span.update(
            input={"final_score": final_score_value},
            output={"final_severity": final_severity}
        )

    total_input_tokens = (
        answer_in + judge_in + fallback_in + fallback_judge_in
    )
    total_output_tokens = (
        answer_out + judge_out + fallback_out + fallback_judge_out
    )

    result = {
        "question": question,
        "expected_answer": expected_answer,
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "initial_judge_score": initial_judge_score,
        "final_judge_score": final_judge_score,
        "initial_score_value": initial_score_value,
        "final_score_value": final_score_value,
        "used_fallback": used_fallback,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "initial_severity": initial_severity,
        "final_severity": final_severity,
        "mitigation_status": mitigation_status,
        "category": category,
        "flags": flags
    }

    evaluation_span.update(
        input={
            "question": question,
            "expected_answer": expected_answer
        },
        output={
            "initial_answer": initial_answer,
            "final_answer": final_answer,
            "initial_judge_score": initial_judge_score,
            "final_judge_score": final_judge_score,
            "initial_score_value": initial_score_value,
            "final_score_value": final_score_value,
            "used_fallback": used_fallback,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        },
        metadata={
            "category": category,
            "flags": flags,
            "initial_severity": initial_severity,
            "final_severity": final_severity,
            "mitigation_status": mitigation_status
        }
    )

    return result


def build_summary(results):
    total_examples = len(results)
    num_fallback = sum(item["used_fallback"] for item in results)

    return {
        "num_examples": total_examples,
        "num_judge_answers": total_examples + num_fallback,
        "judge_answers_preview": [
            item["final_judge_score"] for item in results[:3]
        ],
        "total_input_tokens": sum(
            item["input_tokens"] for item in results
        ),
        "total_output_tokens": sum(
            item["output_tokens"] for item in results
        ),
        "num_fallback_applied": num_fallback
    }


def calculate_avg_score(results):
    scores = [
        item["final_score_value"]
        for item in results
        if item["final_score_value"] is not None
    ]
    return round(sum(scores) / len(scores), 2) if scores else None


def calculate_severity_counts(results):
    critical = sum(
        item["final_severity"] == "critical" for item in results
    )
    review = sum(
        item["final_severity"] in ("critical", "review")
        for item in results
    )
    return critical, review


def calculate_category_summary(results):
    counts = Counter(
        item.get("category") or "unknown" for item in results
    )
    return [
        {"category": category, "count": count}
        for category, count in counts.most_common()
    ]


def run_batch_llm_judge_pipeline(n_examples=N_EXAMPLES):
    results = []

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="customer-support-llm-pipeline"
        ) as root_span:

            with propagate_attributes(
                session_id="llm-judge-production-demo",
                user_id="demo-user-123",
                tags=[
                    "customer-support",
                    "llm-as-judge-batch",
                    OPENAI_MODEL
                ],
                metadata={
                    "environment": "develop",
                    "version": "demo-final",
                    "answer_prompt_label": ANSWER_PROMPT_LABEL,
                    "fallback_prompt_label": FALLBACK_PROMPT_LABEL,
                    "num_examples": n_examples
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
                        raise ValueError(f"Colunas ausentes: {missing}")

                    sample_df = df.head(n_examples).copy()

                    dataset_span.update(
                        input={"dataset_path": str(DATASET_PATH)},
                        output={"num_samples": len(sample_df)}
                    )

                    dataset_span.create_event(name="dataset-loaded")

                with root_span.start_as_current_observation(
                    as_type="span",
                    name="load-prompts"
                ) as prompt_span:

                    judge_prompt = load_prompt(JUDGE_PROMPT_PATH)

                    answer_prompt = langfuse.get_prompt(
                        name=ANSWER_PROMPT_NAME,
                        label=ANSWER_PROMPT_LABEL
                    ).prompt

                    fallback_prompt = langfuse.get_prompt(
                        name=FALLBACK_PROMPT_NAME,
                        label=FALLBACK_PROMPT_LABEL
                    ).prompt

                    if not isinstance(answer_prompt, str):
                        raise TypeError("Prompt production nao e texto.")
                    if not isinstance(fallback_prompt, str):
                        raise TypeError("Prompt fallback nao e texto.")

                    prompt_span.update(
                        output={
                            "production": ANSWER_PROMPT_LABEL,
                            "fallback": FALLBACK_PROMPT_LABEL
                        }
                    )

                for idx, row in sample_df.iterrows():
                    with root_span.start_as_current_observation(
                        as_type="span",
                        name=f"evaluation-{idx}"
                    ) as evaluation_span:

                        result = evaluate_example(
                            row,
                            idx,
                            answer_prompt,
                            fallback_prompt,
                            judge_prompt,
                            evaluation_span
                        )
                        results.append(result)

                with root_span.start_as_current_observation(
                    as_type="span",
                    name="aggregate-results"
                ) as aggregate_span:
                    summary = build_summary(results)
                    aggregate_span.update(output=summary)

                with root_span.start_as_current_observation(
                    as_type="span",
                    name="analyze-priority-cases"
                ) as review_span:

                    review_cases = [
                        item for item in results
                        if item["final_severity"] != "ok"
                    ]

                    review_span.update(
                        input={
                            "total_cases": len(results),
                            "review_threshold": 3,
                            "critical_threshold": CRITICAL_SCORE_THRESHOLD
                        },
                        output={
                            "review_cases_count": len(review_cases),
                            "review_cases": review_cases
                        }
                    )

                root_span.update(
                    input={
                        "dataset_path": str(DATASET_PATH),
                        "num_evaluated_cases": len(results)
                    },
                    output={
                        "summary": summary,
                        "results": results,
                        "review_cases_count": len(review_cases),
                        "review_cases": review_cases
                    }
                )
    finally:
        langfuse.flush()

    return {
        "summary": summary,
        "results": results,
        "review_cases_count": len(review_cases),
        "review_cases": review_cases
    }
