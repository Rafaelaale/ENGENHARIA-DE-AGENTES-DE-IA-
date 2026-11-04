import json
from app import (
    MAX_REVIEW_CASES_TO_PRINT,
    N_EXAMPLES,
    calculate_avg_score,
    calculate_category_summary,
    calculate_severity_counts,
    run_batch_llm_judge_pipeline
)


def run_final_demo():
    batch_result = run_batch_llm_judge_pipeline(N_EXAMPLES)

    summary = batch_result["summary"]
    results = batch_result["results"]
    review_cases = batch_result["review_cases"]

    print("\n===== ALURA - RESUMO FINAL =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    avg_score = calculate_avg_score(results)
    num_critical, num_review = calculate_severity_counts(results)

    print("\n===== INTERPRETACAO =====")
    print("Score medio:", avg_score)
    print("Total de exemplos:", summary["num_examples"])
    print("Casos para revisao:", num_review)
    print("Casos criticos:", num_critical)
    print("Fallbacks aplicados:", summary["num_fallback_applied"])
    print("Tokens de entrada:", summary["total_input_tokens"])
    print("Tokens de saida:", summary["total_output_tokens"])

    if avg_score is None:
        print("Nenhum score numerico valido foi obtido.")
    elif avg_score < 3:
        print("ATENCAO: qualidade media baixa.")
    else:
        print("Media geral >= 3. Verificar tambem casos criticos.")

    if num_critical:
        print("Existem casos criticos para analise manual.")
    else:
        print("Nenhum caso critico identificado.")

    print("\n===== CATEGORIAS =====")
    for item in calculate_category_summary(results)[:3]:
        print(f"{item['category']}: {item['count']} caso(s)")

    print("\n===== REVISAO PRIORITARIA =====")
    print("Total para revisao:", len(review_cases))

    for item in review_cases[:MAX_REVIEW_CASES_TO_PRINT]:
        print("\nPergunta:", item["question"])
        print("Score inicial:", item["initial_score_value"])
        print("Score final:", item["final_score_value"])
        print("Fallback:", item["used_fallback"])
        print("Status:", item["mitigation_status"])
        print("Severidade final:", item["final_severity"])

    print("\nTrace: customer-support-llm-pipeline")
    print("Demonstracao final concluida.")


if __name__ == "__main__":
    run_final_demo()
