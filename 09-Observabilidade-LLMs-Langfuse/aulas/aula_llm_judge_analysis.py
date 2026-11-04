import os
import re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.openai import OpenAI
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env', override=False)
DATASET_PATH = BASE_DIR / 'data' / 'bitext_customer_support.csv'
JUDGE_PROMPT_PATH = 'prompts/customer_support_answer_judge.md'
ANSWER_PROMPT_NAME = 'customer_support_assistant'
ANSWER_PROMPT_LABEL = 'production'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:latest')
OLLAMA_MODEL_JUDGE = os.getenv('OLLAMA_MODEL_JUDGE', OLLAMA_MODEL)
N_EXAMPLES = 5
langfuse = get_client()
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

def load_prompt(path):
    arquivo = BASE_DIR / path
    if not arquivo.is_file():
        raise FileNotFoundError(f'Prompt nao encontrado: {arquivo}')
    return arquivo.read_text(encoding='utf-8-sig').strip()

def extract_score(judge_answer):
    match = re.search('(?im)^\\s*SCORE\\s*:\\s*([1-5])\\s*$', judge_answer)
    if match:
        return int(match.group(1))
    return None

def run_batch_llm_judge_analysis():
    results = []
    with langfuse.start_as_current_observation(as_type='span', name='batch-llm-judge-analysis') as root_span:
        with propagate_attributes(session_id='llm-judge-analysis-session-123', user_id='id-123', tags=['customer-support', 'llm-as-judge-analysis'], metadata={'environment': 'develop', 'version': 'v1', 'dataset_path': str(DATASET_PATH), 'judge_prompt_path': JUDGE_PROMPT_PATH, 'answer_prompt_name': ANSWER_PROMPT_NAME, 'answer_prompt_label': ANSWER_PROMPT_LABEL}):
            with root_span.start_as_current_observation(as_type='span', name='load-dataset') as dataset_span:
                df = pd.read_csv(DATASET_PATH)
                required_columns = {'instruction', 'response', 'category'}
                missing = required_columns - set(df.columns)
                if missing:
                    raise ValueError(f'Colunas ausentes: {sorted(missing)}')
                sample_df = df.head(N_EXAMPLES).copy()
                dataset_span.update(input={'dataset_path': str(DATASET_PATH)}, output={'num_samples': len(sample_df)})
                dataset_span.create_event(name='dataset-loaded', metadata={'num_samples': len(sample_df)})
            with root_span.start_as_current_observation(as_type='span', name='load-prompts') as prompt_load_span:
                judge_prompt = load_prompt(JUDGE_PROMPT_PATH)
                prompt_client = langfuse.get_prompt(name=ANSWER_PROMPT_NAME, label=ANSWER_PROMPT_LABEL)
                answer_system_prompt = prompt_client.prompt
                if not isinstance(answer_system_prompt, str):
                    raise TypeError('Prompt do assistente deve ser texto.')
                prompt_load_span.update(output={'judge_prompt_path': JUDGE_PROMPT_PATH, 'answer_prompt_name': ANSWER_PROMPT_NAME})
            for idx, row in sample_df.iterrows():
                question = str(row['instruction'])
                expected_answer = str(row['response'])
                category = str(row['category'])
                with root_span.start_as_current_observation(as_type='span', name=f'evaluation-{idx}') as evaluation_span:
                    completion_answer = client.chat.completions.create(model=OLLAMA_MODEL, messages=[{'role': 'system', 'content': answer_system_prompt}, {'role': 'user', 'content': question}], name=f'generate-answer-{idx}')
                    model_answer = (completion_answer.choices[0].message.content or '').strip()
                    judge_input = f'\nPergunta: {question}\n\nResposta esperada: {expected_answer}\n\nResposta gerada: {model_answer}\n'
                    completion_judge = client.chat.completions.create(model=OLLAMA_MODEL_JUDGE, messages=[{'role': 'system', 'content': judge_prompt}, {'role': 'user', 'content': judge_input}], name=f'judge-answer-{idx}')
                    judge_score = (completion_judge.choices[0].message.content or '').strip()
                    numeric_score = extract_score(judge_score)
                    evaluation_result = {'question': question, 'expected_answer': expected_answer, 'model_answer': model_answer, 'judge_score': judge_score, 'numeric_score': numeric_score, 'category': category}
                    evaluation_span.update(input={'question': question, 'expected_answer': expected_answer}, output={'model_answer': model_answer, 'judge_score': judge_score, 'numeric_score': numeric_score}, metadata={'example_index': int(idx), 'category': category})
                    results.append(evaluation_result)
                    print(f'\nExemplo {idx + 1}/{len(sample_df)}')
                    print('Categoria:', category)
                    print('Nota:', numeric_score)
                    print('Julgamento:', judge_score)
            with root_span.start_as_current_observation(as_type='span', name='analyze-priority-cases') as review_span:
                review_cases = []
                threshold = 3
                for item in results:
                    score_val = item.get('numeric_score')
                    if score_val is None:
                        score_val = extract_score(item.get('judge_score', ''))
                    if score_val is None or score_val <= threshold:
                        review_cases.append(item)
                review_span.update(input={'total_cases': len(results), 'threshold': str(threshold)}, output={'num_review_cases': len(review_cases), 'review_cases': review_cases})
            root_span.update(input={'dataset_path': str(DATASET_PATH), 'num_examples': N_EXAMPLES}, output={'results': results, 'num_review_cases': len(review_cases), 'review_cases': review_cases})
            valid_scores = [result['numeric_score'] for result in results if result['numeric_score'] is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
    langfuse.flush()
    return {'results': results, 'review_cases': review_cases}
if __name__ == '__main__':
    try:
        resultado = run_batch_llm_judge_analysis()
        print('\n===== ALURA - ANALISE DE PRIORIDADE =====')
        print('Total de casos:', len(resultado['results']))
        print('Casos para revisao:', len(resultado['review_cases']))
        print('Threshold: 3')
        for indice, item in enumerate(resultado['review_cases'], start=1):
            print(f'\nCASO PRIORITARIO {indice}')
            print('Pergunta:', item['question'])
            print('Nota:', item.get('numeric_score'))
            print('Avaliacao do juiz:', item['judge_score'])
        print('\nTrace: batch-llm-judge-analysis')
        print('Analise de prioridade concluida.')
    finally:
        langfuse.flush()
