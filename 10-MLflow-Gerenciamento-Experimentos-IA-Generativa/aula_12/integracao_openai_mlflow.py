import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

PROJETO = Path(__file__).resolve().parent.parent

load_dotenv(PROJETO / ".env")

BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri(
    "sqlite:///" + BANCO.as_posix()
)

mlflow.set_experiment("genai-mlflow-3.0-demo")


class GenAIModel:

    def __init__(self, model_type="gpt-4o-mini"):
        self.model_type = model_type

        chave = os.getenv("OPENAI_API_KEY")

        if not chave:
            raise RuntimeError(
                "OPENAI_API_KEY nao configurada no arquivo .env"
            )

        self.client = OpenAI(api_key=chave)

        print("GenAIModel inicializado")
        print("Modelo:", self.model_type)

    def _initialize_model(self):
        # A criacao do cliente nao faz uma chamada paga.
        # A conexao sera testada durante a geracao.
        print("Cliente OpenAI configurado.")
        return True

    def generate(self, prompt, max_tokens=100, temperature=0.7):

        resposta = self.client.chat.completions.create(
            model=self.model_type,
            messages=[
                {
                    "role": "system",
                    "content": "Voce e um assistente util e preciso."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return resposta.choices[0].message.content or ""

    def generate_with_tracking(self, prompt):

        with mlflow.start_run(
            run_name="aula_12_openai"
        ) as run:

            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("provider", "openai")

            resposta = self.generate(prompt)

            mlflow.log_metric(
                "response_length",
                len(resposta.split())
            )

            mlflow.log_dict(
                {
                    "prompt": prompt,
                    "response": resposta
                },
                "outputs/openai_response.json"
            )

            return {
                "run_id": run.info.run_id,
                "response": resposta
            }


if __name__ == "__main__":

    print("===== ALURA - AULA 12 =====")

    chave = os.getenv("OPENAI_API_KEY")

    if not chave:
        print("OpenAI API Key ainda nao configurada.")
        print("Estrutura preparada com sucesso.")
        print("Nenhuma chamada paga foi realizada.")
        print("===== AULA 12 - PREPARACAO CONCLUIDA =====")

    else:
        print("Chave encontrada no ambiente.")
        print("Integracao pronta para teste autorizado.")
        print("Nenhuma chamada paga foi realizada.")
        print("===== AULA 12 - PREPARACAO CONCLUIDA =====")
