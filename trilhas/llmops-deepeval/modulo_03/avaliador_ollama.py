import asyncio
import json
import urllib.request

from deepeval.models import DeepEvalBaseLLM


class AvaliadorOllama(DeepEvalBaseLLM):

    def __init__(self):
        self.nome = "llama3.2:latest"
        super().__init__()

    def load_model(self):
        return self.nome

    def get_model_name(self):
        return self.nome

    def generate(self, prompt: str, schema=None):

        mensagens = [
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. "
                    "Follow instructions precisely. "
                    "Return valid JSON when requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": self.nome,
            "messages": mensagens,
            "stream": False,
            "options": {"temperature": 0},
        }

        if schema is not None:
            payload["format"] = schema.model_json_schema()
        else:
            payload["format"] = "json"

        dados = json.dumps(payload).encode("utf-8")

        requisicao = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=dados,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(
            requisicao, timeout=180
        ) as resposta:
            resultado = json.load(resposta)

        texto = resultado["message"]["content"].strip()

        if schema is not None:
            return schema.model_validate_json(texto)

        return texto

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(
            self.generate, prompt, schema
        )