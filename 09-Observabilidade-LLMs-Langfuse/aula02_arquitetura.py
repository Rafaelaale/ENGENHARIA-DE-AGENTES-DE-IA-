import time
import json
import uuid
from datetime import datetime
from pathlib import Path


class ObservabilidadeLLM:

    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.observacoes = []
        self.inicio = time.perf_counter()

    def registrar_evento(self, nome, detalhes):
        evento = {
            "tipo": "EVENT",
            "nome": nome,
            "timestamp": datetime.now().isoformat(),
            "detalhes": detalhes
        }

        self.observacoes.append(evento)
        print(f"[EVENT] {nome}")

    def executar_span(self, nome, operacao):
        inicio = time.perf_counter()

        resultado = operacao()

        duracao = time.perf_counter() - inicio

        span = {
            "tipo": "SPAN",
            "nome": nome,
            "duracao_segundos": round(duracao, 6),
            "resultado": resultado
        }

        self.observacoes.append(span)

        print(f"[SPAN] {nome} | {duracao:.6f}s")

        return resultado

    def executar_generation(self, prompt):
        inicio = time.perf_counter()

        # Simulacao de chamada ao modelo
        time.sleep(0.2)

        resposta = (
            "Observabilidade permite acompanhar "
            "o comportamento de aplicacoes com LLMs."
        )

        duracao = time.perf_counter() - inicio

        generation = {
            "tipo": "GENERATION",
            "modelo": "LLM-simulado",
            "prompt": prompt,
            "resposta": resposta,
            "duracao_segundos": round(duracao, 6),
            "palavras_entrada": len(prompt.split()),
            "palavras_saida": len(resposta.split())
        }

        self.observacoes.append(generation)

        print(f"[GENERATION] Resposta gerada | {duracao:.4f}s")

        return resposta

    def executar_pipeline(self, pergunta):
        print("\n===== INICIANDO TRACE =====")
        print(f"Trace ID: {self.trace_id}")

        self.registrar_evento(
            "Pergunta recebida",
            {"pergunta": pergunta}
        )

        contexto = self.executar_span(
            "Preparacao do contexto",
            lambda: "Contexto educativo sobre LLMs"
        )

        prompt = (
            f"Contexto: {contexto}\n"
            f"Pergunta: {pergunta}\n"
            "Responda de forma clara."
        )

        resposta = self.executar_generation(prompt)

        self.registrar_evento(
            "Resposta pronta",
            {"status": "sucesso"}
        )

        duracao_total = time.perf_counter() - self.inicio

        trace = {
            "trace_id": self.trace_id,
            "tipo": "TRACE",
            "pergunta": pergunta,
            "resposta": resposta,
            "duracao_total_segundos": round(duracao_total, 6),
            "observacoes": self.observacoes
        }

        arquivo = Path("trace_aula02.json")

        arquivo.write_text(
            json.dumps(trace, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

        print("\n===== RESULTADO FINAL =====")
        print(f"Resposta: {resposta}")
        print(f"Duracao total: {duracao_total:.4f}s")
        print(f"Observacoes: {len(self.observacoes)}")
        print(f"Arquivo gerado: {arquivo.resolve()}")


if __name__ == "__main__":

    monitor = ObservabilidadeLLM()

    monitor.executar_pipeline(
        "O que e observabilidade para LLMs?"
    )
