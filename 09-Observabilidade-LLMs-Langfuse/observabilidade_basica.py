import time
import logging
from datetime import datetime

# Configuracao do monitoramento
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class MonitoramentoLLM:

    def __init__(self):
        self.historico = []

    def executar_chatbot(self, pergunta):
        inicio = time.perf_counter()

        logging.info("Iniciando processamento da pergunta")

        # Simulacao de resposta de um modelo de linguagem
        resposta = "O prazo de cancelamento e de 7 dias."

        latencia = time.perf_counter() - inicio

        registro = {
            "timestamp": datetime.now().isoformat(),
            "pergunta": pergunta,
            "resposta": resposta,
            "latencia": round(latencia, 6),
            "modelo": "LLM-simulado",
            "tokens_entrada": len(pergunta.split()),
            "tokens_saida": len(resposta.split())
        }

        self.historico.append(registro)

        logging.info("Resposta gerada com sucesso")
        logging.info("Latencia: %.6f segundos", latencia)

        return registro

    def exibir_metricas(self):
        print("\n===== METRICAS DE OBSERVABILIDADE =====")

        for registro in self.historico:
            print(f"Pergunta: {registro['pergunta']}")
            print(f"Resposta: {registro['resposta']}")
            print(f"Modelo: {registro['modelo']}")
            print(f"Latencia: {registro['latencia']} segundos")
            print(f"Palavras entrada: {registro['tokens_entrada']}")
            print(f"Palavras saida: {registro['tokens_saida']}")
            print("-" * 45)


if __name__ == "__main__":

    monitor = MonitoramentoLLM()

    perguntas = [
        "Qual e o prazo de cancelamento?",
        "Como funciona o atendimento?",
        "Posso solicitar reembolso?"
    ]

    for pergunta in perguntas:
        monitor.executar_chatbot(pergunta)

    monitor.exibir_metricas()
