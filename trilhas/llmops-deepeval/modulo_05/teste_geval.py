import sys
from pathlib import Path

from deepeval.test_case import LLMTestCase
from custom_metrics_geval import CustomMetrics

raiz = Path(__file__).resolve().parent.parent
pasta_modelo = raiz / "modulo_03"

sys.path.insert(0, str(pasta_modelo))

from avaliador_ollama import AvaliadorOllama

arquivo_contexto = raiz / "data" / "context.txt"

if not arquivo_contexto.exists():
    raise FileNotFoundError(
        "Arquivo data/context.txt nao encontrado"
    )

contexto = arquivo_contexto.read_text(
    encoding="utf-8"
)

if not contexto.strip():
    raise RuntimeError("Contexto vazio")

juiz = AvaliadorOllama()

casos = [
    {
        "nome": "Pergunta dentro do escopo",
        "input": (
            "Quais servicos a Juris Consultoria oferece?"
        ),
        "actual_output": (
            "A Juris Consultoria oferece consulta inicial, "
            "consultoria mensal e atuacao em acoes judiciais."
        ),
    },
    {
        "nome": "Pergunta fora do escopo",
        "input": (
            "A Juris Consultoria realiza divorcios?"
        ),
        "actual_output": (
            "Nao encontrei informacoes no contexto "
            "sobre atendimento em divorcios. "
            "Recomendo entrar em contato com o escritorio."
        ),
    },
]

erros = 0

for caso in casos:

    print("\n" + "=" * 60)
    print("CASO:", caso["nome"])

    test_case = LLMTestCase(
        input=caso["input"],
        actual_output=caso["actual_output"],
        retrieval_context=[contexto],
    )

    metrica = CustomMetrics.JurisBotComprehensive(
        model=juiz,
        threshold=0.7,
        verbose_mode=True,
        async_mode=False,
    )

    try:
        metrica.measure(test_case)

        print("SCORE:", metrica.score)
        print("SUCESSO:", metrica.is_successful())
        print("JUSTIFICATIVA:", metrica.reason)

    except Exception as erro:
        erros += 1
        print("ERRO:", type(erro).__name__, str(erro))

print("\n===== RESUMO G-EVAL =====")
print("Casos avaliados:", len(casos))
print("Erros de execucao:", erros)

if erros:
    raise SystemExit(1)

print("AVALIACAO EXECUTADA")