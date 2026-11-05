import sys
from pathlib import Path

from deepeval.test_case import LLMTestCase

raiz = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(raiz))
sys.path.insert(0, str(raiz / "modulo_05"))

from custom_metrics_dag import JurisBotComprehensiveDAG
from modulo_03.avaliador_ollama import AvaliadorOllama


# Contexto controlado para saber o resultado esperado.
contexto = """
A Juris Consultoria atua em Direito Empresarial,
Direito Tributario e Direito do Consumidor.

A consulta inicial custa R$ 200,00.

O escritorio nao informa servicos de divorcio
ou Direito de Familia nesta base de conhecimento.
"""

casos = [
    (
        "Dentro do escopo - correta",
        "Quanto custa a consulta inicial?",
        "A consulta inicial custa R$ 200,00.",
    ),
    (
        "Fora do escopo - recusa",
        "O escritorio realiza divorcios?",
        "Nao possuo informacoes suficientes sobre "
        "esse servico. Recomendo consultar o escritorio.",
    ),
    (
        "Dentro do escopo - informacao incorreta",
        "Quanto custa a consulta inicial?",
        "A consulta inicial custa R$ 850,00.",
    ),
]

print("===== DAG - VERIFICANDO ESTRUTURA =====")

metrica_estrutura = JurisBotComprehensiveDAG.criar(
    threshold=0.7
)

print("GRAFO CRIADO COM SUCESSO")

print("\n===== DAG - EXECUTANDO AVALIACOES =====")

juiz = AvaliadorOllama()

for nome, pergunta, resposta in casos:

    print("\n" + "=" * 60)
    print("CASO:", nome)

    teste = LLMTestCase(
        input=pergunta,
        actual_output=resposta,
        retrieval_context=[contexto],
    )

    metrica = JurisBotComprehensiveDAG.criar(
        model=juiz,
        threshold=0.7,
        verbose_mode=True,
        async_mode=False,
    )

    metrica.measure(teste)

    print("SCORE:", metrica.score)
    print("SUCESSO:", metrica.is_successful())
    print("JUSTIFICATIVA:", metrica.reason)

print("\nDAG - TRES CASOS EXECUTADOS")