import inspect

from deepeval.metrics import DAGMetric
from deepeval.test_case import LLMTestCaseParams

try:
    from deepeval.metrics.dag import (
        DeepAcyclicGraph,
        NonBinaryJudgementNode,
        VerdictNode,
    )
except ImportError:
    from deepeval.metrics.dag.graph import DeepAcyclicGraph
    from deepeval.metrics.dag.nodes import (
        NonBinaryJudgementNode,
        VerdictNode,
    )


class JurisBotComprehensiveDAG:

    @staticmethod
    def criar(**kwargs):

        recusa_correta = (
            "A resposta reconhece que nao ha informacao "
            "suficiente no contexto, nao inventa fatos "
            "e orienta o usuario a consultar o escritorio."
        )

        precisao = (
            "Classifique a resposta conforme o contexto. "
            "Accurate and Complete: correta e completa. "
            "Partially Correct: correta mas incompleta. "
            "Inaccurate: contem fatos inventados ou incorretos."
        )

        recusa_indevida = (
            "A pergunta pode ser respondida pelo contexto. "
            "O chatbot respondeu ou recusou indevidamente? "
            "Classifique como Answered ou Refused."
        )

        escopo = (
            "Verifique se a pergunta pode ser respondida "
            "usando SOMENTE o retrieval_context. "
            "Classifique como In-Scope ou Out-of-Scope. "
            "Nao use conhecimento externo."
        )

        antigo = (
            "children" in
            inspect.signature(
                NonBinaryJudgementNode
            ).parameters
        )

        if antigo:

            fora = NonBinaryJudgementNode(
                label="Out-of-Scope Refusal Quality",
                criteria=recusa_correta,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT
                ],
                children=[
                    VerdictNode(
                        verdict="Correct Refusal",
                        score=10
                    ),
                    VerdictNode(
                        verdict="Incorrect Refusal",
                        score=0
                    ),
                ],
            )

            qualidade = NonBinaryJudgementNode(
                label="In-Scope Answer Quality",
                criteria=precisao,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                children=[
                    VerdictNode(
                        verdict="Accurate and Complete",
                        score=10
                    ),
                    VerdictNode(
                        verdict="Partially Correct",
                        score=5
                    ),
                    VerdictNode(
                        verdict="Inaccurate",
                        score=2
                    ),
                ],
            )

            dentro = NonBinaryJudgementNode(
                label="In-Scope Refusal Check",
                criteria=recusa_indevida,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT
                ],
                children=[
                    VerdictNode(
                        verdict="Answered",
                        child=qualidade
                    ),
                    VerdictNode(
                        verdict="Refused",
                        score=0
                    ),
                ],
            )

            raiz = NonBinaryJudgementNode(
                label="Scope Classification",
                criteria=escopo,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
                children=[
                    VerdictNode(
                        verdict="In-Scope",
                        child=dentro
                    ),
                    VerdictNode(
                        verdict="Out-of-Scope",
                        child=fora
                    ),
                ],
            )

        else:

            fora = NonBinaryJudgementNode(
                label="Out-of-Scope Refusal Quality",
                criteria=(
                    "Classifique a resposta fora de escopo "
                    "como Correct Refusal se ela reconhecer "
                    "o limite do contexto e orientar contato "
                    "com o escritorio; caso contrario, "
                    "Incorrect Refusal."
                ),
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT
                ],
            )

            fora.add_verdict(
                verdict="Correct Refusal",
                score=10
            )
            fora.add_verdict(
                verdict="Incorrect Refusal",
                score=0
            )

            qualidade = NonBinaryJudgementNode(
                label="In-Scope Answer Quality",
                criteria=precisao,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
            )

            qualidade.add_verdict(
                verdict="Accurate and Complete",
                score=10
            )
            qualidade.add_verdict(
                verdict="Partially Correct",
                score=5
            )
            qualidade.add_verdict(
                verdict="Inaccurate",
                score=2
            )

            dentro = NonBinaryJudgementNode(
                label="In-Scope Refusal Check",
                criteria=recusa_indevida,
                evaluation_params=[
                    LLMTestCaseParams.ACTUAL_OUTPUT
                ],
            )

            dentro.add_verdict(
                verdict="Answered",
                then=qualidade
            )
            dentro.add_verdict(
                verdict="Refused",
                score=0
            )

            raiz = NonBinaryJudgementNode(
                label="Scope Classification",
                criteria=escopo,
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.RETRIEVAL_CONTEXT,
                ],
            )

            raiz.add_verdict(
                verdict="In-Scope",
                then=dentro
            )
            raiz.add_verdict(
                verdict="Out-of-Scope",
                then=fora
            )

        grafo = DeepAcyclicGraph(
            root_nodes=[raiz]
        )

        return DAGMetric(
            name="JurisBot Comprehensive DAG",
            dag=grafo,
            **kwargs,
        )