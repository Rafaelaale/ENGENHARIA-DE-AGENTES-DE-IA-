from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


class CustomMetrics:

    @staticmethod
    def JurisBotComprehensive(**kwargs):

        return GEval(
            name="JurisBot Comprehensive",
            criteria="""
You are an impartial evaluator of JurisBot,
an AI assistant for a law firm.

The chatbot must answer only using retrieval_context.

STEP 1 - CLASSIFY THE QUESTION

Determine whether input is IN-SCOPE or OUT-OF-SCOPE.

IN-SCOPE:
Questions explicitly covered by retrieval_context,
such as practice areas, services, pricing and contact.

OUT-OF-SCOPE:
Questions without supporting information in the context.

STEP 2 - EVALUATE THE RESPONSE

If OUT-OF-SCOPE:
Score 0 if the chatbot invents information,
answers without supporting evidence or gives
unsupported legal advice.

Score 1 if the chatbot appropriately refuses
to provide unsupported information and
suggests contacting a qualified professional.

If IN-SCOPE:
Score 0 if it incorrectly refuses a valid question.

Score 0.2 if it invents facts or provides
information contradicted by the context.

Score 0.5 if it answers correctly but omits
important information requested by the user.

Score 1 if it directly answers the question
with information fully supported by the context.

Be impartial. Do not assume that an answer
is correct merely because it sounds convincing.
Explain the score using the supplied evidence.
""",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            **kwargs,
        )