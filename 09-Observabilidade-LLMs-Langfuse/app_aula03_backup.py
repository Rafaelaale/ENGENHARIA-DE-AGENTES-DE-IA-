import os
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.openai import OpenAI

load_dotenv()

modelo = os.getenv("OLLAMA_MODEL")

langfuse = get_client()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


def ask_llm(question):

    with langfuse.start_as_current_observation(
        as_type="span",
        name="chat-request",
        input={"question": question}
    ) as root_span:

        with root_span.start_as_current_observation(
            as_type="event",
            name="input-received"
        ):
            pass

        completion = client.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responda perguntas de forma "
                        "clara, educativa e objetiva."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            name="ollama-model"
        )

        answer = completion.choices[0].message.content

        root_span.update(output={"answer": answer})

    langfuse.flush()

    return answer


if __name__ == "__main__":

    print("\n===== LANGFUSE + OLLAMA =====")
    print(f"Modelo: {modelo}")

    try:
        pergunta = input("\nPergunte algo: ").strip()

        if pergunta:
            resposta = ask_llm(pergunta)

            print("\n===== RESPOSTA =====")
            print(resposta)
            print("\nExecucao concluida.")

        else:
            print("Nenhuma pergunta informada.")

    except Exception as erro:
        print(f"\nERRO: {type(erro).__name__}: {erro}")

    finally:
        langfuse.flush()
