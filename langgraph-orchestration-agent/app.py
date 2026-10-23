from uuid import uuid4

import gradio as gr
from new_backend import graph


def generate_essay(topic: str, max_revisions: int):
    if not topic or not topic.strip():
        return "Digite o tema da redação antes de gerar."

    initial_state = {
        "task": topic.strip(),
        "max_revisions": int(max_revisions),
        "revision_number": 0,
        "plan": "",
        "draft": "",
        "critique": "",
        "content": [],
    }

    try:
        result = graph.invoke(
            initial_state,
            {"configurable": {"thread_id": str(uuid4())}},
        )
        return result.get("draft", "O agente não retornou uma redação.")
    except Exception as e:
        return f"Erro ao gerar a redação: {e}"


with gr.Blocks() as demo:
    gr.Markdown("# Gerador de Redações")
    gr.Markdown("Informe o tema e gere uma redação em português.")

    topic = gr.Textbox(
        label="Tema da redação",
        placeholder="Ex.: A importância da educação para o futuro",
    )
    revisions = gr.Slider(
        minimum=0,
        maximum=3,
        step=1,
        value=1,
        label="Número de revisões",
    )
    button = gr.Button("Gerar redação", variant="primary")
    output = gr.Textbox(label="Redação final", lines=20)

    button.click(
        fn=generate_essay,
        inputs=[topic, revisions],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(share=False)
