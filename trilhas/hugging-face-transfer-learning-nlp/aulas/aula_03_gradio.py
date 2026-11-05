import gradio as gr
from transformers import pipeline

print("CARREGANDO MODELO HUGGING FACE...")

classificador = pipeline(
    "zero-shot-classification",
    model="Mel-Iza0/zero-shot",
    tokenizer="Mel-Iza0/zero-shot",
    framework="pt",
    device=-1
)

print("MODELO CARREGADO COM SUCESSO")

def mostrar_resultado(texto):
    if not texto or not texto.strip():
        return "Digite uma redacao."

    resultado = classificador(
        texto,
        candidate_labels=[str(i) for i in range(11)]
    )

    return resultado["labels"][0]

app = gr.Interface(
    fn=mostrar_resultado,
    inputs=gr.Textbox(
        lines=12,
        label="Digite sua redacao"
    ),
    outputs=gr.Textbox(
        label="Classificacao de 0 a 10"
    ),
    title="Avaliador Experimental de Redacoes",
    description=(
        "Aplicacao demonstrativa. "
        "A classificacao ainda nao e uma nota escolar validada."
    )
)

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False
    )