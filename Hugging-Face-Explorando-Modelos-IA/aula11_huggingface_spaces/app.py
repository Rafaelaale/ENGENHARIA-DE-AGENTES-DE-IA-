# ==========================================================
# HUGGING FACE - AULA 11
# APLICACAO DE REMOCAO DE FUNDO NO SPACES
# ==========================================================

import gradio as gr
import torch

from PIL import Image, ImageOps
from transformers import pipeline


MODELO = "briaai/RMBG-1.4"

print("Carregando modelo de segmentacao...")

torch.set_num_threads(4)

# Carregar modelo uma unica vez
pipe = pipeline(
    "image-segmentation",
    model=MODELO,
    trust_remote_code=True,
    device=-1
)

print("Modelo carregado com sucesso!")


# ==========================================================
# FUNCAO DE REMOCAO DE FUNDO
# ==========================================================

def remove_background(image):

    if image is None:
        raise gr.Error(
            "Envie uma imagem para remover o fundo."
        )

    imagem_original = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    # Segmentar o produto
    with torch.inference_mode():

        pillow_mask = pipe(
            imagem_original,
            return_mask=True
        )

    if not isinstance(pillow_mask, Image.Image):
        raise gr.Error(
            "Nao foi possivel gerar a mascara."
        )

    pillow_mask = pillow_mask.convert("L")

    if pillow_mask.size != imagem_original.size:

        pillow_mask = pillow_mask.resize(
            imagem_original.size,
            Image.Resampling.LANCZOS
        )

    # Aplicar mascara no canal alfa
    pillow_image = imagem_original.convert("RGBA")

    pillow_image.putalpha(
        pillow_mask
    )

    return pillow_image


# ==========================================================
# INTERFACE GRADIO
# ==========================================================

app = gr.Interface(

    fn=remove_background,

    inputs=gr.Image(
        type="pil",
        image_mode="RGB",
        sources=["upload", "clipboard"],
        label="Imagem original"
    ),

    outputs=gr.Image(
        type="pil",
        image_mode="RGBA",
        format="png",
        show_download_button=True,
        label="Imagem sem fundo"
    ),

    title="Remoção de Background com IA",

    description=(
        "Envie uma imagem de produto. "
        "A inteligência artificial identifica "
        "o objeto e remove o fundo automaticamente. "
        "Baixe o resultado em PNG transparente."
    ),

    allow_flagging="never"
)


# ==========================================================
# INICIALIZAR APLICACAO
# ==========================================================

if __name__ == "__main__":

    print("Iniciando interface Gradio...")

    # No Spaces nao precisamos de share=True.
    # O proprio Spaces fornece o endereco.

    app.launch()
