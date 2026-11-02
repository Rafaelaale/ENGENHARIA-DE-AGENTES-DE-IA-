# ==========================================================
# HUGGING FACE - AULA 10
# INTERFACE GRADIO PARA REMOCAO DE FUNDO
# ==========================================================

from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import torch

from PIL import Image, ImageOps
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "briaai/RMBG-1.4"

ARQUIVO_RESULTADO = (
    RESULTADOS / "aula10_ultimo_produto_sem_fundo.png"
)

EXEMPLO = (
    BASE / "dados" / "aula09_produto_original.png"
)

print("=" * 65)
print("AULA 10 - INTERFACE GRADIO")
print("REMOCAO AUTOMATICA DE FUNDO")
print("=" * 65)


# ==========================================================
# 2 - CARREGAR MODELO UMA UNICA VEZ
# ==========================================================

print("\n1 - CARREGANDO MODELO BRIAAI")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

pipe = pipeline(
    "image-segmentation",
    model=MODELO,
    trust_remote_code=True,
    device=-1
)

print("Modelo carregado: OK")


# ==========================================================
# 3 - FUNCAO PARA REMOVER FUNDO
# ==========================================================

def remove_background(image):

    if image is None:
        raise gr.Error(
            "Envie uma imagem para remover o fundo."
        )

    print("\n2 - PROCESSANDO NOVA IMAGEM")

    # Preparar imagem recebida do Gradio
    imagem_original = ImageOps.exif_transpose(
        image
    ).convert("RGB")

    # Usar um arquivo temporario para a pipeline
    # receber o mesmo tipo de entrada da Aula 09.
    with TemporaryDirectory() as pasta:

        caminho = Path(pasta) / "entrada.png"

        imagem_original.save(caminho)

        with torch.inference_mode():

            pillow_mask = pipe(
                str(caminho),
                return_mask=True
            )

    if not isinstance(pillow_mask, Image.Image):
        raise gr.Error(
            "O modelo nao retornou uma mascara valida."
        )

    # Preparar mascara
    pillow_mask = pillow_mask.convert("L")

    if pillow_mask.size != imagem_original.size:

        pillow_mask = pillow_mask.resize(
            imagem_original.size,
            Image.Resampling.LANCZOS
        )

    # Criar imagem com transparencia
    pillow_image = imagem_original.convert("RGBA")

    pillow_image.putalpha(
        pillow_mask
    )

    # Salvar uma copia no projeto
    pillow_image.save(
        ARQUIVO_RESULTADO,
        format="PNG"
    )

    print("Mascara: OK")
    print("Fundo removido: OK")
    print("Imagem PNG salva:")
    print(ARQUIVO_RESULTADO)

    return pillow_image


# ==========================================================
# 4 - CRIAR INTERFACE GRADIO
# ==========================================================

print("\n3 - PREPARANDO INTERFACE")

# Usar a imagem da Aula 09 como exemplo,
# caso esteja disponivel.
exemplos = (
    [[str(EXEMPLO)]]
    if EXEMPLO.exists()
    else None
)

app = gr.Interface(

    fn=remove_background,

    inputs=gr.Image(
        type="pil",
        image_mode="RGB",
        label="Imagem original",
        sources=["upload", "clipboard"]
    ),

    outputs=gr.Image(
        type="pil",
        image_mode="RGBA",
        format="png",
        label="Imagem sem fundo",
        show_download_button=True
    ),

    title="Remoção de Background de Imagens",

    description=(
        "Envie uma imagem de produto e remova "
        "o fundo automaticamente com inteligência "
        "artificial. Baixe o resultado em PNG "
        "com fundo transparente."
    ),

    examples=exemplos,

    allow_flagging="never"
)


# ==========================================================
# 5 - INICIAR APLICACAO LOCAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO INICIAL")
print("=" * 65)

print("Modelo RMBG-1.4: OK")
print("Funcao remove_background: OK")
print("Interface Gradio criada: OK")
print("Formato de saida: PNG")
print("Acesso: local e privado")

print("\nINICIANDO INTERFACE...")
print("Endereco: http://127.0.0.1:7860")
print("Para encerrar, pressione Ctrl+C no terminal.")

app.launch(
    share=False,
    inbrowser=True,
    server_name="127.0.0.1",
    server_port=7860,
    show_error=True
)


