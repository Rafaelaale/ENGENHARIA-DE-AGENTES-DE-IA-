# ==========================================================
# HUGGING FACE - AULA 09
# REMOCAO DE FUNDO DE IMAGENS DE PRODUTOS
# ==========================================================

from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps, ImageDraw
from transformers import pipeline


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "briaai/RMBG-1.4"

ORIGINAL = DADOS / "aula09_produto_original.png"

MASCARA = RESULTADOS / "aula09_mascara_produto.png"

SEM_FUNDO = RESULTADOS / "aula09_produto_sem_fundo.png"

COMPARACAO = RESULTADOS / "aula09_comparacao.png"

print("=" * 65)
print("AULA 09 - REMOCAO DE FUNDO DE PRODUTOS")
print("=" * 65)


# ==========================================================
# 2 - RECEBER IMAGEM ESCOLHIDA
# ==========================================================

print("\n1 - CARREGANDO FOTO DO PRODUTO")

if len(sys.argv) < 2:
    raise ValueError(
        "Informe o caminho de uma imagem."
    )

caminho_imagem = Path(sys.argv[1])

if not caminho_imagem.is_file():
    raise FileNotFoundError(
        f"Imagem nao encontrada: {caminho_imagem}"
    )

with Image.open(caminho_imagem) as imagem:
    imagem_original = ImageOps.exif_transpose(
        imagem
    ).convert("RGB")

imagem_original.save(ORIGINAL)

print("Imagem selecionada:", caminho_imagem)
print("Imagem salva:", ORIGINAL)
print("Dimensoes:", imagem_original.size)


# ==========================================================
# 3 - CARREGAR MODELO DE SEGMENTACAO
# ==========================================================

print("\n2 - CARREGANDO MODELO BRIAAI")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

pipe = pipeline(
    "image-segmentation",
    model=MODELO,
    trust_remote_code=True,
    device=-1
)

print("Modelo:", MODELO)
print("Pipeline carregada: OK")


# ==========================================================
# 4 - GERAR MASCARA
# ==========================================================

print("\n3 - GERANDO MASCARA DO PRODUTO")

with torch.inference_mode():

    pillow_mask = pipe(
        str(ORIGINAL),
        return_mask=True
    )

if not isinstance(pillow_mask, Image.Image):
    raise TypeError(
        "A pipeline nao retornou uma imagem PIL."
    )

pillow_mask = pillow_mask.convert("L")

if pillow_mask.size != imagem_original.size:

    pillow_mask = pillow_mask.resize(
        imagem_original.size,
        Image.Resampling.LANCZOS
    )

pillow_mask.save(MASCARA)

print("Mascara gerada com sucesso!")
print("Arquivo:", MASCARA)


# ==========================================================
# 5 - REMOVER FUNDO DO PRODUTO
# ==========================================================

print("\n4 - REMOVENDO FUNDO")

pillow_image = imagem_original.convert("RGBA")

pillow_image.putalpha(
    pillow_mask
)

pillow_image.save(
    SEM_FUNDO,
    format="PNG"
)

print("Produto com fundo transparente:")
print(SEM_FUNDO)


# ==========================================================
# 6 - CRIAR PAINEL COMPARATIVO
# ==========================================================

print("\n5 - CRIANDO COMPARACAO VISUAL")

TAMANHO = (450, 450)

def ajustar(imagem):

    resultado = imagem.copy()
    resultado.thumbnail(
        TAMANHO,
        Image.Resampling.LANCZOS
    )

    painel = Image.new(
        "RGB",
        TAMANHO,
        "white"
    )

    x = (450 - resultado.width) // 2
    y = (450 - resultado.height) // 2

    painel.paste(
        resultado.convert("RGB"),
        (x, y)
    )

    return painel


painel_original = ajustar(
    imagem_original
)

painel_mascara = ajustar(
    pillow_mask
)


# Fundo quadriculado para visualizar
# transparencia do PNG
quadriculado = Image.new(
    "RGBA",
    imagem_original.size,
    "white"
)

desenho = ImageDraw.Draw(quadriculado)

passo = 40

for y in range(
    0,
    imagem_original.height,
    passo
):
    for x in range(
        0,
        imagem_original.width,
        passo
    ):

        if ((x // passo) + (y // passo)) % 2:

            desenho.rectangle(
                [
                    x,
                    y,
                    min(x + passo, imagem_original.width),
                    min(y + passo, imagem_original.height)
                ],
                fill=(220, 220, 220, 255)
            )

painel_transparente = (
    Image.alpha_composite(
        quadriculado,
        pillow_image
    )
)

painel_sem_fundo = ajustar(
    painel_transparente
)


# Montar comparacao com titulos
comparacao = Image.new(
    "RGB",
    (1350, 490),
    "white"
)

comparacao.paste(
    painel_original,
    (0, 40)
)

comparacao.paste(
    painel_mascara,
    (450, 40)
)

comparacao.paste(
    painel_sem_fundo,
    (900, 40)
)

desenho = ImageDraw.Draw(comparacao)

desenho.text(
    (15, 15),
    "IMAGEM ORIGINAL",
    fill="black"
)

desenho.text(
    (465, 15),
    "MASCARA",
    fill="black"
)

desenho.text(
    (915, 15),
    "PRODUTO SEM FUNDO",
    fill="black"
)

comparacao.save(
    COMPARACAO
)

print("Comparacao criada:")
print(COMPARACAO)


# ==========================================================
# 7 - VALIDACAO FINAL
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert ORIGINAL.exists()
assert MASCARA.exists()
assert SEM_FUNDO.exists()
assert COMPARACAO.exists()

assert pillow_mask.size == imagem_original.size

assert pillow_image.mode == "RGBA"

alpha_min, alpha_max = (
    pillow_image
    .getchannel("A")
    .getextrema()
)

assert alpha_min < 255, (
    "Nenhuma area transparente foi criada."
)

assert alpha_max > 0, (
    "A imagem resultante ficou vazia."
)

with Image.open(SEM_FUNDO) as verificacao:

    assert verificacao.mode == "RGBA"
    assert verificacao.size == imagem_original.size

print("Imagem original: OK")
print("Modelo RMBG-1.4: OK")
print("Mascara de segmentacao: OK")
print("Imagem PNG transparente: OK")
print("Dimensoes preservadas: OK")
print("Painel comparativo: OK")

print("\nARQUIVOS GERADOS:")
print(ORIGINAL)
print(MASCARA)
print(SEM_FUNDO)
print(COMPARACAO)

print("\nAULA 09 EXECUTADA COM SUCESSO!")
