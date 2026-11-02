# ==========================================================
# HUGGING FACE - AULA 08
# REMOVENDO FUNDO DE IMAGENS COM IA
# MODELO BRIAAI/RMBG-1.4
# ==========================================================

from pathlib import Path
from urllib.request import Request, urlopen
from io import BytesIO

from PIL import Image
from transformers import pipeline
import torch


# ==========================================================
# 1 - CAMINHOS E CONFIGURACOES
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DADOS.mkdir(parents=True, exist_ok=True)
RESULTADOS.mkdir(parents=True, exist_ok=True)

MODELO = "briaai/RMBG-1.4"

ARQUIVO_ORIGINAL = (
    DADOS / "aula08_girafa_original.png"
)

ARQUIVO_MASCARA = (
    RESULTADOS / "aula08_mascara_girafa.png"
)

ARQUIVO_SEM_FUNDO = (
    RESULTADOS / "aula08_girafa_sem_fundo.png"
)

ARQUIVO_COMPARACAO = (
    RESULTADOS / "aula08_comparacao.png"
)

print("=" * 65)
print("AULA 08 - REMOCAO DE FUNDO COM IA")
print("=" * 65)


# ==========================================================
# 2 - OBTER IMAGEM ORIGINAL
# ==========================================================

print("\n1 - CARREGANDO IMAGEM DA GIRAFA")

URLS = [
    "https://farm5.staticflickr.com/4007/4322154488_997e69e4cf_z.jpg",
    "https://cdn1.gnarususercontent.com.br/1/795715/1dc281ef-6dbc-4cb1-9457-2814050df776.png"
]

if ARQUIVO_ORIGINAL.exists():

    print("Usando imagem ja salva.")

else:

    imagem_baixada = False

    for url in URLS:

        try:
            print("Tentando baixar imagem...")

            req = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            with urlopen(req, timeout=45) as resposta:
                conteudo = resposta.read()

            imagem = Image.open(
                BytesIO(conteudo)
            ).convert("RGB")

            imagem.save(
                ARQUIVO_ORIGINAL
            )

            imagem_baixada = True

            print("Imagem baixada com sucesso!")
            break

        except Exception as erro:
            print(
                "Falha nesta fonte:",
                str(erro)
            )

    if not imagem_baixada:
        raise RuntimeError(
            "Nao foi possivel baixar a imagem "
            "de exemplo da girafa."
        )

imagem_original = Image.open(
    ARQUIVO_ORIGINAL
).convert("RGB")

print("Arquivo:", ARQUIVO_ORIGINAL)
print("Dimensoes:", imagem_original.size)
print("Formato:", imagem_original.format)


# ==========================================================
# 3 - CARREGAR MODELO BRIAAI
# ==========================================================

print("\n2 - CARREGANDO MODELO DE SEGMENTACAO")

print("Modelo:", MODELO)
print("Dispositivo: CPU")

torch.set_num_threads(
    min(4, torch.get_num_threads())
)

pipe = pipeline(
    "image-segmentation",
    model=MODELO,
    trust_remote_code=True,
    device=-1
)

print("\nMODELO CARREGADO COM SUCESSO!")


# ==========================================================
# 4 - GERAR MASCARA DE SEGMENTACAO
# ==========================================================

print("\n3 - GERANDO MASCARA")

with torch.inference_mode():

    pillow_mask = pipe(
        str(ARQUIVO_ORIGINAL),
        return_mask=True
    )

assert isinstance(
    pillow_mask,
    Image.Image
)

pillow_mask = pillow_mask.convert("L")

assert pillow_mask.size == imagem_original.size

pillow_mask.save(
    ARQUIVO_MASCARA
)

print("Mascara salva:")
print(ARQUIVO_MASCARA)


# ==========================================================
# 5 - REMOVER FUNDO DA IMAGEM
# ==========================================================

print("\n4 - REMOVENDO FUNDO DA GIRAfA")

# Aplicar a mascara gerada no canal alfa.
# Isso evita executar a IA uma segunda vez.

pillow_image = imagem_original.convert("RGBA")

pillow_image.putalpha(
    pillow_mask
)

pillow_image.save(
    ARQUIVO_SEM_FUNDO
)

print("Imagem sem fundo salva:")
print(ARQUIVO_SEM_FUNDO)


# ==========================================================
# 6 - CRIAR COMPARACAO VISUAL
# ==========================================================

print("\n5 - CRIANDO COMPARACAO")

largura, altura = imagem_original.size

# Imagem original
painel_original = imagem_original.convert("RGB")

# Mascara em escala de cinza
painel_mascara = pillow_mask.convert("RGB")

# Mostrar transparencia sobre fundo branco
fundo_branco = Image.new(
    "RGBA",
    imagem_original.size,
    "white"
)

painel_sem_fundo = Image.alpha_composite(
    fundo_branco,
    pillow_image
).convert("RGB")

comparacao = Image.new(
    "RGB",
    (largura * 3, altura),
    "white"
)

comparacao.paste(
    painel_original,
    (0, 0)
)

comparacao.paste(
    painel_mascara,
    (largura, 0)
)

comparacao.paste(
    painel_sem_fundo,
    (largura * 2, 0)
)

comparacao.save(
    ARQUIVO_COMPARACAO
)

print("Comparacao salva:")
print(ARQUIVO_COMPARACAO)

print(
    "\nOrdem da comparacao: "
    "ORIGINAL | MASCARA | SEM FUNDO"
)


# ==========================================================
# 7 - VALIDACOES
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert ARQUIVO_ORIGINAL.exists()
assert ARQUIVO_MASCARA.exists()
assert ARQUIVO_SEM_FUNDO.exists()
assert ARQUIVO_COMPARACAO.exists()

assert pillow_mask.size == imagem_original.size

assert pillow_image.mode == "RGBA"

assert pillow_image.getchannel("A").getextrema()[0] < 255, (
    "A mascara nao produziu transparencia."
)

print("Imagem original: OK")
print("Pipeline RMBG-1.4: OK")
print("Mascara gerada: OK")
print("Fundo transparente: OK")
print("Comparacao criada: OK")
print("Arquivos PNG: OK")

print("\nARQUIVOS GERADOS:")
print(ARQUIVO_ORIGINAL)
print(ARQUIVO_MASCARA)
print(ARQUIVO_SEM_FUNDO)
print(ARQUIVO_COMPARACAO)

print("\nAULA 08 EXECUTADA COM SUCESSO!")
