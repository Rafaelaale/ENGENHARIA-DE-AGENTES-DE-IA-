# ==========================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 03 - PRIMEIRAS APLICACOES COM DIFFUSERS
# ==========================================================

from pathlib import Path
import argparse
import json

import numpy as np
import torch

from PIL import Image, ImageDraw
from diffusers import (
    DDPMPipeline,
    DDPMScheduler,
    UNet2DModel
)

BASE = Path(__file__).resolve().parent.parent
RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(min(4, torch.get_num_threads()))

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=[
        "ruido",
        "cifar",
        "rostos",
        "igreja",
        "stable"
    ],
    default="ruido"
)

parser.add_argument(
    "--passos",
    type=int,
    default=30
)

args = parser.parse_args()

if not 1 <= args.passos <= 1000:
    parser.error("--passos deve estar entre 1 e 1000")

print("=" * 65)
print("AULA 03 - INTRODUCAO A BIBLIOTECA DIFFUSERS")
print("=" * 65)

print("\nPyTorch:", torch.__version__)
print("CUDA disponivel:", CUDA)
print("Dispositivo:", DEVICE)

if CUDA:
    print("GPU:", torch.cuda.get_device_name(0))

print("Experimento:", args.modo)


# ==========================================================
# FUNCAO - CONVERTER TENSOR EM IMAGEM
# ==========================================================

def tensor_para_imagem(tensor):

    imagem = tensor.detach().float().cpu()[0]

    imagem = imagem.permute(1, 2, 0)

    imagem = (
        (imagem.clamp(-1, 1) + 1.0)
        * 127.5
    )

    imagem = imagem.round().to(torch.uint8)

    return Image.fromarray(
        imagem.numpy()
    )


# ==========================================================
# EXPERIMENTO 1 - ADICAO DE RUIDO COM SCHEDULER
# ==========================================================

def demonstrar_ruido():

    print("\n1 - CRIANDO IMAGEM ORIGINAL")

    imagem = Image.new(
        "RGB",
        (256, 256),
        (220, 235, 250)
    )

    desenho = ImageDraw.Draw(imagem)

    desenho.rectangle(
        (35, 35, 220, 220),
        fill=(60, 120, 220)
    )

    desenho.ellipse(
        (75, 75, 180, 180),
        fill=(245, 200, 60)
    )

    desenho.line(
        (20, 225, 235, 20),
        fill=(245, 70, 70),
        width=8
    )

    original = RESULTADOS / "aula03_imagem_original.png"
    imagem.save(original)

    array = np.asarray(
        imagem,
        dtype=np.float32
    )

    array = array / 127.5 - 1.0

    amostra = torch.from_numpy(
        array
    ).permute(2, 0, 1).unsqueeze(0)

    print("Imagem original criada: OK")

    print("\n2 - CONFIGURANDO DDPM SCHEDULER")

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    torch.manual_seed(42)

    ruido = torch.randn_like(amostra)

    timesteps = [
        0, 49, 149, 299,
        499, 699, 899, 999
    ]

    frames = []

    print("\n3 - ADICIONANDO RUIDO")

    for t in timesteps:

        instante = torch.tensor(
            [t],
            dtype=torch.long
        )

        imagem_ruidosa = scheduler.add_noise(
            original_samples=amostra,
            noise=ruido,
            timesteps=instante
        )

        frame = tensor_para_imagem(
            imagem_ruidosa
        )

        frames.append(frame)

        print(f"Ruido no timestep {t}: OK")

    print("\n4 - SALVANDO RESULTADOS")

    gif = RESULTADOS / "aula03_adicao_ruido.gif"

    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0
    )

    painel = Image.new(
        "RGB",
        (1024, 560),
        "white"
    )

    caneta = ImageDraw.Draw(painel)

    for i, frame in enumerate(frames):

        coluna = i % 4
        linha = i // 4

        pos_x = coluna * 256
        pos_y = linha * 280

        painel.paste(
            frame,
            (pos_x, pos_y + 20)
        )

        caneta.text(
            (pos_x + 8, pos_y + 4),
            f"Timestep {timesteps[i]}",
            fill="black"
        )

    imagem_painel = (
        RESULTADOS
        / "aula03_evolucao_ruido.png"
    )

    painel.save(imagem_painel)

    parametros = {
        "experimento": "forward diffusion",
        "total_timesteps": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "timesteps_visualizados": timesteps,
        "rede_neural_utilizada": False
    }

    with (
        RESULTADOS / "aula03_scheduler.json"
    ).open("w", encoding="utf-8") as arquivo:

        json.dump(
            parametros,
            arquivo,
            indent=4
        )

    assert len(frames) == 8
    assert gif.exists()
    assert imagem_painel.exists()

    print("GIF de ruido: OK")
    print("Painel PNG: OK")
    print("Configuracao JSON: OK")


# ==========================================================
# EXPERIMENTO 2 - DDPM CIFAR-10
# MODELO PEQUENO PARA CPU OU GPU
# ==========================================================

def gerar_cifar():

    print("\nCARREGANDO DDPM CIFAR-10")

    modelo_id = "google/ddpm-cifar10-32"

    pipe = DDPMPipeline.from_pretrained(
        modelo_id,
        use_safetensors=True
    )

    pipe = pipe.to(DEVICE)

    print("Modelo carregado: OK")
    print("Gerando imagem com DDPM...")

    with torch.inference_mode():

        imagem = pipe(
            num_inference_steps=args.passos
        ).images[0]

    destino = (
        RESULTADOS / "aula03_ddpm_cifar10.png"
    )

    imagem.save(destino)

    ampliada = imagem.resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

    ampliada.save(
        RESULTADOS
        / "aula03_ddpm_cifar10_ampliada.png"
    )

    assert destino.exists()

    print("Imagem gerada por rede neural: OK")


# ==========================================================
# PROTEGER EXPERIMENTOS GRANDES
# ==========================================================

def exigir_gpu():

    if not CUDA:
        raise SystemExit(
            "\nGPU CUDA nao detectada.\n"
            "Este experimento utiliza um modelo grande.\n"
            "Execute --modo ruido ou --modo cifar.\n"
            "Nenhum modelo pesado foi baixado."
        )


# ==========================================================
# EXPERIMENTO 3 - DDPM CELEBA-HQ 256
# ==========================================================

def gerar_rosto():

    exigir_gpu()

    print("\nCARREGANDO DDPM CELEBA-HQ")

    ddpm = DDPMPipeline.from_pretrained(
        "google/ddpm-celebahq-256"
    ).to("cuda")

    print("Gerando imagem...")

    with torch.inference_mode():

        imagem = ddpm(
            num_inference_steps=args.passos
        ).images[0]

    destino = (
        RESULTADOS / "aula03_ddpm_rosto.png"
    )

    imagem.save(destino)

    print("Rosto sintetico salvo:", destino)


# ==========================================================
# EXPERIMENTO 4 - U-NET + SCHEDULER
# GOOGLE DDPM CHURCH 256
# ==========================================================

def gerar_igreja():

    exigir_gpu()

    print("\nCARREGANDO U-NET")

    model = UNet2DModel.from_pretrained(
        "google/ddpm-church-256"
    ).to("cuda")

    model.eval()

    print("CARREGANDO SCHEDULER")

    scheduler = DDPMScheduler.from_pretrained(
        "google/ddpm-church-256"
    )

    scheduler.set_timesteps(
        args.passos,
        device="cuda"
    )

    sample_size = model.config.sample_size

    sample = torch.randn(
        (
            1,
            model.config.in_channels,
            sample_size,
            sample_size
        ),
        device="cuda"
    )

    frames = []

    intervalo = max(
        1,
        args.passos // 20
    )

    print("\nEXECUTANDO PROCESSO REVERSO")

    with torch.inference_mode():

        for i, t in enumerate(
            scheduler.timesteps
        ):

            residual = model(
                sample,
                t
            ).sample

            sample = scheduler.step(
                residual,
                t,
                sample
            ).prev_sample

            if (
                (i + 1) % intervalo == 0
                or i == args.passos - 1
            ):

                frame = tensor_para_imagem(
                    sample
                )

                frames.append(frame)

            print(
                f"Passo {i + 1}/{args.passos}",
                flush=True
            )

    assert frames

    imagem_final = (
        RESULTADOS / "aula03_igreja_final.png"
    )

    frames[-1].save(imagem_final)

    gif = (
        RESULTADOS
        / "aula03_processo_reverso_igreja.gif"
    )

    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=200,
        loop=0
    )

    print("Imagem final: OK")
    print("GIF do processo reverso: OK")


# ==========================================================
# EXPERIMENTO 5 - STABLE DIFFUSION 2.1
# ==========================================================

def gerar_stable():

    exigir_gpu()

    from diffusers import StableDiffusionPipeline

    print("\nCARREGANDO STABLE DIFFUSION 2.1")

    sdp = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda")

    prompt = (
        "A cat riding a horse in a futuristic city, "
        "cinematic lighting, detailed digital art"
    )

    print("PROMPT:", prompt)

    with torch.inference_mode():

        imagem = sdp(
            prompt,
            num_inference_steps=args.passos
        ).images[0]

    destino = (
        RESULTADOS
        / "aula03_stable_diffusion_gato.png"
    )

    imagem.save(destino)

    print("Imagem Stable Diffusion salva:", destino)


# ==========================================================
# EXECUCAO
# ==========================================================

if args.modo == "ruido":
    demonstrar_ruido()

elif args.modo == "cifar":
    gerar_cifar()

elif args.modo == "rostos":
    gerar_rosto()

elif args.modo == "igreja":
    gerar_igreja()

elif args.modo == "stable":
    gerar_stable()


# ==========================================================
# VALIDACAO
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

print("PyTorch: OK")
print("Diffusers: OK")
print("Dispositivo:", DEVICE)
print("Experimento executado:", args.modo)
print("Resultados salvos: OK")

print("\nAULA 03 - EXPERIMENTO CONCLUIDO!")
