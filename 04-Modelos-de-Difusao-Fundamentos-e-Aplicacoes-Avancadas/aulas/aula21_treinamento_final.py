# PROJETO 04 - AULA 21
# TREINAMENTO FINAL, AVALIACAO E MLOPS

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from PIL import Image

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================================================
# 1 - CONFIGURACAO
# ==================================================

BASE = Path(__file__).resolve().parent.parent
SAIDA = BASE / "resultados"
SAIDA.mkdir(exist_ok=True, parents=True)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}
T = 1000
SEED = 42

# Mapeamento REAL do dataset do Projeto 04
DESCRICOES = {
    "x": "sagital",
    "y": "coronal",
    "z": "axial"
}

CONFIGS = {
    "teste": dict(
        epocas=1, por_classe=1, validacao=1,
        base=16, micro=1, acum=1
    ),
    "reduzido": dict(
        epocas=3, por_classe=16, validacao=4,
        base=16, micro=2, acum=2
    ),
    "completo": dict(
        epocas=50, por_classe=None, validacao=257,
        base=64, micro=1, acum=128
    )
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modo",
    choices=["teste", "reduzido", "completo", "gerar", "avaliar"],
    default="teste"
)

parser.add_argument(
    "--pesos",
    choices=["reduzido", "completo"],
    default="reduzido"
)

parser.add_argument("--iniciar-aula20", action="store_true")
parser.add_argument("--metricas-periodicas", action="store_true")
parser.add_argument("--eval-frequency", type=int, default=5)
parser.add_argument("--patience", type=int, default=25)
parser.add_argument("--checkpoint-cada", type=int, default=8)
parser.add_argument("--amostras", type=int, default=60)
parser.add_argument("--passos", type=int, default=50)

parser.add_argument(
    "--classe",
    choices=["x", "y", "z", "todas"],
    default="todas"
)

args = parser.parse_args()

if (
    args.eval_frequency < 1
    or args.patience < 1
    or args.checkpoint_cada < 1
    or args.amostras < 1
    or not 2 <= args.passos <= T
):
    parser.error("Parametros invalidos.")

modo_pesos = (
    args.modo if args.modo in CONFIGS else args.pesos
)

cfg = CONFIGS[modo_pesos]

PREFIXO = "aula21_" + modo_pesos

PATH_CK = SAIDA / (PREFIXO + "_checkpoint.pth")
PATH_BEST = SAIDA / (PREFIXO + "_melhor_modelo.pth")
PATH_LAST = SAIDA / (PREFIXO + "_ultimo_modelo.pth")
PATH_CSV = SAIDA / (PREFIXO + "_historico.csv")
PATH_PNG = SAIDA / (PREFIXO + "_curva.png")
PATH_META = SAIDA / (PREFIXO + "_manifesto.json")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

if device.type == "cpu":
    torch.set_num_threads(min(4, os.cpu_count() or 1))

print("=" * 65)
print("AULA 21 - TREINAMENTO FINAL / METRICAS / MLOPS")
print("=" * 65)
print("Modo:", args.modo)
print("Dispositivo:", device)


# ==================================================
# 2 - REUTILIZAR ARQUITETURA DA AULA 15
# ==================================================

nomes = [
    "BrainMRIConditionalDataset",
    "transform_mri",
    "sinusoidal_embedding",
    "LabelEmbedding",
    "ConditionalTimeEmbedding",
    "ConditionalResidualBlock",
    "ConditionalDown",
    "ConditionalUp",
    "ConditionalUNet"
]

caminho = BASE / "aulas" / "aula15_unet_condicional.py"

arvore = ast.parse(
    caminho.read_text(encoding="utf-8-sig")
)

defs = {
    n.name: n
    for n in arvore.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
}

ausentes = set(nomes) - set(defs)

if ausentes:
    raise RuntimeError(
        f"Definicoes da Aula 15 ausentes: {ausentes}"
    )

modulo = ast.Module(
    body=[defs[n] for n in nomes],
    type_ignores=[]
)

ast.fix_missing_locations(modulo)

exec(
    compile(modulo, str(caminho), "exec"),
    globals()
)

dataset = BrainMRIConditionalDataset(
    BASE / "dados" / "brain-mri-images",
    transform=transform_mri,
    slice_plane=None
)

assert dataset.label_to_idx == {
    "x": 0, "y": 1, "z": 2
}

print("\n1 - DATASET")
print("Imagens:", len(dataset))
print("Classes:", dataset.label_to_idx)
print("Descricoes:", DESCRICOES)


# ==================================================
# 3 - MODELO
# ==================================================

model = ConditionalUNet(
    in_channels=1,
    base_channels=cfg["base"],
    num_classes=3,
    time_dim=256,
    label_dim=128,
    dropout_prob=0.1
).to(device)

assert model.null_label == 3

parametros = sum(
    p.numel() for p in model.parameters()
)

print("\n2 - MODELO")
print("Parametros:", parametros)
print("Base channels:", cfg["base"])
print("Classe nula CFG:", model.null_label)


# ==================================================
# 4 - SCHEDULER DDPM
# ==================================================

betas = torch.linspace(
    1e-4, 0.02, T, device=device
)

alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def q_sample(x0, t, ruido):
    a = alpha_bar[t].reshape(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * ruido


def conditional_loss_fn(model, x0, labels, t):
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    pred = model(x_t, t, labels)
    return F.mse_loss(pred.float(), noise.float())


# ==================================================
# 5 - DIVIDIR TREINO E VALIDACAO
# ==================================================

rng = np.random.default_rng(2026)

train_indices = []
val_indices = []

for classe in range(3):

    disponiveis = np.asarray([
        i for i, label in enumerate(dataset.labels)
        if label == classe
    ])

    perm = rng.permutation(disponiveis).tolist()

    n_val = cfg["validacao"]

    n_treino = (
        len(perm) - n_val
        if cfg["por_classe"] is None
        else cfg["por_classe"]
    )

    if n_treino + n_val > len(perm):
        raise RuntimeError("Dataset insuficiente.")

    train_indices.extend(perm[:n_treino])
    val_indices.extend(
        perm[n_treino:n_treino + n_val]
    )

split_hash = hashlib.sha256(
    json.dumps(
        [train_indices, val_indices]
    ).encode()
).hexdigest()

print("Imagens de treino:", len(train_indices))
print("Imagens de validacao:", len(val_indices))


# ==================================================
# 6 - HISTORICO E CURVA
# ==================================================

def escrever_historico(h):

    campos = [
        "epoca",
        "mse_treino",
        "mse_validacao",
        "learning_rate",
        "fid",
        "is_score",
        "lpips_diversidade"
    ]

    with PATH_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(h)

    if not h:
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    epocas = [e["epoca"] for e in h]

    ax.plot(
        epocas,
        [e["mse_treino"] for e in h],
        marker="o",
        label="Treinamento"
    )

    ax.plot(
        epocas,
        [e["mse_validacao"] for e in h],
        marker="o",
        label="Validacao"
    )

    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE")
    ax.set_title("Aula 21 - Treino e Validacao")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(PATH_PNG, dpi=140)
    plt.close(fig)


# ==================================================
# 7 - VALIDACAO
# ==================================================

@torch.inference_mode()
def validar():

    model.eval()

    soma = 0.0
    n = 0

    dl = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg["micro"],
        shuffle=False,
        num_workers=0
    )

    for idx, batch in enumerate(dl):

        torch.manual_seed(20000 + idx)

        x = batch["image"].to(device)
        y = batch["label"].to(device, dtype=torch.long)

        t = torch.randint(
            0, T, (len(x),), device=device
        )

        mse = conditional_loss_fn(model, x, y, t)

        soma += mse.item() * len(x)
        n += len(x)

    return soma / n


# ==================================================
# 8 - SAMPLER DDIM
# ==================================================

@torch.inference_mode()
def amostrar_ddim(labels, passos=50, seed=None):

    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    labels = labels.to(
        device=device,
        dtype=torch.long
    )

    x = torch.randn(
        len(labels), 1, 64, 64,
        device=device
    )

    tempos = (
        torch.linspace(T - 1, 0, passos)
        .round()
        .long()
        .unique_consecutive()
        .tolist()
    )

    for i, t in enumerate(tempos):

        anterior = (
            tempos[i + 1]
            if i + 1 < len(tempos)
            else -1
        )

        ts = torch.full(
            (len(labels),),
            t,
            device=device,
            dtype=torch.long
        )

        eps_cond = model(x, ts, labels)
        eps_null = model(x, ts, None)

        eps = (
            eps_null
            + 3.0 * (eps_cond - eps_null)
        )

        a = alpha_bar[t]

        ap = (
            alpha_bar[anterior]
            if anterior >= 0
            else x.new_tensor(1.0)
        )

        x0 = (
            (x - (1 - a).sqrt() * eps)
            / a.sqrt()
        ).clamp(-1, 1)

        x = (
            ap.sqrt() * x0
            + (1 - ap).sqrt() * eps
        )

    return x.clamp(-1, 1)


# ==================================================
# 9 - METRICAS GENERATIVAS
# ==================================================

def rgb01(x):
    return (
        (x.float().clamp(-1, 1) + 1) / 2
    ).repeat(1, 3, 1, 1)


def rgb11(x):
    return x.float().clamp(
        -1, 1
    ).repeat(1, 3, 1, 1)


def avaliar_metricas(num_samples=60, passos=50):

    if num_samples < 30 or num_samples % 3 != 0:
        raise ValueError(
            "Utilize ao menos 30 amostras, "
            "em multiplo de tres."
        )

    try:
        import torch_fidelity

        from torchmetrics.image.fid import (
            FrechetInceptionDistance
        )

        from torchmetrics.image.inception import (
            InceptionScore
        )

        from torchmetrics.image.lpip import (
            LearnedPerceptualImagePatchSimilarity
        )

    except ImportError as erro:
        raise RuntimeError(
            "Instale torchmetrics e torch-fidelity."
        ) from erro

    model.eval()

    n = num_samples // 3
    real = []
    fake = []

    treino_set = set(train_indices)

    for classe in range(3):

        escolhas = [
            i
            for i, label in enumerate(dataset.labels)
            if label == classe and i not in treino_set
        ]

        rng = np.random.default_rng(3000 + classe)

        indices = rng.choice(
            escolhas,
            n,
            replace=False
        ).tolist()

        for inicio in range(0, n, 2):

            qtd = min(2, n - inicio)

            labels = torch.full(
                (qtd,),
                classe,
                device=device,
                dtype=torch.long
            )

            imagens = amostrar_ddim(
                labels,
                passos,
                seed=40000 + classe * 1000 + inicio
            )

            fake.append(imagens.cpu())

            real.append(
                torch.stack([
                    dataset[i]["image"]
                    for i in indices[
                        inicio:inicio + qtd
                    ]
                ])
            )

            print(
                f"Classe {classe}: "
                f"{inicio + qtd}/{n}",
                flush=True
            )

    real = torch.cat(real, dim=0)
    fake = torch.cat(fake, dim=0)

    fid = FrechetInceptionDistance(
        feature=64,
        normalize=True
    ).to(device)

    iscore = InceptionScore(
        splits=10,
        normalize=True
    ).to(device)

    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex",
        reduction="none",
        normalize=False
    ).to(device)

    with torch.inference_mode():

        for i in range(0, len(real), 4):

            fid.update(
                rgb01(real[i:i + 4].to(device)),
                real=True
            )

            geradas = rgb01(
                fake[i:i + 4].to(device)
            )

            fid.update(geradas, real=False)
            iscore.update(geradas)

        fid_num = float(fid.compute().item())

        is_mean, is_std = iscore.compute()

        # LPIPS entre imagens sinteticas
        # da mesma classe: medida de diversidade.
        lp_values = []

        for i in range(0, len(fake), n):

            bloco = fake[i:i + n]

            for j in range(0, n - 1, 4):

                a = bloco[
                    j:min(j + 4, n - 1)
                ].to(device)

                b = bloco[
                    j + 1:min(j + 5, n)
                ].to(device)

                scores = lpips(
                    rgb11(a),
                    rgb11(b)
                )

                lp_values.extend(
                    scores.flatten()
                    .cpu()
                    .tolist()
                )

    return {
        "fid64": fid_num,
        "is_media": float(is_mean.item()),
        "is_desvio": float(is_std.item()),
        "lpips_diversidade": float(
            np.mean(lp_values)
        ),
        "amostras": num_samples,
        "passos_ddim": passos,
        "limitacoes": (
            "FID64 exploratorio. Inception treinado "
            "fora do dominio MRI. LPIPS aqui mede "
            "diversidade entre imagens sinteticas, "
            "nao fidelidade clinica."
        )
    }


# ==================================================
# 10 - LOOP DE TREINAMENTO FINAL
# ==================================================

if args.modo in CONFIGS:

    print("\n3 - TREINAMENTO")

    print("Epocas:", cfg["epocas"])
    print("Microbatch:", cfg["micro"])
    print("Acumulacao:", cfg["acum"])
    print(
        "Batch efetivo:",
        cfg["micro"] * cfg["acum"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epocas"]
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    print(
        "Mixed precision CUDA:",
        scaler.is_enabled()
    )

    estado = {
        "epoca": 0,
        "prox_batch": 0,
        "soma": 0.0,
        "n_imagens": 0,
        "updates": 0,
        "sem_melhora": 0,
        "best_val": float("inf"),
        "early_stop": False
    }

    historico = []

    if PATH_CK.exists():

        ck = torch.load(
            PATH_CK,
            map_location=device,
            weights_only=True
        )

        if (
            ck["cfg"] != cfg
            or ck["split_hash"] != split_hash
        ):
            raise RuntimeError(
                "Checkpoint incompativel."
            )

        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])

        estado = ck["estado"]
        historico = ck["historico"]

        print(
            "Checkpoint recuperado. Epoca:",
            estado["epoca"] + 1
        )

        print(
            "Proximo microbatch:",
            estado["prox_batch"]
        )

    elif args.iniciar_aula20:

        if modo_pesos != "reduzido":
            raise ValueError(
                "Os pesos reduzidos da Aula 20 "
                "exigem modo reduzido."
            )

        pesos_prev = (
            SAIDA
            / "aula20_reduzido_modelo_melhor.pth"
        )

        if not pesos_prev.exists():
            raise FileNotFoundError(
                f"Pesos da Aula 20 ausentes: {pesos_prev}"
            )

        model.load_state_dict(
            torch.load(
                pesos_prev,
                map_location=device,
                weights_only=True
            )
        )

        print(
            "Inicializado com:",
            pesos_prev.name
        )

    else:
        print("Inicializacao: pesos aleatorios.")

    if args.metricas_periodicas:

        try:
            import torchmetrics
            import torch_fidelity

        except ImportError as erro:
            raise RuntimeError(
                "Instale torchmetrics e torch-fidelity."
            ) from erro

    manifesto = {
        "aula": 21,
        "modo": modo_pesos,
        "classes": dataset.label_to_idx,
        "descricoes": DESCRICOES,
        "quantidade_dataset": len(dataset),
        "amostras_treino": len(train_indices),
        "amostras_validacao": len(val_indices),
        "configuracao": cfg,
        "scheduler": {
            "T": 1000,
            "beta_inicio": 0.0001,
            "beta_fim": 0.02
        },
        "torch": torch.__version__,
        "device": str(device),
        "uso": "Educacional. Sem validacao clinica."
    }

    PATH_META.write_text(
        json.dumps(
            manifesto,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    def checkpoint():

        pacote = {
            "cfg": cfg,
            "split_hash": split_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "estado": estado.copy(),
            "historico": historico
        }

        temporario = PATH_CK.with_suffix(".tmp")

        torch.save(pacote, temporario)
        os.replace(temporario, PATH_CK)

    if (
        estado["epoca"] >= cfg["epocas"]
        or estado.get("early_stop", False)
    ):

        print(
            "Treinamento ja finalizado no checkpoint."
        )

    else:

        for epoca in range(
            estado["epoca"],
            cfg["epocas"]
        ):

            gerador = torch.Generator()
            gerador.manual_seed(SEED + epoca)

            ordem = [
                train_indices[i]
                for i in torch.randperm(
                    len(train_indices),
                    generator=gerador
                ).tolist()
            ]

            total = math.ceil(
                len(ordem) / cfg["micro"]
            )

            inicio = (
                estado["prox_batch"]
                if epoca == estado["epoca"]
                else 0
            )

            soma = (
                estado["soma"]
                if epoca == estado["epoca"]
                else 0.0
            )

            n_imagens = (
                estado["n_imagens"]
                if epoca == estado["epoca"]
                else 0
            )

            dl = DataLoader(
                Subset(
                    dataset,
                    ordem[inicio * cfg["micro"]:]
                ),
                batch_size=cfg["micro"],
                shuffle=False,
                num_workers=0
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)

            print(
                f"\nEpoca {epoca + 1}/{cfg['epocas']}"
                f" | microbatches: {total}",
                flush=True
            )

            for offset, batch in enumerate(dl):

                idx = inicio + offset

                seed_batch = (
                    500000 + epoca * 100000 + idx
                )

                torch.manual_seed(seed_batch)

                if device.type == "cuda":
                    torch.cuda.manual_seed_all(
                        seed_batch
                    )

                x = batch["image"].to(device)

                labels = batch["label"].to(
                    device,
                    dtype=torch.long
                )

                t = torch.randint(
                    0, T, (len(x),),
                    device=device
                )

                with torch.amp.autocast(
                    "cuda",
                    enabled=(device.type == "cuda")
                ):

                    loss = conditional_loss_fn(
                        model, x, labels, t
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Loss nao finito."
                    )

                primeira = (
                    idx // cfg["acum"]
                ) * cfg["acum"]

                ultima = min(
                    primeira + cfg["acum"],
                    total
                )

                n_grupo = (
                    min(
                        ultima * cfg["micro"],
                        len(ordem)
                    )
                    - primeira * cfg["micro"]
                )

                loss_ajustada = (
                    loss * (len(x) / n_grupo)
                )

                scaler.scale(
                    loss_ajustada
                ).backward()

                soma += (
                    float(loss.item()) * len(x)
                )

                n_imagens += len(x)

                if idx + 1 == ultima:

                    scaler.unscale_(optimizer)

                    clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    estado.update(
                        epoca=epoca,
                        prox_batch=idx + 1,
                        soma=soma,
                        n_imagens=n_imagens,
                        updates=estado["updates"] + 1
                    )

                    if (
                        estado["updates"]
                        % args.checkpoint_cada == 0
                    ):

                        checkpoint()

                        print(
                            f"Batch {idx + 1}/{total}"
                            f" | MSE {soma / n_imagens:.6f}"
                            " | checkpoint OK",
                            flush=True
                        )

            treino = soma / n_imagens
            val = validar()

            melhor = (
                val < estado["best_val"] - 1e-6
            )

            if melhor:

                estado["best_val"] = val
                estado["sem_melhora"] = 0

                torch.save(
                    model.state_dict(),
                    PATH_BEST
                )

            else:
                estado["sem_melhora"] += 1

            scheduler.step()

            registro = {
                "epoca": epoca + 1,
                "mse_treino": treino,
                "mse_validacao": val,
                "learning_rate": (
                    optimizer.param_groups[0]["lr"]
                ),
                "fid": "",
                "is_score": "",
                "lpips_diversidade": ""
            }

            if (
                args.metricas_periodicas
                and (
                    (epoca + 1) % args.eval_frequency == 0
                    or epoca + 1 == cfg["epocas"]
                )
            ):

                print(
                    "Avaliando FID/IS/LPIPS...",
                    flush=True
                )

                metricas = avaliar_metricas(
                    args.amostras,
                    args.passos
                )

                registro.update(
                    fid=metricas["fid64"],
                    is_score=metricas["is_media"],
                    lpips_diversidade=(
                        metricas["lpips_diversidade"]
                    )
                )

                destino = (
                    SAIDA
                    / f"{PREFIXO}_metricas_epoca{epoca+1:03d}.json"
                )

                destino.write_text(
                    json.dumps(
                        metricas,
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

            historico.append(registro)

            estado.update(
                epoca=epoca + 1,
                prox_batch=0,
                soma=0.0,
                n_imagens=0
            )

            if (
                estado["sem_melhora"] >= args.patience
            ):
                estado["early_stop"] = True

            checkpoint()
            escrever_historico(historico)

            if modo_pesos == "reduzido":

                torch.save(
                    model.state_dict(),
                    SAIDA
                    / f"{PREFIXO}_epoca_{epoca+1:03d}.pth"
                )

            print(
                f"Epoca {epoca + 1}/{cfg['epocas']}"
                f" | treino={treino:.6f}"
                f" | validacao={val:.6f}"
                f" | melhor={melhor}"
                f" | LR={optimizer.param_groups[0]['lr']:.8f}",
                flush=True
            )

            if estado["early_stop"]:

                print(
                    "EARLY STOP: sem melhoria no "
                    "MSE de validacao por",
                    args.patience,
                    "epocas."
                )

                break

    torch.save(
        model.state_dict(),
        PATH_LAST
    )

    escrever_historico(historico)

    assert historico
    assert PATH_CK.exists()
    assert PATH_BEST.exists()
    assert PATH_LAST.exists()

    print("\n" + "=" * 65)
    print("VALIDACAO FINAL - AULA 21")
    print("=" * 65)

    print("Dataset e classe nula: OK")
    print("Treinamento, MSE, backward e AdamW: OK")
    print("Scheduler Cosine / GradScaler / clipping: OK")
    print("Melhor modelo segundo MSE validacao: SALVO")
    print("Checkpoint / ultimo modelo: OK")
    print("Curva / CSV / manifesto: OK")

    print(
        "Epocas registradas:",
        len(historico)
    )

    print(
        "FID / IS / LPIPS:",
        "calculadas sob demanda"
        if args.metricas_periodicas
        else "pendentes"
    )

    print(
        f"AULA 21 - {args.modo.upper()} CONCLUIDO!"
    )


# ==================================================
# 11 - GERACAO OU AVALIACAO SEPARADA
# ==================================================

elif args.modo in ("gerar", "avaliar"):

    if not PATH_BEST.exists():
        raise FileNotFoundError(
            f"Modelo ausente: {PATH_BEST}"
        )

    model.load_state_dict(
        torch.load(
            PATH_BEST,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    if args.modo == "gerar":

        classes = (
            ("x", "y", "z")
            if args.classe == "todas"
            else (args.classe,)
        )

        for classe in classes:

            rotulo = dataset.label_to_idx[classe]

            quantidade = min(6, args.amostras)

            labels = torch.full(
                (quantidade,),
                rotulo,
                device=device,
                dtype=torch.long
            )

            imagens = amostrar_ddim(
                labels,
                args.passos,
                seed=2026 + rotulo
            )

            for i, img in enumerate(imagens):

                arr = (
                    ((img[0].detach().cpu() + 1) / 2)
                    .clamp(0, 1) * 255
                ).to(torch.uint8).numpy()

                path = (
                    SAIDA
                    / f"{PREFIXO}_{classe}_ddim_{i+1:02d}.png"
                )

                Image.fromarray(arr).resize(
                    (256, 256),
                    Image.Resampling.NEAREST
                ).save(path)

                print("Imagem:", path)

        print(
            "AULA 21 - GERACAO DDIM CONCLUIDA!"
        )

    else:

        print(
            "Avaliando:",
            args.amostras,
            "imagens | DDIM:",
            args.passos,
            "passos"
        )

        metricas = avaliar_metricas(
            args.amostras,
            args.passos
        )

        path = (
            SAIDA
            / f"{PREFIXO}_metricas.json"
        )

        path.write_text(
            json.dumps(
                metricas,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print("FID64:", metricas["fid64"])
        print("IS:", metricas["is_media"])
        print(
            "LPIPS diversidade:",
            metricas["lpips_diversidade"]
        )

        print("Relatorio:", path)

        print(
            "AULA 21 - AVALIACAO CONCLUIDA!"
        )
