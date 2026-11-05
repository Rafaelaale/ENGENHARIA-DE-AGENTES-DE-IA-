from pathlib import Path
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

BASE = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")
DADOS = BASE / "dados" / "redacoes_prontas_treinamento"
INICIAL = BASE / "modelos" / "distilbert_redacoes_inicial"
TREINADO = BASE / "modelos" / "distilbert_redacoes_treinado"
CHECKPOINT = BASE / "checkpoints" / "aula_10_estado.pt"
RELATORIO = BASE / "resultados" / "aula_10_treinamento.json"

EPOCAS = 2
BATCH = 2
ACUMULACAO = 8
TAXA = 2e-5

torch.manual_seed(42)
random.seed(42)
torch.set_num_threads(min(4, os.cpu_count() or 1))

print("\n===== AULA 10 - TRANSFERENCIA DE APRENDIZADO =====")
print("Dispositivo: CPU")

dados = load_from_disk(str(DADOS))

assert len(dados["treino"]) == 3656
assert len(dados["teste"]) == 914

tokenizador = AutoTokenizer.from_pretrained(
    str(INICIAL),
    local_files_only=True
)

modelo = AutoModelForSequenceClassification.from_pretrained(
    str(INICIAL),
    local_files_only=True
)

assert modelo.config.num_labels == 11
modelo.to("cpu")

collator = DataCollatorWithPadding(
    tokenizer=tokenizador,
    return_tensors="pt"
)

def criar_loader(nome, embaralhar=False, epoca=0):
    gerador = torch.Generator()
    gerador.manual_seed(42 + epoca)

    return DataLoader(
        dados[nome],
        batch_size=BATCH,
        shuffle=embaralhar,
        generator=gerador if embaralhar else None,
        collate_fn=collator,
        num_workers=0
    )

validacao = criar_loader("teste")

lotes = math.ceil(len(dados["treino"]) / BATCH)
passos_epoca = math.ceil(lotes / ACUMULACAO)
total_passos = EPOCAS * passos_epoca

otimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=TAXA
)

scheduler = get_linear_schedule_with_warmup(
    otimizador,
    num_warmup_steps=0,
    num_training_steps=total_passos
)

epoca_inicial = 0
passo_global = 0
historico = []
melhor_loss = float("inf")

if CHECKPOINT.exists():
    print("\n===== RECUPERANDO CHECKPOINT =====")

    estado = torch.load(
        str(CHECKPOINT),
        map_location="cpu"
    )

    modelo.load_state_dict(estado["modelo"])
    otimizador.load_state_dict(estado["otimizador"])
    scheduler.load_state_dict(estado["scheduler"])

    epoca_inicial = estado["proxima_epoca"]
    passo_global = estado["passo_global"]
    historico = estado["historico"]
    melhor_loss = estado["melhor_loss"]

    print("Proxima epoca:", epoca_inicial + 1)

def salvar_checkpoint(proxima_epoca):
    CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporario = CHECKPOINT.with_suffix(".tmp")

    torch.save({
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "scheduler": scheduler.state_dict(),
        "proxima_epoca": proxima_epoca,
        "passo_global": passo_global,
        "historico": historico,
        "melhor_loss": melhor_loss
    }, temporario)

    os.replace(temporario, CHECKPOINT)
    print("Checkpoint salvo.", flush=True)

def avaliar():
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.no_grad():
        for lote in validacao:
            saida = modelo(**lote)
            n = lote["labels"].shape[0]

            perda_total += saida.loss.item() * n
            acertos += (
                saida.logits.argmax(-1) == lote["labels"]
            ).sum().item()

            total += n

    return perda_total / total, acertos / total

print("\n===== CONFIGURACAO =====")
print("Treino:", len(dados["treino"]))
print("Avaliacao:", len(dados["teste"]))
print("Epocas:", EPOCAS)
print("Batch:", BATCH)
print("Acumulacao:", ACUMULACAO)
print("Batch efetivo:", BATCH * ACUMULACAO)
print("Passos por epoca:", passos_epoca)
print("Total de passos:", total_passos)

print("\n===== INICIANDO TREINAMENTO =====", flush=True)

for epoca in range(epoca_inicial, EPOCAS):

    print(
        f"\n===== EPOCA {epoca + 1}/{EPOCAS} =====",
        flush=True
    )

    treino = criar_loader(
        "treino",
        embaralhar=True,
        epoca=epoca
    )

    modelo.train()
    otimizador.zero_grad()

    perda_total = 0.0
    exemplos = 0

    for indice, lote in enumerate(treino):

        saida = modelo(**lote)

        quantidade = lote["labels"].shape[0]
        perda_total += saida.loss.item() * quantidade
        exemplos += quantidade

        grupo = (indice // ACUMULACAO) * ACUMULACAO
        divisor = min(ACUMULACAO, len(treino) - grupo)

        (saida.loss / divisor).backward()

        fim_grupo = (
            (indice + 1) % ACUMULACAO == 0
            or indice + 1 == len(treino)
        )

        if fim_grupo:
            torch.nn.utils.clip_grad_norm_(
                modelo.parameters(), 1.0
            )

            otimizador.step()
            scheduler.step()
            otimizador.zero_grad()

            passo_global += 1

            if (
                passo_global % 10 == 0
                or indice + 1 == len(treino)
            ):
                print(
                    f"Epoca {epoca + 1} | "
                    f"Lote {indice + 1}/{len(treino)} | "
                    f"Passo {passo_global}/{total_passos} | "
                    f"Loss: {saida.loss.item():.4f}",
                    flush=True
                )

    print("\n===== AVALIANDO =====", flush=True)

    loss_validacao, acuracia = avaliar()

    resultado = {
        "epoca": epoca + 1,
        "loss_treino": perda_total / exemplos,
        "loss_avaliacao": loss_validacao,
        "acuracia_avaliacao": acuracia
    }

    historico.append(resultado)

    print(
        f"Loss avaliacao: {loss_validacao:.4f}",
        flush=True
    )

    print(
        f"Acuracia avaliacao: {acuracia:.2%}",
        flush=True
    )

    if loss_validacao < melhor_loss:
        melhor_loss = loss_validacao

        TREINADO.mkdir(
            parents=True,
            exist_ok=True
        )

        modelo.save_pretrained(
            str(TREINADO),
            safe_serialization=True
        )
        tokenizador.save_pretrained(str(TREINADO))

        print("Melhor modelo salvo.", flush=True)

    salvar_checkpoint(epoca + 1)

RELATORIO.parent.mkdir(
    parents=True,
    exist_ok=True
)

RELATORIO.write_text(
    json.dumps({
        "modelo": "Geotrend/distilbert-base-pt-cased",
        "epocas": EPOCAS,
        "batch_efetivo": BATCH * ACUMULACAO,
        "taxa_aprendizado": TAXA,
        "melhor_loss": melhor_loss,
        "historico": historico,
        "modelo_salvo": str(TREINADO)
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n===== RESULTADO FINAL =====")
print(json.dumps(historico, indent=2))
print("\nAULA 10 - TREINAMENTO CONCLUIDO")