import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    pipeline,
)

BASE = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")
ORIGEM = BASE / "modelos" / "distilbert_redacoes_inicial"
DESTINO = BASE / "modelos" / "distilbert_redacoes_treinado"
DADOS = BASE / "dados" / "redacoes_prontas_treinamento"
CHECK = BASE / "checkpoints" / "aula_10_final_cpu.pt"
RELATORIO = BASE / "resultados" / "aula_10_treinamento.json"
PREVISOES = BASE / "resultados" / "aula_12_previsoes.json"

EPOCAS = 2
ACUMULACAO = 16
LR = 2e-5

torch.set_num_threads(1)
torch.manual_seed(42)
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--probe", type=int)
parser.add_argument("--train", type=int)
args = parser.parse_args()

dados = load_from_disk(str(DADOS))
assert len(dados["treino"]) == 3656
assert len(dados["teste"]) == 914

tokenizador = AutoTokenizer.from_pretrained(
    str(ORIGEM), local_files_only=True
)

modelo = AutoModelForSequenceClassification.from_pretrained(
    str(ORIGEM), local_files_only=True
)
assert modelo.config.num_labels == 11

limite = args.probe or args.train
assert limite in (128, 256)

def preparar(registro):
    ids = registro["input_ids"]
    mascara = registro["attention_mask"]

    if len(ids) > limite:
        ids = ids[:limite - 1] + [ids[-1]]
        mascara = mascara[:limite - 1] + [1]

    return {
        "input_ids": ids,
        "attention_mask": mascara,
        "labels": registro["label"],
    }

class DatasetPreparado:
    def __init__(self, conjunto):
        self.conjunto = dados[conjunto]

    def __len__(self):
        return len(self.conjunto)

    def __getitem__(self, indice):
        return preparar(self.conjunto[indice])

collator = DataCollatorWithPadding(
    tokenizer=tokenizador,
    return_tensors="pt"
)

def loader(conjunto, epoca=0):
    gerador = torch.Generator().manual_seed(42 + epoca)
    return DataLoader(
        DatasetPreparado(conjunto),
        batch_size=1,
        shuffle=(conjunto == "treino"),
        generator=gerador if conjunto == "treino" else None,
        collate_fn=collator,
        num_workers=0
    )

if args.probe:
    print(f"===== TESTE REAL COM {limite} TOKENS =====", flush=True)
    modelo.train()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=LR)
    lote = preparar(dados["treino"][0])
    tensores = collator([lote])

    saida = modelo(**tensores)
    assert torch.isfinite(saida.loss)
    saida.loss.backward()
    otimizador.step()

    print("Forward: OK", flush=True)
    print("Backward: OK", flush=True)
    print("AdamW: OK", flush=True)
    print("Loss:", saida.loss.item(), flush=True)
    print("TESTE APROVADO", flush=True)
    raise SystemExit(0)

# TREINAMENTO COMPLETO

passos_epoca = math.ceil(len(dados["treino"]) / ACUMULACAO)
total_passos = passos_epoca * EPOCAS

otimizador = torch.optim.AdamW(modelo.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    otimizador, num_warmup_steps=0,
    num_training_steps=total_passos
)

epoca_inicial = 0
proximo_lote = 0
passo_global = 0
historico = []
melhor_loss = float("inf")

if CHECK.exists():
    estado = torch.load(str(CHECK), map_location="cpu")
    if estado["limite"] != limite:
        raise RuntimeError("Limite de tokens diferente do checkpoint.")

    modelo.load_state_dict(estado["modelo"])
    otimizador.load_state_dict(estado["otimizador"])
    scheduler.load_state_dict(estado["scheduler"])

    epoca_inicial = estado["epoca"]
    proximo_lote = estado["proximo_lote"]
    passo_global = estado["passo_global"]
    historico = estado["historico"]
    melhor_loss = estado["melhor_loss"]

    torch.set_rng_state(estado["rng_torch"])

    print("CHECKPOINT RECUPERADO", flush=True)
    print("Epoca:", epoca_inicial + 1, flush=True)
    print("Lote:", proximo_lote + 1, flush=True)

def salvar_checkpoint(epoca, lote):
    temporario = CHECK.with_suffix(".tmp")

    torch.save({
        "modelo": modelo.state_dict(),
        "otimizador": otimizador.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoca": epoca,
        "proximo_lote": lote,
        "passo_global": passo_global,
        "historico": historico,
        "melhor_loss": melhor_loss,
        "limite": limite,
        "rng_torch": torch.get_rng_state(),
    }, str(temporario))

    os.replace(str(temporario), str(CHECK))
    print("CHECKPOINT SALVO:", passo_global, flush=True)

def avaliar():
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.no_grad():
        for i, lote in enumerate(loader("teste")):
            saida = modelo(**lote)
            perda_total += saida.loss.item()
            acertos += (
                saida.logits.argmax(-1) == lote["labels"]
            ).sum().item()
            total += 1

            if (i + 1) % 200 == 0:
                print("Avaliadas:", i + 1, flush=True)

    return perda_total / total, acertos / total

print("\n===== TREINAMENTO DISTILBERT =====", flush=True)
print("Threads: 1", flush=True)
print("Tokens maximos:", limite, flush=True)
print("Redacoes treino:", len(dados["treino"]), flush=True)
print("Redacoes avaliacao:", len(dados["teste"]), flush=True)
print("Epocas:", EPOCAS, flush=True)
print("Batch efetivo:", ACUMULACAO, flush=True)
print("Atualizacoes totais:", total_passos, flush=True)

for epoca in range(epoca_inicial, EPOCAS):
    print(f"\n===== EPOCA {epoca+1}/{EPOCAS} =====", flush=True)

    treino = loader("treino", epoca)
    modelo.train()
    otimizador.zero_grad(set_to_none=True)

    for indice, lote in enumerate(treino):
        if epoca == epoca_inicial and indice < proximo_lote:
            continue

        saida = modelo(**lote)
        assert torch.isfinite(saida.loss)

        inicio_grupo = (indice // ACUMULACAO) * ACUMULACAO
        divisor = min(ACUMULACAO, len(treino) - inicio_grupo)

        (saida.loss / divisor).backward()

        fim_grupo = (
            (indice + 1) % ACUMULACAO == 0
            or indice + 1 == len(treino)
        )

        if fim_grupo:
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            scheduler.step()
            otimizador.zero_grad(set_to_none=True)
            passo_global += 1

            if passo_global % 10 == 0:
                print(
                    f"Epoca {epoca+1} | Lote {indice+1}/{len(treino)} "
                    f"| Passo {passo_global}/{total_passos} "
                    f"| Loss {saida.loss.item():.4f}",
                    flush=True
                )

            if passo_global % 50 == 0:
                salvar_checkpoint(epoca, indice + 1)

    print("\n===== AVALIACAO =====", flush=True)
    loss, acuracia = avaliar()

    historico.append({
        "epoca": epoca + 1,
        "loss_avaliacao": loss,
        "acuracia_avaliacao": acuracia
    })

    print("Loss avaliacao:", round(loss, 4), flush=True)
    print("Acuracia avaliacao:", round(acuracia, 4), flush=True)

    if loss < melhor_loss:
        melhor_loss = loss
        DESTINO.mkdir(parents=True, exist_ok=True)
        modelo.save_pretrained(
            str(DESTINO), safe_serialization=True
        )
        tokenizador.save_pretrained(str(DESTINO))
        print("MELHOR MODELO SALVO", flush=True)

    salvar_checkpoint(epoca + 1, 0)
    proximo_lote = 0

if len(historico) != EPOCAS:
    raise RuntimeError("Duas epocas nao foram concluidas.")

RELATORIO.write_text(
    json.dumps({
        "modelo": "Geotrend/distilbert-base-pt-cased",
        "framework": "PyTorch",
        "epocas": EPOCAS,
        "limite_tokens": limite,
        "batch_efetivo": ACUMULACAO,
        "taxa_aprendizado": LR,
        "melhor_loss": melhor_loss,
        "historico": historico
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

# VERIFICAR PREVISOES DO MELHOR MODELO

print("\n===== VALIDANDO PREVISOES =====", flush=True)

classificador = pipeline(
    "text-classification",
    model=str(DESTINO),
    tokenizer=str(DESTINO),
    framework="pt",
    device=-1
)

indices = [2, 909]
textos_originais = load_from_disk(str(DADOS))

# Recuperar o texto original da base CSV de teste
import pandas as pd
csv = Path(
    r"C:\Users\rafal\Desktop\HUGGING-FACE-TRANSFER-LEARNING-NLP"
) / "dados" / "redacoes_teste.csv"

df = pd.read_csv(csv)
resultados = []

for indice in indices:
    previsao = classificador(
        str(df.iloc[indice]["essay"]),
        truncation=True,
        max_length=limite
    )[0]

    registro = {
        "indice": indice,
        "nota_real": int(df.iloc[indice]["score"]),
        "nota_prevista": int(previsao["label"]),
        "score_modelo": float(previsao["score"])
    }

    resultados.append(registro)
    print(registro, flush=True)

PREVISOES.write_text(
    json.dumps({
        "aula": 12,
        "modelo": str(DESTINO),
        "previsoes": resultados
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nAULA 10 - TREINAMENTO CONCLUIDO", flush=True)
print("AULA 12 - PREVISOES EXECUTADAS", flush=True)