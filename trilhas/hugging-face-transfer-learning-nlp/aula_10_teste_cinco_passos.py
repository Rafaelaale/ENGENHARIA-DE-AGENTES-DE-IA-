import faulthandler
faulthandler.enable()

from pathlib import Path
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

BASE = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")

DADOS = BASE / "dados" / "redacoes_prontas_treinamento"
MODELO = BASE / "modelos" / "distilbert_redacoes_inicial"

torch.set_num_threads(1)
torch.manual_seed(42)

print("\n===== AULA 10 - TESTE DE 5 PASSOS =====", flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("Threads:", torch.get_num_threads(), flush=True)
print("Dispositivo: CPU", flush=True)

print("\n===== CARREGANDO DADOS =====", flush=True)

dados = load_from_disk(str(DADOS))
treino = dados["treino"]

assert len(treino) == 3656

print("Redacoes:", len(treino), flush=True)

print("\n===== CARREGANDO MODELO =====", flush=True)

modelo = AutoModelForSequenceClassification.from_pretrained(
    str(MODELO),
    local_files_only=True
)

assert modelo.config.num_labels == 11

modelo.train()

print("Modelo carregado.", flush=True)

print("\n===== CONFIGURANDO ADAMW =====", flush=True)

otimizador = torch.optim.AdamW(
    modelo.parameters(),
    lr=2e-5
)

print("Otimizador configurado.", flush=True)

print("\n===== INICIANDO CINCO PASSOS =====", flush=True)

for passo in range(5):

    registro = treino[passo]

    ids = registro["input_ids"][:128]
    mascara = registro["attention_mask"][:128]

    entradas = {
        "input_ids": torch.tensor(
            [ids],
            dtype=torch.long
        ),
        "attention_mask": torch.tensor(
            [mascara],
            dtype=torch.long
        ),
        "labels": torch.tensor(
            [registro["label"]],
            dtype=torch.long
        )
    }

    otimizador.zero_grad(set_to_none=True)

    print(
        f"Iniciando passo {passo + 1}/5",
        flush=True
    )

    saida = modelo(**entradas)

    perda = saida.loss

    if not torch.isfinite(perda):
        raise RuntimeError("Loss invalida.")

    perda.backward()

    torch.nn.utils.clip_grad_norm_(
        modelo.parameters(),
        max_norm=1.0
    )

    otimizador.step()

    print(
        f"Passo {passo + 1}/5 | "
        f"Nota real: {registro['label']} | "
        f"Loss: {perda.item():.4f}",
        flush=True
    )

print("\n===== RESULTADO FINAL =====", flush=True)
print("Cinco atualizacoes concluidas.", flush=True)
print("Modelo original preservado.", flush=True)
print("Nenhum checkpoint foi alterado.", flush=True)

print("\nTESTE DE 5 PASSOS CONCLUIDO", flush=True)