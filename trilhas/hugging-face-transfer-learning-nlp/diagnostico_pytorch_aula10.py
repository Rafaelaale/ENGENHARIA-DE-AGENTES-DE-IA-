import faulthandler
faulthandler.enable()

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

caminho = (
    r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP"
    r"\modelos\distilbert_redacoes_inicial"
)

print("===== DIAGNOSTICO PYTORCH =====", flush=True)
print("Versao:", torch.__version__, flush=True)

torch.set_num_threads(1)
torch.manual_seed(42)

print("Threads:", torch.get_num_threads(), flush=True)
print("Carregando modelo...", flush=True)

tokenizador = AutoTokenizer.from_pretrained(
    caminho,
    local_files_only=True
)

modelo = AutoModelForSequenceClassification.from_pretrained(
    caminho,
    local_files_only=True
)

modelo.train()

print("Modelo carregado.", flush=True)

texto = (
    "A educacao publica e importante para o futuro. "
    "Os estudantes precisam de boas escolas e professores."
)

entradas = tokenizador(
    texto,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

entradas["labels"] = torch.tensor([6])

print("Executando forward...", flush=True)

saida = modelo(**entradas)

print("Loss:", float(saida.loss.item()), flush=True)

print("Executando backward...", flush=True)

saida.loss.backward()

print("Backward concluido.", flush=True)

print("Testando atualizacao simples...", flush=True)

otimizador = torch.optim.SGD(
    modelo.parameters(),
    lr=1e-5
)

otimizador.step()
otimizador.zero_grad(set_to_none=True)

print("Atualizacao concluida.", flush=True)

print(
    "DIAGNOSTICO: FORWARD, BACKWARD E OTIMIZADOR OK",
    flush=True
)