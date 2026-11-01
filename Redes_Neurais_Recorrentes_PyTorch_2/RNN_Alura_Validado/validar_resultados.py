import json
import sys
from pathlib import Path
import torch

pasta = Path(sys.argv[1])
arquivo_modelo = pasta / "modelo_rnn.pt"
arquivo_resultados = pasta / "resultados.json"

if not arquivo_modelo.is_file():
    raise FileNotFoundError(arquivo_modelo)

if not arquivo_resultados.is_file():
    raise FileNotFoundError(arquivo_resultados)

dados = json.loads(
    arquivo_resultados.read_text(encoding="utf-8")
)

checkpoint = torch.load(
    arquivo_modelo,
    map_location="cpu",
    weights_only=True
)

assert "pesos" in checkpoint
assert "categorias" in checkpoint
assert len(checkpoint["categorias"]) == len(dados["categorias"])
assert dados["historico"]
assert dados["previsoes"]

print("MODELO VALIDADO")
print("Categorias:", len(dados["categorias"]))
print("Amostras de treino:", dados["treino"])
print("Amostras de teste:", dados["teste"])
print("Epocas executadas:", len(dados["historico"]))
print("Acuracia de teste:", dados["acuracia_teste"])

for nome, previsoes in dados["previsoes"].items():
    print(nome, ":", previsoes)