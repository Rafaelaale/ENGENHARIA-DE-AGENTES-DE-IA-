from pathlib import Path
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

raiz = Path(__file__).resolve().parent.parent

arquivo_csv = raiz / "dados" / "redacoes_treino.csv"

arquivo_json = raiz / "resultados" / "aula_06_tokens_ids.json"
arquivo_txt = raiz / "resultados" / "aula_06_relatorio_tokenizacao.txt"

checkpoint_modelo = "Geotrend/distilbert-base-pt-cased"

print("\n===== HUGGING FACE - AULA 06 =====")
print("TOKENIZANDO DADOS TEXTUAIS")

# 1. CARREGAR BASE DE TREINAMENTO

print("\n===== CARREGANDO DADOS =====")

dados_redacoes = load_dataset(
    "csv",
    data_files=str(arquivo_csv)
)

treino = dados_redacoes["train"]

assert len(treino) == 3656
assert "essay" in treino.column_names
assert "score" in treino.column_names

texto = treino["essay"][0]
nota_original = treino["score"][0]

print("Redacoes de treinamento:", len(treino))
print("Nota da primeira redacao:", nota_original)

# 2. CARREGAR TOKENIZADOR DO DISTILBERT

print("\n===== CARREGANDO TOKENIZADOR =====")

tokenizador = AutoTokenizer.from_pretrained(
    checkpoint_modelo,
    use_fast=True
)

print("Checkpoint:", checkpoint_modelo)
print("Tokenizador:", type(tokenizador).__name__)
print("Vocabulario:", tokenizador.vocab_size)

# 3. TRANSFORMAR TEXTO EM TOKENS

print("\n===== TOKENIZACAO =====")

tokens = tokenizador.tokenize(texto)

print("Total de tokens:", len(tokens))
print("Primeiros 40 tokens:")
print(tokens[:40])

# 4. CONVERTER TOKENS EM IDs

print("\n===== CONVERTENDO TOKENS EM IDs =====")

ids = tokenizador.convert_tokens_to_ids(tokens)

assert len(tokens) == len(ids)

print("Primeiros 40 IDs:")
print(ids[:40])

# 5. DECODIFICAR

print("\n===== DECODIFICANDO TEXTO =====")

texto_decodificado = tokenizador.decode(ids)

print("Primeiros 500 caracteres:")
print(texto_decodificado[:500])

# 6. GERAR INPUT_IDS E ATTENTION_MASK

print("\n===== CODIFICANDO COM PYTORCH =====")

input_codificado = tokenizador(
    texto,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    return_attention_mask=True
)

input_ids = input_codificado["input_ids"]
attention_mask = input_codificado["attention_mask"]

print("Tipo input_ids:", type(input_ids))
print("Dimensoes input_ids:", tuple(input_ids.shape))
print("Dimensoes attention_mask:", tuple(attention_mask.shape))

print("\nPrimeiros 40 input_ids:")
print(input_ids[0][:40].tolist())

print("\nPrimeiros 40 valores da attention_mask:")
print(attention_mask[0][:40].tolist())

assert isinstance(input_ids, torch.Tensor)
assert isinstance(attention_mask, torch.Tensor)
assert input_ids.shape == attention_mask.shape
assert input_ids.shape[0] == 1
assert input_ids.shape[1] <= 512

# 7. SALVAR RESULTADOS

print("\n===== SALVANDO RESULTADOS =====")

resultado = {
    "aula": 6,
    "checkpoint": checkpoint_modelo,
    "total_redacoes_treino": len(treino),
    "nota_primeira_redacao": int(nota_original),
    "total_tokens_sem_tokens_especiais": len(tokens),
    "tokens": tokens,
    "ids": ids,
    "texto_decodificado": texto_decodificado,
    "input_ids": input_ids[0].tolist(),
    "attention_mask": attention_mask[0].tolist(),
    "shape": list(input_ids.shape),
    "max_length": 512
}

for arquivo in (arquivo_json, arquivo_txt):
    if arquivo.exists():
        raise FileExistsError(arquivo)

arquivo_json.write_text(
    json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

relatorio = [
    "HUGGING FACE - AULA 06",
    "TOKENIZACAO DE DADOS TEXTUAIS",
    "",
    f"Modelo: {checkpoint_modelo}",
    f"Total de redacoes no treino: {len(treino)}",
    f"Nota da primeira redacao: {nota_original}",
    f"Quantidade de tokens: {len(tokens)}",
    f"Shape input_ids: {tuple(input_ids.shape)}",
    f"Shape attention_mask: {tuple(attention_mask.shape)}",
    "",
    "PRIMEIROS 40 TOKENS",
    str(tokens[:40]),
    "",
    "PRIMEIROS 40 IDS",
    str(ids[:40]),
    "",
    "PRIMEIROS 40 VALORES DA MASCARA",
    str(attention_mask[0][:40].tolist()),
    "",
    "DECODIFICACAO - PRIMEIROS 500 CARACTERES",
    texto_decodificado[:500],
    "",
    "OBSERVACOES",
    "Tokenizacao realizada com DistilBERT.",
    "O attention_mask indica posicoes validas ou padding.",
    "Ele nao representa os pesos de atencao do modelo.",
    "Foi usada a estrutura de tensores PyTorch.",
    "O limite de 512 tokens evita entradas excessivas.",
    "Apenas a primeira redacao foi tokenizada nesta aula.",
    "Nenhum modelo foi treinado nesta etapa."
]

arquivo_txt.write_text(
    "\n".join(relatorio),
    encoding="utf-8"
)

for arquivo in (arquivo_json, arquivo_txt):
    assert arquivo.is_file()
    assert arquivo.stat().st_size > 0

print("\n===== ARQUIVOS GERADOS =====")

for arquivo in (arquivo_json, arquivo_txt):
    print(f"{arquivo.name}: {arquivo.stat().st_size} bytes")

print("\nAULA 06 - TOKENIZACAO VALIDADA COM SUCESSO")