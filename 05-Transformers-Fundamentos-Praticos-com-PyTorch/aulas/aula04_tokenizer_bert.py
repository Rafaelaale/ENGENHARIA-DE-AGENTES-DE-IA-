"""
Projeto 05 - Transformers com PyTorch
Aula 04 - Tokenizador BERT Pre-Treinado
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


BASE = Path(__file__).resolve().parent.parent
DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

MODELO = "neuralmind/bert-large-portuguese-cased"


class BertTokenizerWrapper:

    def __init__(self):

        print("Carregando tokenizador BERTimbau...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODELO,
            use_fast=True
        )

        self.tokenizer.clean_up_tokenization_spaces = False

        if self.tokenizer.sep_token_id is not None:
            self.tokenizer.eos_token = self.tokenizer.sep_token

        if self.tokenizer.cls_token_id is not None:
            self.tokenizer.bos_token = self.tokenizer.cls_token

        self.vocab_size = len(self.tokenizer)
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> List[int]:

        return self.tokenizer.encode(
            text,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False
        )

    def decode(self, ids: List[int]) -> str:

        return self.tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512
    ) -> Dict:

        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False
        )


def main():

    print("=" * 60)
    print("AULA 04 - TOKENIZADOR BERT")
    print("=" * 60)

    tokenizer = BertTokenizerWrapper()

    print("\nVocabulario:", tokenizer.vocab_size)
    print("PAD ID:", tokenizer.pad_id)
    print("BOS ID:", tokenizer.bos_id)
    print("EOS ID:", tokenizer.eos_id)

    # Teste com frase

    texto = "Eu amo PIZZA!"

    encoded = tokenizer.encode(texto)
    decoded = tokenizer.decode(encoded)

    tokens = tokenizer.tokenizer.convert_ids_to_tokens(
        encoded
    )

    print("\nTexto original:", texto)
    print("IDs:", encoded)
    print("Tokens:", tokens)
    print("Decodificado:", decoded)

    # Teste com separador

    texto_sep = "[SEP] Eu amo PIZZA!"

    ids_sep = tokenizer.encode(texto_sep)
    texto_recuperado = tokenizer.decode(ids_sep)

    print("\nCom separador:", texto_sep)
    print("IDs:", ids_sep)
    print("Sem separador:", texto_recuperado)

    assert tokenizer.eos_id in ids_sep
    assert "[SEP]" not in texto_recuperado

    print("\nTeste do separador: OK")

    # Carregar corpus da Aula 03

    caminho = DADOS / "wikipedia_pt_limpo_5000.txt"

    if not caminho.exists():
        raise FileNotFoundError(
            f"Corpus nao encontrado: {caminho}"
        )

    corpus = caminho.read_text(
        encoding="utf-8-sig"
    )

    artigos = [
        trecho.strip()
        for trecho in corpus.split("[SEP]")
        if trecho.strip()
    ]

    if not artigos:
        raise RuntimeError("Corpus vazio.")

    print("\nArtigos encontrados:", len(artigos))

    # Tokenizar todos os artigos em lote

    batch = tokenizer.batch_encode(
        artigos,
        max_length=128
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print("\nFormato de input_ids:", tuple(input_ids.shape))
    print("Formato da mascara:", tuple(attention_mask.shape))

    print(
        "\nPrimeiros 20 IDs:",
        input_ids[0][:20].tolist()
    )

    print(
        "Primeiros 20 valores da mascara:",
        attention_mask[0][:20].tolist()
    )

    # Validar resultados

    assert input_ids.ndim == 2
    assert attention_mask.ndim == 2
    assert input_ids.shape == attention_mask.shape
    assert input_ids.shape[0] == len(artigos)
    assert input_ids.shape[1] <= 128

    print("\nValidacoes PyTorch: OK")

    # Salvar resultados

    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    relatorio = {
        "aula": 4,
        "tokenizador": MODELO,
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "texto_exemplo": texto,
        "ids_exemplo": encoded,
        "tokens_exemplo": tokens,
        "texto_decodificado": decoded,
        "artigos": len(artigos),
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "shape": list(input_ids.shape)
    }

    destino = RESULTADOS / "aula04_tokenizacao_bert.json"

    destino.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nArquivo salvo:", destino)
    print("\nAULA 04 - EXECUCAO CONCLUIDA!")


if __name__ == "__main__":
    main()
