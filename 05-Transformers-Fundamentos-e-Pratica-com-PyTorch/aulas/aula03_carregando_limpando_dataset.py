# ============================================================
# PROJETO 05 - TRANSFORMERS COM PYTORCH
# AULA 03 - CARREGANDO E LIMPANDO O DATASET
# ============================================================

import argparse
import json
import re
import random
import unicodedata

from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================
# 1 - CONFIGURACOES
# ============================================================

BASE = Path(__file__).resolve().parent.parent

DADOS = BASE / "dados"
RESULTADOS = BASE / "resultados"

DATASET_ID = "pablo-moreira/wikipedia-pt"

CONFIG = "default"
SPLIT = "train"

API = (
    "https://datasets-server.huggingface.co/rows"
)

random.seed(42)


# ============================================================
# 2 - FUNCAO DE LIMPEZA COM REGEX
# ============================================================

def limpar_texto(texto):

    if not isinstance(texto, str):
        return ""

    s = unicodedata.normalize(
        "NFC",
        texto
    )

    # Templates da Wikipedia
    s = re.sub(
        r"\{\{[^{}]*\}\}",
        " ",
        s
    )

    # Links internos da Wikipedia
    s = re.sub(
        r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]",
        r"\1",
        s
    )

    # Referencias numericas
    s = re.sub(
        r"\[\d+\]",
        " ",
        s
    )

    # Titulos da Wikipedia
    s = re.sub(
        r"(?m)^\s*={2,}[^\n]*?={2,}\s*$",
        " ",
        s
    )

    # URLs
    s = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        s
    )

    # Emails
    s = re.sub(
        r"\S+@\w+\.\w+",
        " ",
        s
    )

    # Tags HTML
    s = re.sub(
        r"<[^>]*>",
        " ",
        s
    )

    # Entidades HTML
    s = re.sub(
        r"&(?:#\d+|#x[0-9a-fA-F]+|\w+);",
        " ",
        s
    )

    # Caracteres de controle
    s = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        s
    )

    # Normalizacao de pontuacao
    s = s.translate(
        str.maketrans({
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-"
        })
    )

    # Numeros muito longos
    s = re.sub(
        r"\b\d{5,}\b",
        " ",
        s
    )

    # Simbolos matematicos
    s = re.sub(
        r"[=≈≠≤≥∑∫∏√∞∇∂∈∉∩∪⊂⊃∧∨¬∀∃]",
        " ",
        s
    )

    # Caracteres validos
    s = re.sub(
        r"""[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s.,;:!?()\-'"[\]%]""",
        " ",
        s
    )

    # Reticencias
    s = re.sub(
        r"\.{3,}",
        "...",
        s
    )

    # Pontuacao duplicada
    for marca in ",:;!?":

        s = re.sub(
            re.escape(marca) + r"{2,}",
            marca,
            s
        )

    # Espacos antes da pontuacao
    s = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        s
    )

    # Espacos extras
    s = re.sub(
        r"\s+",
        " ",
        s
    ).strip()

    return s


# ============================================================
# 3 - FILTRO DE QUALIDADE DOS TEXTOS
# ============================================================

def texto_valido(texto):

    tamanho = len(texto)

    if not 50 < tamanho < 5000:
        return False

    alfabeticos = sum(
        caractere.isalpha()
        for caractere in texto
    )

    percentual = (
        alfabeticos / tamanho
    )

    if percentual <= 0.70:
        return False

    # Pelo menos uma frase seguida de maiuscula
    possui_frase = re.search(
        r"[.!?]\s+[A-ZÀ-ÖØ-Ý]",
        texto
    )

    return bool(possui_frase)


# ============================================================
# 4 - CLASSE DO DATASET
# ============================================================

class WikipediaPortugueseLoader:

    def __init__(
        self,
        textos,
        origem="Hugging Face Wikipedia PT"
    ):

        self.textos = textos
        self.origem = origem

        self.estatisticas = Counter()

    # ========================================================
    # 5 - LIMPEZA E CONCATENACAO
    # ========================================================

    def carregar_tudo(
        self,
        limite_caracteres=500_000,
        sep_token="[SEP]",
        inserir_sep_final=True
    ):

        if limite_caracteres < 200:
            raise ValueError(
                "Limite minimo: 200 caracteres."
            )

        if not sep_token or "\n" in sep_token:
            raise ValueError(
                "Separador invalido."
            )

        print(
            "\nProcessando textos...",
            flush=True
        )

        corpus_parts = []

        total_len = 0
        textos_adicionados = 0

        prefix = f"{sep_token}\n"
        suffix = "\n"

        # Reservar espaco para o ultimo separador
        reserva_final = (
            len(sep_token) + 1
            if inserir_sep_final
            else 0
        )

        espaco_conteudo = (
            limite_caracteres
            - reserva_final
        )

        # ====================================================
        # 6 - PERCORRER ARTIGOS
        # ====================================================

        for bruto in self.textos:

            self.estatisticas["recebidos"] += 1

            if (
                not isinstance(bruto, str)
                or len(bruto) <= 100
            ):

                self.estatisticas[
                    "descartados_curto_bruto"
                ] += 1

                continue

            t = limpar_texto(bruto)

            if not texto_valido(t):

                self.estatisticas[
                    "descartados_filtro"
                ] += 1

                continue

            self.estatisticas["aprovados"] += 1

            bloco = (
                f"{prefix}{t}{suffix}"
            )

            bloco_len = len(bloco)

            # =================================================
            # 7 - ARTIGO COMPLETO
            # =================================================

            if (
                total_len + bloco_len
                <= espaco_conteudo
            ):

                corpus_parts.append(bloco)

                total_len += bloco_len

                textos_adicionados += 1

                continue

            # =================================================
            # 8 - CORTE EM FRONTEIRA DE FRASE
            # =================================================

            restante = (
                espaco_conteudo
                - total_len
                - len(prefix)
                - len(suffix)
            )

            if restante > 100:

                parcial = t[:restante]

                ponto = max(
                    parcial.rfind("."),
                    parcial.rfind("!"),
                    parcial.rfind("?")
                )

                if ponto > 0:

                    bloco = (
                        f"{prefix}"
                        f"{parcial[:ponto + 1].strip()}"
                        f"{suffix}"
                    )

                    corpus_parts.append(bloco)

                    total_len += len(bloco)

                    textos_adicionados += 1

            break

        # =====================================================
        # 9 - FORMAR CORPUS
        # =====================================================

        corpus = "".join(
            corpus_parts
        ).strip()

        # Separador final
        if inserir_sep_final and corpus:

            final = (
                "\n"
                if not corpus.endswith("\n")
                else ""
            ) + sep_token + "\n"

            if (
                len(corpus) + len(final)
                <= limite_caracteres
            ):

                corpus += final

        if len(corpus) > limite_caracteres:

            raise AssertionError(
                "Corpus excedeu o limite."
            )

        self.estatisticas[
            "textos_no_corpus"
        ] = textos_adicionados

        self.estatisticas[
            "caracteres"
        ] = len(corpus)

        print(
            "\nProcessamento concluido!"
        )

        print(
            "Textos utilizados:",
            textos_adicionados
        )

        print(
            "Caracteres:",
            len(corpus)
        )

        return corpus


# ============================================================
# 10 - LER WIKIPEDIA EM PEQUENOS LOTES
# ============================================================

def iterar_api(
    lote=10,
    max_artigos=250
):

    offset = 0

    while offset < max_artigos:

        length = min(
            lote,
            max_artigos - offset,
            100
        )

        parametros = urlencode({
            "dataset": DATASET_ID,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length
        })

        url = (
            API + "?" + parametros
        )

        requisicao = Request(
            url,
            headers={
                "User-Agent": (
                    "Projeto05-PyTorch-Transformers/1.0"
                ),
                "Accept": "application/json"
            }
        )

        try:

            with urlopen(
                requisicao,
                timeout=25
            ) as resposta:

                # Limite de seguranca por lote
                tamanho_maximo = (
                    12 * 1024 * 1024
                )

                conteudo = resposta.read(
                    tamanho_maximo + 1
                )

                if (
                    len(conteudo)
                    > tamanho_maximo
                ):

                    raise RuntimeError(
                        "Lote muito grande. "
                        "Execute com --lote 1."
                    )

                dados = json.loads(
                    conteudo.decode("utf-8")
                )

        except (
            HTTPError,
            URLError,
            TimeoutError
        ) as erro:

            raise RuntimeError(
                "Falha na API do Hugging Face: "
                + str(erro)
            ) from erro

        linhas = dados.get(
            "rows",
            []
        )

        if not linhas:
            return

        for linha in linhas:

            artigo = linha.get(
                "row",
                {}
            )

            yield artigo.get(
                "text",
                ""
            )

        offset += len(linhas)

        print(
            f"Artigos consultados: {offset}",
            flush=True
        )

        if len(linhas) < length:
            return


# ============================================================
# 11 - TESTES LOCAIS SEM DOWNLOAD
# ============================================================

def teste_local():

    print("=" * 60)
    print("AULA 03 - TESTE LOCAL")
    print("=" * 60)

    exemplos = [

        (
            "A inteligência artificial estuda modelos "
            "computacionais para processar informações. "
            "Os Transformers utilizam atenção para "
            "compreender relações entre palavras diferentes. "
            "Esses métodos são investigados em tarefas "
            "de linguagem natural."
        ),

        (
            "A [[matemática|Matemática]] utiliza modelos "
            "abstratos para representar fenômenos. "
            "As pesquisas fornecem instrumentos úteis "
            "para aplicações em ciência de dados. "
            "Muitos modelos passam por avaliação antes "
            "da utilização. [12] == Referências =="
        ),

        (
            "O aprendizado de máquina aplica algoritmos "
            "para identificar padrões presentes nos dados. "
            "A pesquisa moderna usa textos em português "
            "e métodos estatísticos. "
            "Os resultados dependem de qualidade "
            "e preparação dos exemplos."
        )

    ]

    loader = WikipediaPortugueseLoader(
        exemplos,
        "exemplos locais"
    )

    corpus = loader.carregar_tudo(
        limite_caracteres=1000
    )

    assert (
        corpus.count("[SEP]") >= 2
    )

    assert (
        "[[matemática" not in corpus
    )

    assert (
        "[12]" not in corpus
    )

    assert len(corpus) <= 1000

    assert (
        "Matemática" in corpus
    )

    assert (
        limpar_texto(
            "Olá [12], mundo!"
        )
        == "Olá, mundo!"
    )

    print(
        "\nSeparador [SEP]: OK"
    )

    print(
        "Limpeza com Regex: OK"
    )

    print(
        "Filtragem dos textos: OK"
    )

    print(
        "Limite de caracteres: OK"
    )

    print(
        "\nTrecho do corpus:"
    )

    print(
        corpus[:300]
    )

    print(
        "\nAULA 03 - TESTE CONCLUIDO!"
    )


# ============================================================
# 12 - EXECUCAO PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modo",
        choices=["teste", "real"],
        default="teste"
    )

    parser.add_argument(
        "--limite",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--lote",
        type=int,
        default=10
    )

    parser.add_argument(
        "--max-artigos",
        type=int,
        default=250
    )

    args = parser.parse_args()

    if args.modo == "teste":

        teste_local()
        return

    if not 200 <= args.limite <= 500_000:

        parser.error(
            "Limite entre 200 e 500000."
        )

    if not 1 <= args.lote <= 100:

        parser.error(
            "Lote entre 1 e 100."
        )

    if not 1 <= args.max_artigos <= 2000:

        parser.error(
            "Max-artigos entre 1 e 2000."
        )

    print("=" * 60)
    print("AULA 03 - WIKIPEDIA PORTUGUES")
    print("=" * 60)

    print(
        "Dataset:",
        DATASET_ID
    )

    print(
        "Subset:",
        CONFIG
    )

    print(
        "Limite de caracteres:",
        args.limite
    )

    print(
        "Acesso por pequenos lotes."
    )

    # ========================================================
    # 13 - CARREGAR TEXTOS REAIS
    # ========================================================

    loader = WikipediaPortugueseLoader(
        iterar_api(
            args.lote,
            args.max_artigos
        )
    )

    corpus = loader.carregar_tudo(
        limite_caracteres=args.limite
    )

    if not corpus:

        raise RuntimeError(
            "Nenhum texto passou pelos filtros. "
            "Tente --max-artigos 500."
        )

    # ========================================================
    # 14 - SALVAR RESULTADOS
    # ========================================================

    DADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivo_texto = (
        DADOS
        / f"wikipedia_pt_limpo_{args.limite}.txt"
    )

    arquivo_json = (
        RESULTADOS
        / f"aula03_resumo_{args.limite}.json"
    )

    arquivo_texto.write_text(
        corpus,
        encoding="utf-8"
    )

    relatorio = {

        "aula": 3,

        "dataset": DATASET_ID,

        "subset": CONFIG,

        "split": SPLIT,

        "limite_caracteres": args.limite,

        "caracteres": len(corpus),

        "estatisticas": dict(
            loader.estatisticas
        ),

        "arquivo_corpus": str(
            arquivo_texto.relative_to(BASE)
        ),

        "data_processamento": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "observacao": (
            "Amostra do dataset. "
            "Nao representa a Wikipedia inteira."
        )
    }

    arquivo_json.write_text(
        json.dumps(
            relatorio,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # ========================================================
    # 15 - EXIBIR RESULTADOS
    # ========================================================

    print(
        "\nCorpus salvo:",
        arquivo_texto
    )

    print(
        "Relatorio salvo:",
        arquivo_json
    )

    print(
        "Total de caracteres:",
        len(corpus)
    )

    print(
        "Artigos no corpus:",
        loader.estatisticas["textos_no_corpus"]
    )

    print(
        "\nPrimeiros 400 caracteres:"
    )

    print(
        corpus[:400]
    )

    print(
        "\nAULA 03 - DATASET PROCESSADO!"
    )


if __name__ == "__main__":

    main()

