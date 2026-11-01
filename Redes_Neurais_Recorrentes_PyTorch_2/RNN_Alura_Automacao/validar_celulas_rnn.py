import ast
import contextlib
import io
import json
import re
import traceback
from pathlib import Path

import nbformat

ROOT = Path.cwd()
PASTA = ROOT / "RNN_Alura_Automacao"
ARQUIVO = PASTA / "Classificacao_Sequencias_Corrigido.ipynb"
RELATORIO = PASTA / "diagnostico_celulas.txt"

mensagens = []

def registrar(texto):
    mensagens.append(str(texto))
    print(texto, flush=True)

def salvar_relatorio():
    RELATORIO.write_text(
        "\n".join(mensagens) + "\n",
        encoding="utf-8"
    )

try:
    nb = nbformat.read(str(ARQUIVO), as_version=4)
    nbformat.validate(nb)

    registrar("NOTEBOOK: ESTRUTURA VALIDA")
    registrar(f"TOTAL DE CELULAS: {len(nb.cells)}")

    alteradas = []
    pendentes = []

    for numero, celula in enumerate(nb.cells, 1):
        if celula.cell_type != "code":
            continue

        fonte = celula.source
        anterior = fonte

        # Corrige somente o uso confirmado de decode
        # em dados_arabe, que contem strings.
        fonte = fonte.replace(
            "dados_arabe[0].decode('utf-8')[0]",
            "dados_arabe[0][0]"
        )

        # Corrige LogSoftmax apenas quando falta dim.
        fonte = re.sub(
            r"\bnn\.LogSoftmax\(\s*\)",
            "nn.LogSoftmax(dim=1)",
            fonte
        )

        if fonte != anterior:
            celula.source = fonte
            alteradas.append(numero)

        try:
            ast.parse(fonte)
        except SyntaxError as erro:
            # Comandos % e ! pertencem ao Jupyter.
            if any(
                linha.lstrip().startswith(("%", "!", "?"))
                for linha in fonte.splitlines()
            ):
                pendentes.append(
                    f"Celula {numero}: comando especial Jupyter"
                )
            else:
                pendentes.append(
                    f"Celula {numero}: sintaxe invalida "
                    f"na linha {erro.lineno}: {erro.msg}"
                )

        if ".decode(" in fonte:
            pendentes.append(
                f"Celula {numero}: outro uso de decode "
                "exige verificacao do tipo de dado"
            )

    if alteradas:
        nbformat.write(nb, str(ARQUIVO))

    registrar(f"CELULAS CORRIGIDAS: {alteradas}")
    registrar(f"PENDENCIAS ESTATICAS: {len(pendentes)}")

    for item in pendentes:
        registrar(item)

    # A execucao usa um ambiente compartilhado entre
    # as celulas, como acontece no notebook.
    # Para evitar repeticao de treinamento, para antes
    # de celulas que executam loops de treinamento.
    contexto = {
        "__name__": "__main__",
        "__file__": str(ARQUIVO)
    }

    executadas = 0
    falhas = []

    registrar("")
    registrar("INICIANDO VALIDACAO SEQUENCIAL")

    for numero, celula in enumerate(nb.cells, 1):
        if celula.cell_type != "code":
            continue

        fonte = celula.source.strip()

        if not fonte:
            continue

        # Magicas nao sao codigo Python puro.
        if any(
            linha.lstrip().startswith(("%", "!", "?"))
            for linha in fonte.splitlines()
        ):
            registrar(
                f"CELULA {numero}: NAO EXECUTADA "
                "(comando especial Jupyter)"
            )
            continue

        # Evita executar novamente treinamento
        # do notebook original sem limite de custo.
        if re.search(
            r"(?m)^\s*for\s+\w+\s+in\s+range\s*\("
            r"\s*(?:num_epocas|n_epocas|epochs|epocas)",
            fonte
        ):
            registrar(
                f"CELULA {numero}: NAO EXECUTADA "
                "(possivel treinamento)"
            )
            continue

        try:
            codigo = compile(
                fonte,
                f"<celula {numero}>",
                "exec"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                exec(codigo, contexto)

            executadas += 1
            registrar(f"CELULA {numero}: OK")

        except Exception:
            erro = traceback.format_exc()
            falhas.append(numero)

            registrar(f"CELULA {numero}: ERRO")
            registrar(erro)

            # Interrompe para nao produzir erros em
            # cascata causados pela primeira falha.
            break

    registrar("")
    registrar(f"CELULAS EXECUTADAS: {executadas}")
    registrar(f"FALHAS: {falhas}")

    if falhas:
        registrar("RESULTADO: VALIDACAO INCOMPLETA")
    elif pendentes:
        registrar("RESULTADO: VALIDACAO PARCIAL")
    else:
        registrar(
            "RESULTADO: CELULAS ELEGIVEIS "
            "EXECUTADAS SEM ERRO"
        )

    salvar_relatorio()

    if falhas:
        raise SystemExit(1)

except Exception:
    registrar(traceback.format_exc())
    salvar_relatorio()
    raise