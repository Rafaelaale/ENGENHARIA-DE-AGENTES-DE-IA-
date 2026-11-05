from pathlib import Path
import argparse
import json

import gradio as gr
import torch
from transformers import pipeline

BASE = Path(r"F:\HUGGING-FACE-TRANSFER-LEARNING-NLP")

MODELO = BASE / "modelos" / "distilbert_redacoes_treinado"
RELATORIO_TREINO = BASE / "resultados" / "aula_10_treinamento.json"

torch.set_num_threads(1)

classificador = None


def treinamento_confirmado():
    arquivos = [
        MODELO / "config.json",
        MODELO / "model.safetensors",
        MODELO / "tokenizer_config.json",
        RELATORIO_TREINO,
    ]

    if not all(p.is_file() for p in arquivos):
        return False

    try:
        relatorio = json.loads(
            RELATORIO_TREINO.read_text(encoding="utf-8")
        )

        epocas = {
            registro.get("epoca")
            for registro in relatorio.get("historico", [])
        }

        return {1, 2}.issubset(epocas)

    except (ValueError, OSError, TypeError):
        return False


def carregar_classificador():
    global classificador

    if classificador is not None:
        return classificador

    if not treinamento_confirmado():
        return None

    classificador = pipeline(
        task="text-classification",
        model=str(MODELO),
        tokenizer=str(MODELO),
        framework="pt",
        device=-1,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )

    return classificador


def mostrar_resultado(texto):
    texto = texto or ""

    if len(texto.strip()) < 400:
        return "O texto precisa ter no minimo 400 caracteres."

    modelo = carregar_classificador()

    if modelo is None:
        return (
            "A avaliacao ainda nao esta disponivel. "
            "Precisamos concluir o treinamento da Aula 10."
        )

    resultado = modelo(
        texto,
        truncation=True,
        max_length=512
    )[0]

    nota = int(resultado["label"])

    if not 0 <= nota <= 10:
        raise ValueError("Nota prevista fora do intervalo esperado.")

    return f"Nota estimada: {nota * 100} / 1000"


def criar_aplicativo():
    return gr.Interface(
        fn=mostrar_resultado,
        inputs=gr.Textbox(
            label="Digite sua redacao",
            placeholder="Escreva sua redacao aqui...",
            lines=15
        ),
        outputs=gr.Textbox(
            label="Resultado da avaliacao"
        ),
        title="Avaliador de Redacoes com Inteligencia Artificial",
        description=(
            "Digite um texto com pelo menos 400 caracteres. "
            "Quando o modelo estiver treinado, "
            "a nota estimada sera apresentada entre 0 e 1000. "
            "Esta avaliacao e experimental e nao substitui "
            "a correcao de um professor."
        ),
        submit_btn="Avaliar redacao",
        clear_btn="Limpar"
    )


def verificar():
    print("\n===== VALIDACAO DO APLICATIVO =====")
    print("Gradio:", gr.__version__)
    print("PyTorch:", torch.__version__)
    print("Modelo treinado:", treinamento_confirmado())

    curto = mostrar_resultado("A")

    assert "400 caracteres" in curto

    print("Validacao de texto curto: OK")

    if not treinamento_confirmado():
        resultado = mostrar_resultado("Texto " * 100)

        assert "Aula 10" in resultado

        print("Bloqueio de notas sem modelo: OK")
        print("Aplicativo preparado; treinamento pendente.")
    else:
        print("Modelo treinado encontrado.")
        print("Aplicativo pronto para executar.")

    print("\nAULA 13 - CODIGO VALIDADO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verificar",
        action="store_true"
    )

    parser.add_argument(
        "--executar",
        action="store_true"
    )

    argumentos = parser.parse_args()

    if argumentos.executar:
        print("\n===== INICIANDO GRADIO LOCAL =====")

        app = criar_aplicativo()

        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            inbrowser=False
        )

    else:
        verificar()