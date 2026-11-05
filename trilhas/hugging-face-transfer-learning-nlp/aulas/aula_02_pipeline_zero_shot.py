import json
from pathlib import Path
from transformers import pipeline

print("===== HUGGING FACE - ZERO SHOT =====")

modelo = "Mel-Iza0/zero-shot"

classificador = pipeline(
    task="zero-shot-classification",
    model=modelo,
    tokenizer=modelo,
    framework="pt",
    device=-1
)

# Redacao de demonstracao baseada no tema da aula.
texto = """
Com o inicio da era digital, a transmissao de informacoes
facilitou o acesso dos jovens a temas como a educacao sexual.
Entretanto, a ausencia de dialogo com pais e professores
contribui para a disseminacao de informacoes inadequadas.

A educacao sexual nas escolas pode ajudar os estudantes
a compreender os riscos de infeccoes sexualmente
transmissiveis e da gravidez precoce.

Portanto, o Ministerio da Educacao, em parceria com
as escolas e as familias, deve promover aulas e palestras
sobre sexualidade, com orientacao de profissionais
qualificados e espaco para esclarecer duvidas.
"""

notas = [str(i) for i in range(11)]

print("\n===== CLASSIFICANDO REDACAO =====")

resultado = classificador(
    texto,
    candidate_labels=notas,
    hypothesis_template="A nota atribuida a esta redacao e {}."
)

print("\n===== CLASSIFICACAO =====")

for nota, score in zip(
    resultado["labels"],
    resultado["scores"]
):
    print(f"Nota {nota}: score {score:.6f}")

print("\nRotulo mais bem classificado:", resultado["labels"][0])

print(
    "\nATENCAO: Estes scores nao representam "
    "notas escolares validadas."
)

saida = Path("resultado_pipeline_zero_shot.json")

saida.write_text(
    json.dumps(
        resultado,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\nResultado salvo em:", saida.resolve())
print("\nAULA 02 - CLASSIFICACAO EXECUTADA")