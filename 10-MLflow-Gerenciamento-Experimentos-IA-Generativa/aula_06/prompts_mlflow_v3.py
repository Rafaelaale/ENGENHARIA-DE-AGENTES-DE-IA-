from pathlib import Path
import mlflow

PROJETO = Path(__file__).resolve().parent.parent
BANCO = PROJETO / "mlflow.db"

mlflow.set_tracking_uri("sqlite:///" + BANCO.as_posix())
mlflow.set_registry_uri(mlflow.get_tracking_uri())

EXPERIMENTO = "aula_06_mlflow_v3_genai"
NOME_PROMPT = "assistente_tecnologia_educativo"

mlflow.set_experiment(EXPERIMENTO)

print("===== ALURA - AULA 6 =====")
print("CONHECENDO MLFLOW V3")
print("Versao MLflow:", mlflow.__version__)
print("Tracking URI:", mlflow.get_tracking_uri())

prompts = [
    {
        "versao": "v1",
        "template": (
            "Voce e um assistente especializado em {domain}. "
            "Responda: {question}"
        )
    },
    {
        "versao": "v2",
        "template": (
            "Voce e um assistente especializado em {domain}. "
            "Explique de forma clara, objetiva e com um exemplo pratico. "
            "Pergunta: {question}"
        )
    }
]

print("\n===== REGISTRANDO PROMPTS =====")

for item in prompts:

    versao = item["versao"]
    template = item["template"]

    prompt_final = template.format(
        domain="tecnologia",
        question="Como funciona o MLflow?"
    )

    with mlflow.start_run(run_name=f"prompt_{versao}") as run:

        mlflow.log_param("prompt_version", versao)
        mlflow.log_param("domain", "tecnologia")
        mlflow.log_param("model_type", "generative_ai")
        mlflow.log_text(template, "prompt_template.txt")
        mlflow.log_text(prompt_final, "final_prompt.txt")
        mlflow.set_tag("lesson", "aula_06")
        mlflow.set_tag("prompt_name", NOME_PROMPT)

        print(f"\nVersao: {versao}")
        print("Run ID:", run.info.run_id)
        print("Prompt:", prompt_final)

print("\n===== PROMPT REGISTRY =====")

if not hasattr(mlflow, "genai") or not hasattr(mlflow.genai, "register_prompt"):
    raise RuntimeError(
        "A instalacao atual nao disponibiliza mlflow.genai.register_prompt."
    )

for item in prompts:

    registro = mlflow.genai.register_prompt(
        name=NOME_PROMPT,
        template=item["template"],
        commit_message=f"Aula 6 - {item['versao']}"
    )

    print(
        f"Prompt {item['versao']} registrado "
        f"na versao {registro.version}"
    )

print("\n===== COMPARACAO =====")

print("Versao 1: resposta direta.")
print("Versao 2: resposta com explicacao e exemplo.")
print("Nenhum LLM foi executado nesta aula.")
print("Nenhuma metrica de qualidade foi calculada.")

print("\n===== AULA 6 CONCLUIDA =====")
