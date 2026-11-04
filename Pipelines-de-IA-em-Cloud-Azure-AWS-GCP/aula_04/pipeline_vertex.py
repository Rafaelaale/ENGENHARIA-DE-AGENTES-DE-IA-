from pathlib import Path
from kfp import compiler, dsl


@dsl.component(base_image="python:3.11-slim")
def sum_numbers(a: float, b: float) -> float:
    return a + b


@dsl.component(base_image="python:3.11-slim")
def format_result(sum_value: float) -> str:
    message = f"A soma e {sum_value}"
    print(message)
    return message


@dsl.pipeline(name="sum-and-format-pipeline")
def sum_and_format_pipeline(a: float = 1.0, b: float = 2.0):
    sum_task = sum_numbers(a=a, b=b)
    sum_task.set_cpu_limit("1")

    format_task = format_result(sum_value=sum_task.output)
    format_task.set_cpu_limit("1")


def main():
    destino = Path(__file__).resolve().parent
    arquivo_json = destino / "sum_and_format_pipeline.json"

    compiler.Compiler().compile(
        pipeline_func=sum_and_format_pipeline,
        package_path=str(arquivo_json)
    )

    print("Pipeline compilado com sucesso.")
    print(f"Arquivo gerado: {arquivo_json}")
    print("Componentes: sum_numbers e format_result")
    print("Parametros padrao: a=1.0, b=2.0")
    print("Execucao na nuvem: nao realizada")


if __name__ == "__main__":
    main()
