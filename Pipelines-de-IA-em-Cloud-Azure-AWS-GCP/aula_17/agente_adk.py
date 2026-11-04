import asyncio
import datetime
import json
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part

BASE = Path(__file__).resolve().parent
RELATORIO = BASE / "resultado_adk.json"

CIDADES = {
    "sao paulo": {
        "timezone": "America/Sao_Paulo",
        "coords": (-23.5505, -46.6333)
    },
    "nova york": {
        "timezone": "America/New_York",
        "coords": (40.7128, -74.0060)
    },
    "lisboa": {
        "timezone": "Europe/Lisbon",
        "coords": (38.7223, -9.1393)
    },
    "rio de janeiro": {
        "timezone": "America/Sao_Paulo",
        "coords": (-22.9068, -43.1729)
    }
}

CHAMADAS = []


def normalizar(texto):
    return "".join(
        caractere
        for caractere in unicodedata.normalize(
            "NFKD", texto.strip().lower()
        )
        if not unicodedata.combining(caractere)
    )


def get_current_time(city: str) -> dict:
    """Consulta a hora atual de uma cidade cadastrada.

    Args:
        city: Nome da cidade.

    Returns:
        Dicionario com status, cidade e horario.
    """
    chave = normalizar(city)

    if chave not in CIDADES:
        resultado = {
            "status": "error",
            "report": f"Cidade nao cadastrada: {city}"
        }
    else:
        timezone = CIDADES[chave]["timezone"]
        agora = datetime.datetime.now(ZoneInfo(timezone))

        resultado = {
            "status": "success",
            "report": (
                f"Hora em {city}: "
                f"{agora:%Y-%m-%d %H:%M:%S %Z%z}"
            )
        }

    CHAMADAS.append({
        "tool": "get_current_time",
        "city": city,
        "resultado": resultado
    })

    return resultado


def get_weather_online(city: str) -> dict:
    """Consulta temperatura, vento e umidade na Open-Meteo.

    Args:
        city: Nome da cidade.

    Returns:
        Dicionario com status, descricao e origem.
    """
    chave = normalizar(city)

    if chave not in CIDADES:
        resultado = {
            "status": "error",
            "report": f"Sem coordenadas para {city}",
            "source": "Open-Meteo"
        }
    else:
        latitude, longitude = CIDADES[chave]["coords"]

        try:
            resposta = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "wind_speed_10m"
                    ],
                    "timezone": "auto"
                },
                timeout=15
            )

            resposta.raise_for_status()
            dados = resposta.json()
            atual = dados.get("current", {})

            temperatura = atual.get("temperature_2m")
            umidade = atual.get("relative_humidity_2m")
            vento = atual.get("wind_speed_10m")

            if any(
                valor is None
                for valor in (temperatura, umidade, vento)
            ):
                raise ValueError(
                    "Dados meteorologicos incompletos."
                )

            resultado = {
                "status": "success",
                "report": (
                    f"Clima em {city}: {temperatura} C, "
                    f"umidade {umidade}%, "
                    f"vento {vento} km/h."
                ),
                "source": "Open-Meteo",
                "observado_em": atual.get("time")
            }

        except (requests.RequestException, ValueError) as erro:
            resultado = {
                "status": "error",
                "report": f"Falha consultando clima: {erro}",
                "source": "Open-Meteo"
            }

    CHAMADAS.append({
        "tool": "get_weather_online",
        "city": city,
        "resultado": resultado
    })

    return resultado


async def main():
    print("=== AULA 17 - GOOGLE ADK ===")

    from google.adk.models.lite_llm import LiteLlm

    agente = Agent(
        name="agente_clima_hora",
        model=LiteLlm(
            model="ollama_chat/llama3.2:latest",
            api_base="http://127.0.0.1:11434"
        ),
        description=(
            "Agente para consultar horario e clima."
        ),
        instruction=(
            "Voce responde em portugues brasileiro. "
            "Para perguntas sobre hora, use get_current_time. "
            "Para perguntas sobre clima, use get_weather_online. "
            "Nunca invente temperaturas ou horarios. "
            "Se uma ferramenta retornar erro, informe o erro. "
            "Use apenas os valores retornados pelas ferramentas."
        ),
        tools=[
            get_current_time,
            get_weather_online
        ]
    )

    runner = InMemoryRunner(
        agent=agente,
        app_name="aula17_adk"
    )

    sessao = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="aluno_local"
    )

    perguntas = [
        "Qual e a hora atual em Sao Paulo?",
        "Qual e o clima atual em Sao Paulo?",
        "Qual e a hora e o clima em Nova York?",
        "Qual e o clima em Campinas?"
    ]

    resultados = []

    for pergunta in perguntas:
        print("\nPergunta:", pergunta)

        mensagem = Content(
            role="user",
            parts=[Part(text=pergunta)]
        )

        respostas = []

        async for evento in runner.run_async(
            user_id="aluno_local",
            session_id=sessao.id,
            new_message=mensagem
        ):
            if (
                evento.is_final_response()
                and evento.content
                and evento.content.parts
            ):
                texto = "".join(
                    parte.text
                    for parte in evento.content.parts
                    if parte.text
                )

                if texto:
                    respostas.append(texto)

        resposta_final = "\n".join(respostas)

        print("Resposta:", resposta_final)

        resultados.append({
            "pergunta": pergunta,
            "resposta": resposta_final
        })

    relatorio = {
        "framework": "Google ADK",
        "modelo": "llama3.2:latest",
        "provedor": "Ollama local",
        "session_id": sessao.id,
        "perguntas": resultados,
        "chamadas_ferramentas": CHAMADAS
    }

    RELATORIO.write_text(
        json.dumps(
            relatorio,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    print("\n=== VALIDACAO ===")
    print("Perguntas executadas:", len(resultados))
    print("Chamadas de ferramentas:", len(CHAMADAS))
    print(
        "Ferramentas utilizadas:",
        sorted(set(
            item["tool"] for item in CHAMADAS
        ))
    )
    print("Relatorio:", RELATORIO)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    asyncio.run(main())