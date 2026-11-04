import unicodedata
import requests

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm


CITY_COORDS = {
    "sao paulo": (-23.5505, -46.6333),
    "nova york": (40.7128, -74.0060),
    "lisboa": (38.7223, -9.1393),
    "rio de janeiro": (-22.9068, -43.1729),
    "porto alegre": (-30.0346, -51.2177),
}


def norm_city(city: str) -> str:
    return "".join(
        c for c in unicodedata.normalize(
            "NFKD", city.strip().lower()
        )
        if not unicodedata.combining(c)
    )


def get_weather_online(city: str) -> dict:
    """Consulta o clima atual de uma cidade cadastrada.

    Args:
        city: Nome da cidade.

    Returns:
        Dicionario com status, report e source.
    """
    key = norm_city(city)

    if key not in CITY_COORDS:
        return {
            "status": "error",
            "report": f"Cidade nao cadastrada: {city}",
            "source": "Open-Meteo"
        }

    latitude, longitude = CITY_COORDS[key]

    try:
        response = requests.get(
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

        response.raise_for_status()

        current = response.json().get("current", {})

        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")

        if any(
            value is None
            for value in (temperature, humidity, wind)
        ):
            raise ValueError("Dados meteorologicos incompletos.")

        return {
            "status": "success",
            "report": (
                f"Clima em {city}: {temperature} C, "
                f"umidade {humidity}%, vento {wind} km/h."
            ),
            "source": "Open-Meteo",
            "observed_at": current.get("time")
        }

    except (requests.RequestException, ValueError) as error:
        return {
            "status": "error",
            "report": f"Falha na consulta meteorologica: {error}",
            "source": "Open-Meteo"
        }


root_agent = Agent(
    name="agente_clima_aula18",
    model=LiteLlm(
        model="ollama_chat/llama3.2:latest",
        api_base="http://127.0.0.1:11434"
    ),
    description="Agente ADK para consultar clima via Open-Meteo.",
    instruction=(
        "Responda em portugues brasileiro. "
        "Para consultar o clima, sempre utilize get_weather_online. "
        "Nunca invente temperatura, umidade ou vento. "
        "Se a ferramenta retornar erro, informe o erro. "
        "Utilize somente os dados retornados pela ferramenta."
    ),
    tools=[get_weather_online]
)