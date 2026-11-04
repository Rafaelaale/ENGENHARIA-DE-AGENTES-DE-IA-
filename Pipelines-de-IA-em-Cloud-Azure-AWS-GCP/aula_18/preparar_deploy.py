import importlib
import json
import py_compile
from pathlib import Path

BASE = Path(__file__).resolve().parent

ROOT = BASE / "root_agent.py"
REQUIREMENTS = BASE / "requirements.txt"
MANIFESTO = BASE / "manifesto_deploy.json"

py_compile.compile(str(ROOT), doraise=True)

modulo = importlib.import_module("root_agent")
agente = modulo.root_agent

ferramentas = [
    getattr(tool, "__name__", type(tool).__name__)
    for tool in agente.tools
]

resultado_mock = modulo.get_weather_online("Cidade Inexistente")

assert agente.name == "agente_clima_aula18"
assert "get_weather_online" in ferramentas
assert resultado_mock["status"] == "error"

manifesto = {
    "aula": 18,
    "framework": "Google ADK",
    "agent_name": agente.name,
    "model": "ollama_chat/llama3.2:latest",
    "execution_mode": "local",
    "cloud_deployed": False,
    "cloud_ready": False,
    "tools": ferramentas,
    "requirements_file": REQUIREMENTS.name,
    "agent_file": ROOT.name,
    "checks": {
        "python_syntax": True,
        "agent_import": True,
        "tool_registered": True,
        "unknown_city_handled": True
    },
    "deployment_blockers": [
        "Ollama utiliza endereco localhost, inacessivel no Agent Engine remoto.",
        "Projeto Google Cloud nao configurado.",
        "Autenticacao e permissoes nao verificadas.",
        "Bucket de staging nao configurado.",
        "Dependencias ainda nao foram validadas em ambiente remoto."
    ]
}

MANIFESTO.write_text(
    json.dumps(
        manifesto,
        ensure_ascii=False,
        indent=4
    ),
    encoding="utf-8"
)

print("=== AULA 18 - PREPARACAO DO AGENTE ===")
print("Agente:", agente.name)
print("Framework: Google ADK")
print("Ferramentas:", ferramentas)
print("Teste cidade inexistente:", resultado_mock["status"])
print("Sintaxe: OK")
print("Importacao do agente: OK")
print("Implantado na Cloud: NAO")
print("Pronto para deploy remoto: NAO")
print("Manifesto:", MANIFESTO)