import json
import urllib.request
from typing import Optional, Dict, Any

BANCO_DADOS_CLIENTES = {
    "12345678900": {"nome": "Rafael", "conta_ativa": True, "score_credito": 820},
    "98765432100": {"nome": "Carlos", "conta_ativa": False, "score_credito": 450}
}

def tool_consultar_cadastro(cpf: str) -> Dict[str, Any]:
    cliente = BANCO_DADOS_CLIENTES.get(cpf)
    if cliente:
        return {"status": "encontrado", **cliente}
    return {"status": "nao_encontrado", "conta_ativa": False}

def tool_calcular_limite(score: int) -> float:
    if score >= 800:
        return 10000.0
    elif score >= 600:
        return 3500.0
    return 800.0

def agente_abertura_conta(mensagem: str, cpf: Optional[str] = None) -> str:
    return f"[Agente Abertura] Iniciando abertura de conta para CPF {cpf or 'não informado'}."

def agente_cartao_credito(mensagem: str, cpf: Optional[str] = None) -> str:
    if not cpf:
        return "[Agente Cartão] Por favor, informe seu CPF para verificarmos se você já possui conta."
    dados = tool_consultar_cadastro(cpf)
    if not dados["conta_ativa"]:
        return f"[Agente Cartão] CPF {cpf}: nenhuma conta ativa encontrada. Redirecionando para abertura de conta."
    limite = tool_calcular_limite(dados["score_credito"])
    return f"[Agente Cartão] Olá {dados['nome']}! Conta validada. Limite pré-aprovado: R$ {limite:,.2f}."

def agente_suporte(mensagem: str) -> str:
    return f"[Agente Suporte] Atendimento geral para a dúvida: '{mensagem}'"

def classificar_com_llm(mensagem: str) -> str:
    prompt = f"""Você é o roteador do MD Bank. Classifique a mensagem do usuário estritamente em uma destas 3 categorias:
- abertura_conta
- cartao_credito
- suporte_geral

Responda APENAS o nome exato da categoria em minúsculas, sem pontuação e sem explicações.

Mensagem: "{mensagem}"
Categoria:"""

    payload = json.dumps({
        "model": "gemma3:1b",
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rota = data.get("response", "").strip().lower()
            for opcao in ["abertura_conta", "cartao_credito", "suporte_geral"]:
                if opcao in rota:
                    return opcao
            return "suporte_geral"
    except Exception:
        # Fallback local se Ollama estiver desligado
        msg = mensagem.lower()
        if any(k in msg for k in ["abrir", "abro", "conta", "cadastro"]):
            return "abertura_conta"
        if any(k in msg for k in ["cartao", "cartão", "limite", "credito"]):
            return "cartao_credito"
        return "suporte_geral"

def roteador_md_bank(mensagem: str, cpf: Optional[str] = None) -> str:
    rota = classificar_com_llm(mensagem)

    if rota == "abertura_conta":
        return agente_abertura_conta(mensagem, cpf)
    elif rota == "cartao_credito":
        return agente_cartao_credito(mensagem, cpf)
    else:
        return agente_suporte(mensagem)

if __name__ == "__main__":
    testes = [
        ("Como abro uma conta?", "11122233344"),
        ("Gostaria de solicitar cartão", "12345678900"),
        ("Qual o telefone da central?", None)
    ]
    for msg, cpf in testes:
        print(f"\n[Cliente]: {msg}")
        print(f"[MD Bank]: {roteador_md_bank(msg, cpf)}")
