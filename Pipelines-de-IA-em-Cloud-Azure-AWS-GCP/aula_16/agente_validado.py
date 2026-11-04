import json
import sqlite3
import urllib.request
from pathlib import Path
from decimal import Decimal
from time import perf_counter

BASE = Path(__file__).resolve().parent
BANCO = BASE / "clientes.db"
SAIDA = BASE / "resultado_agente_validado.json"
MODELO = "llama3.2:latest"

def consultar_cliente(nome):
    with sqlite3.connect(BANCO) as con:
        registro = con.execute(
            "SELECT id, nome FROM clientes WHERE LOWER(nome)=LOWER(?)",
            (nome,)
        ).fetchone()
    return registro

def consultar_pedidos(cliente_id):
    with sqlite3.connect(BANCO) as con:
        return con.execute(
            "SELECT id, valor_centavos FROM pedidos WHERE cliente_id=? ORDER BY id",
            (cliente_id,)
        ).fetchall()

def calcular_total(pedidos):
    if not pedidos:
        raise ValueError("Nao e permitido calcular lista vazia.")
    total_centavos = sum(valor for _, valor in pedidos)
    return {
        "quantidade": len(pedidos),
        "total": float(Decimal(total_centavos) / Decimal(100))
    }

def chamar_llm(mensagens, ferramentas=None):
    payload = {
        "model": MODELO,
        "messages": mensagens,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 250}
    }
    if ferramentas:
        payload["tools"] = ferramentas

    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resposta:
        return json.load(resposta)["message"]

def main():
    inicio = perf_counter()
    eventos = []

    ferramenta = [{
        "type": "function",
        "function": {
            "name": "consultar_cliente",
            "description": "Localiza um cliente pelo nome antes de consultar pedidos.",
            "parameters": {
                "type": "object",
                "properties": {"nome": {"type": "string"}},
                "required": ["nome"]
            }
        }
    }]

    mensagens = [
        {
            "role": "system",
            "content": (
                "Voce consulta pedidos em banco de dados. "
                "Para responder sobre Maria, solicite consultar_cliente "
                "com nome Maria. Nunca invente registros."
            )
        },
        {
            "role": "user",
            "content": "Qual o total em reais e quantos pedidos Maria realizou?"
        }
    ]

    decisao = chamar_llm(mensagens, ferramenta)
    chamadas = decisao.get("tool_calls") or []

    chamada_valida = False
    for chamada in chamadas:
        funcao = chamada.get("function", {})
        argumentos = funcao.get("arguments", {})
        if isinstance(argumentos, str):
            try:
                argumentos = json.loads(argumentos)
            except json.JSONDecodeError:
                argumentos = {}

        if (
            funcao.get("name") == "consultar_cliente"
            and isinstance(argumentos, dict)
            and argumentos.get("nome", "").strip().lower() == "maria"
        ):
            chamada_valida = True
            break

    eventos.append({
        "etapa": "decisao_llm",
        "ferramenta_correta": chamada_valida
    })

    if not chamada_valida:
        raise RuntimeError(
            "Agente reprovado: nao solicitou consultar_cliente(Maria)."
        )

    cliente = consultar_cliente("Maria")
    if cliente is None:
        raise RuntimeError("Cliente nao encontrado.")

    eventos.append({
        "etapa": "consultar_cliente",
        "cliente_id": cliente[0],
        "nome": cliente[1]
    })

    pedidos = consultar_pedidos(cliente[0])
    eventos.append({
        "etapa": "consultar_pedidos",
        "pedidos": [
            {"id": identificador, "valor_centavos": valor}
            for identificador, valor in pedidos
        ]
    })

    resultado = calcular_total(pedidos)
    eventos.append({
        "etapa": "calcular_total",
        "resultado": resultado
    })

    resposta_modelo = chamar_llm([
        {
            "role": "system",
            "content": (
                "Voce e um assistente de pedidos. "
                "Responda somente com os dados fornecidos. "
                "Nao invente valores. Seja conciso."
            )
        },
        {
            "role": "user",
            "content": (
                "Informe quantos pedidos Maria realizou e o total em reais. "
                f"Dados verificados: {json.dumps(resultado)}"
            )
        }
    ]).get("content", "").strip()

    quantidade = resultado["quantidade"]
    total = resultado["total"]

    # Resposta estruturada e garantida pelos calculos Python.
    resposta_validada = (
        f"Maria realizou {quantidade} pedidos, "
        f"totalizando R$ {total:.2f}."
    )

    with sqlite3.connect(BANCO) as con:
        verificacao = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(valor_centavos),0) "
            "FROM pedidos WHERE cliente_id=?",
            (cliente[0],)
        ).fetchone()

    aprovado = (
        quantidade == verificacao[0]
        and round(total * 100) == verificacao[1]
        and chamada_valida
    )

    relatorio = {
        "modelo": MODELO,
        "aprovado": aprovado,
        "decisao_llm_valida": chamada_valida,
        "eventos": eventos,
        "resposta_llm_nao_confiavel": resposta_modelo,
        "resposta_validada": resposta_validada,
        "resultado": resultado,
        "tempo_segundos": round(perf_counter() - inicio, 3)
    }

    SAIDA.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("=== AULA 16 - VALIDACAO DO AGENTE ===")
    print("Decisao do LLM correta:", chamada_valida)
    print("Consulta SQL realizada:", True)
    print("Calculo Python realizado:", True)
    print("Resposta validada:", resposta_validada)
    print("Resultado aprovado:", aprovado)
    print("Tempo:", relatorio["tempo_segundos"], "segundos")
    print("Relatorio:", SAIDA)

    if not aprovado:
        raise RuntimeError("Resultado nao aprovado.")

if __name__ == "__main__":
    main()