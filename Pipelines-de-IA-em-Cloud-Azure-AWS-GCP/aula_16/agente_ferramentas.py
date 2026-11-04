from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import urllib.request
from time import perf_counter

BASE = Path(__file__).resolve().parent
BANCO = BASE / "clientes.db"
RELATORIO = BASE / "resultado_agente.json"

MODELO = "llama3.2:latest"
URL = "http://127.0.0.1:11434/api/chat"

FERRAMENTAS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_cliente",
            "description": "Busca cliente pelo nome no banco de dados.",
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {"type": "string"}
                },
                "required": ["nome"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_pedidos",
            "description": "Consulta pedidos de um cliente pelo ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cliente_id": {"type": "integer"}
                },
                "required": ["cliente_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_total",
            "description": "Calcula total e quantidade de pedidos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "valores": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                },
                "required": ["valores"]
            }
        }
    }
]


def preparar_banco():
    with sqlite3.connect(BANCO) as conexao:
        conexao.executescript("""
            CREATE TABLE clientes (
                id INTEGER PRIMARY KEY,
                nome TEXT NOT NULL
            );

            CREATE TABLE pedidos (
                id INTEGER PRIMARY KEY,
                cliente_id INTEGER NOT NULL,
                valor_centavos INTEGER NOT NULL,
                FOREIGN KEY(cliente_id) REFERENCES clientes(id)
            );

            INSERT INTO clientes VALUES (1, 'Maria');
            INSERT INTO clientes VALUES (2, 'Joao');

            INSERT INTO pedidos VALUES (1, 1, 12050);
            INSERT INTO pedidos VALUES (2, 1, 7990);
            INSERT INTO pedidos VALUES (3, 1, 20000);
            INSERT INTO pedidos VALUES (4, 2, 5000);
        """)


def consultar_cliente(nome):
    with sqlite3.connect(BANCO) as conexao:
        registro = conexao.execute(
            "SELECT id, nome FROM clientes WHERE LOWER(nome) = LOWER(?)",
            (nome,)
        ).fetchone()

    if registro is None:
        return {"encontrado": False}

    return {
        "encontrado": True,
        "cliente_id": registro[0],
        "nome": registro[1]
    }


def consultar_pedidos(cliente_id):
    with sqlite3.connect(BANCO) as conexao:
        registros = conexao.execute(
            """
            SELECT id, valor_centavos
            FROM pedidos
            WHERE cliente_id = ?
            ORDER BY id
            """,
            (cliente_id,)
        ).fetchall()

    return {
        "cliente_id": cliente_id,
        "pedidos": [
            {
                "pedido_id": pedido_id,
                "valor": centavos / 100
            }
            for pedido_id, centavos in registros
        ]
    }


def calcular_total(valores):
    centavos = [
        round(float(valor) * 100)
        for valor in valores
    ]

    return {
        "quantidade": len(centavos),
        "total": round(sum(centavos) / 100, 2)
    }


def executar_ferramenta(nome, argumentos):
    if not isinstance(argumentos, dict):
        raise ValueError("Argumentos invalidos.")

    if nome == "consultar_cliente":
        if set(argumentos) != {"nome"}:
            raise ValueError("Argumentos invalidos para cliente.")

        if not isinstance(argumentos["nome"], str):
            raise ValueError("Nome invalido.")

        return consultar_cliente(argumentos["nome"])

    if nome == "consultar_pedidos":
        if set(argumentos) != {"cliente_id"}:
            raise ValueError("Argumentos invalidos para pedidos.")

        cliente_id = argumentos["cliente_id"]

        if type(cliente_id) is not int:
            raise ValueError("ID invalido.")

        return consultar_pedidos(cliente_id)

    if nome == "calcular_total":
        if set(argumentos) != {"valores"}:
            raise ValueError("Argumentos invalidos para calculo.")

        valores = argumentos["valores"]

        if (
            not isinstance(valores, list)
            or len(valores) > 100
            or any(
                type(valor) not in (int, float)
                or not (0 <= valor <= 1000000)
                for valor in valores
            )
        ):
            raise ValueError("Valores invalidos.")

        return calcular_total(valores)

    raise ValueError(f"Ferramenta nao autorizada: {nome}")


def chamar_modelo(mensagens):
    payload = {
        "model": MODELO,
        "messages": mensagens,
        "tools": FERRAMENTAS,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 350
        }
    }

    requisicao = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        requisicao,
        timeout=180
    ) as resposta:
        return json.loads(
            resposta.read().decode("utf-8")
        )["message"]


def main():
    preparar_banco()

    pergunta = (
        "Qual e o total em reais dos pedidos da cliente "
        "Maria e quantos pedidos ela realizou? "
        "Consulte o banco de dados e utilize as ferramentas "
        "disponiveis. Nao invente valores."
    )

    mensagens = [
        {
            "role": "system",
            "content": (
                "Voce e um agente de consulta de pedidos. "
                "Use as ferramentas para obter dados reais. "
                "Nao invente clientes, pedidos ou valores. "
                "Utilize calcular_total para somar os pedidos. "
                "Responda em portugues brasileiro."
            )
        },
        {
            "role": "user",
            "content": pergunta
        }
    ]

    historico = []
    resposta_final = ""
    inicio = perf_counter()

    print("=== AULA 16 - AGENTE COM FERRAMENTAS ===")
    print("Pergunta:", pergunta)

    for rodada in range(1, 9):
        mensagem = chamar_modelo(mensagens)

        chamadas = mensagem.get("tool_calls") or []

        mensagens.append(mensagem)

        if not chamadas:
            resposta_final = mensagem.get("content", "").strip()
            print("\nResposta final:", resposta_final)
            break

        for chamada in chamadas:
            funcao = chamada["function"]
            nome = funcao["name"]
            argumentos = funcao.get("arguments", {})

            if isinstance(argumentos, str):
                argumentos = json.loads(argumentos)

            try:
                resultado = executar_ferramenta(
                    nome,
                    argumentos
                )
            except (ValueError, TypeError, KeyError) as erro:
                resultado = {"erro": str(erro)}

            print(f"\nRodada {rodada}")
            print("Ferramenta:", nome)
            print("Argumentos:", argumentos)
            print("Resultado:", resultado)

            historico.append({
                "rodada": rodada,
                "ferramenta": nome,
                "argumentos": argumentos,
                "resultado": resultado
            })

            mensagens.append({
                "role": "tool",
                "tool_name": nome,
                "content": json.dumps(
                    resultado,
                    ensure_ascii=False
                )
            })

    tempo = round(perf_counter() - inicio, 3)

    nomes = [
        item["ferramenta"]
        for item in historico
    ]

    ferramentas_esperadas = [
        "consultar_cliente",
        "consultar_pedidos",
        "calcular_total"
    ]

    ferramentas_utilizadas = all(
        nome in nomes
        for nome in ferramentas_esperadas
    )

    with sqlite3.connect(BANCO) as conexao:
        valores = [
            registro[0]
            for registro in conexao.execute(
                """
                SELECT p.valor_centavos
                FROM pedidos p
                JOIN clientes c ON c.id = p.cliente_id
                WHERE c.nome = 'Maria'
                """
            )
        ]

    esperado = {
        "quantidade": len(valores),
        "total": round(sum(valores) / 100, 2)
    }

    relatorio = {
        "modelo": MODELO,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "pergunta": pergunta,
        "resposta_final": resposta_final,
        "ferramentas_utilizadas": ferramentas_utilizadas,
        "quantidade_chamadas": len(historico),
        "historico": historico,
        "resultado_esperado": esperado,
        "tempo_segundos": tempo
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
    print("Ferramentas esperadas utilizadas:", ferramentas_utilizadas)
    print("Resultado correto no banco:", esperado)
    print("Resposta final gerada:", bool(resposta_final))
    print("Chamadas de ferramentas:", len(historico))
    print("Tempo:", tempo, "segundos")
    print("Relatorio:", RELATORIO)
    print("Nenhum recurso Cloud foi criado.")


if __name__ == "__main__":
    main()