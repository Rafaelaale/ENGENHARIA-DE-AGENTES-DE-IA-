from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODELO = BASE / "modelo_churn.joblib"

if not MODELO.exists():
    raise RuntimeError("Execute treinar.py antes de iniciar a API.")

modelo = joblib.load(MODELO)
app = FastAPI(title="API Churn - Aula 07")

class Cliente(BaseModel):
    CreditScore: int = Field(ge=300, le=900)
    Balance: float = Field(ge=0)

@app.get("/status")
def status():
    return {"status": "online", "modelo": "LogisticRegression"}

@app.post("/predict")
def predict(cliente: Cliente):
    dados = pd.DataFrame([cliente.model_dump()])
    probabilidade = float(modelo.predict_proba(dados)[0][1])
    previsao = int(probabilidade >= 0.5)

    resultado = {
        "previsao_churn": previsao,
        "probabilidade_churn": round(probabilidade, 6),
        "limiar": 0.5
    }

    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entrada": cliente.model_dump(),
        "resultado": resultado
    }

    with (BASE / "previsoes.jsonl").open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return resultado
