import json
from pathlib import Path

def salvar_relatorio(metricas, caminho):
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(metricas, indent=4), encoding='utf-8')
    return destino

def verificar_qualidade(metricas, minimo_f1=0.60):
    return 'APROVADO' if metricas['f1'] >= minimo_f1 else 'REVISAR'
