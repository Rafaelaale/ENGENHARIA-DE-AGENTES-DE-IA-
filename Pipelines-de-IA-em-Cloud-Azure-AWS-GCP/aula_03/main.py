from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from dados import gerar_dados
from treinamento import ATRIBUTOS, treinar_modelo
from avaliacao import avaliar_modelo
from monitoramento import salvar_relatorio, verificar_qualidade

BASE = Path(__file__).resolve().parent

def executar_pipeline():
    print('=== PIPELINE DE CHURN ===')

    dados = gerar_dados()
    dados.to_csv(BASE / 'clientes_sinteticos.csv', index=False)
    print(f'Dados preparados: {len(dados)} clientes')

    X = dados[ATRIBUTOS]
    y = dados['churn']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    modelo = treinar_modelo(X_treino, y_treino)
    print('Modelo treinado')

    metricas = avaliar_modelo(modelo, X_teste, y_teste)
    print('Metricas:', metricas)

    status = verificar_qualidade(metricas)
    print('Avaliacao de qualidade:', status)

    if status != 'APROVADO':
        salvar_relatorio(metricas, BASE / 'relatorios' / 'avaliacao.json')
        print('Implantacao interrompida: revisar modelo')
        return

    (BASE / 'modelos').mkdir(exist_ok=True)
    joblib.dump(modelo, BASE / 'modelos' / 'modelo_churn.joblib')
    print('Modelo salvo')

    cliente = pd.DataFrame([{
        'meses_cliente': 3,
        'mensalidade': 250,
        'chamados_suporte': 5,
        'atrasos_pagamento': 3
    }])

    probabilidade = float(modelo.predict_proba(cliente)[0][1])
    print(f'Probabilidade estimada de churn: {probabilidade:.2%}')

    metricas['status'] = status
    metricas['probabilidade_cliente_exemplo'] = probabilidade
    salvar_relatorio(metricas, BASE / 'relatorios' / 'avaliacao.json')
    print('Relatorio salvo. Pipeline concluido.')

if __name__ == '__main__':
    executar_pipeline()
