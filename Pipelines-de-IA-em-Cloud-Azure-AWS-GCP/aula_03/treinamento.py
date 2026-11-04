from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ATRIBUTOS = ['meses_cliente', 'mensalidade', 'chamados_suporte', 'atrasos_pagamento']

def treinar_modelo(X_treino, y_treino):
    preprocessamento = ColumnTransformer([
        ('numericos', Pipeline([
            ('imputador', SimpleImputer(strategy='median')),
            ('normalizador', StandardScaler())
        ]), ATRIBUTOS)
    ])
    modelo = Pipeline([
        ('preprocessamento', preprocessamento),
        ('classificador', LogisticRegression(max_iter=1000, random_state=42))
    ])
    modelo.fit(X_treino, y_treino)
    return modelo
