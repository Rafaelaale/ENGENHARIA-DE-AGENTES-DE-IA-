import numpy as np
import pandas as pd

def gerar_dados(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    meses = rng.integers(1, 73, n)
    mensalidade = rng.uniform(30, 300, n)
    chamados = rng.poisson(2, n)
    atraso = rng.integers(0, 4, n)
    logit = -2.5 - 0.025 * meses + 0.01 * mensalidade + 0.4 * chamados + 0.8 * atraso
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({
        'meses_cliente': meses,
        'mensalidade': mensalidade,
        'chamados_suporte': chamados,
        'atrasos_pagamento': atraso,
        'churn': churn
    })
