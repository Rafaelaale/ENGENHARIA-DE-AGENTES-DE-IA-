from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.diagnostic import het_breuschpagan

raiz = Path(__file__).resolve().parent.parent

treino = raiz / "dados" / "aula_02_treino.csv"
resultados = raiz / "resultados"

if not treino.is_file():
    raise FileNotFoundError(
        f"Dados de treinamento nao encontrados: {treino}"
    )

# ==========================================
# 1. RECUPERAR DADOS DE TREINAMENTO
# ==========================================

dados = pd.read_csv(treino)

y_train = dados["preco_de_venda"]

X_train = sm.add_constant(
    dados.drop(columns=["preco_de_venda"]),
    has_constant="add"
)

# ==========================================
# 2. RECRIAR MODELO 3
# ==========================================

explicativas_3 = [
    "const",
    "area_primeiro_andar",
    "existe_segundo_andar",
    "quantidade_banheiros",
    "qualidade_da_cozinha_Excelente"
]

modelo_3 = sm.OLS(
    y_train,
    X_train[explicativas_3]
).fit()

print("\n===== MODELO 3 =====")
print("Imoveis:", len(dados))
print(f"R2 treinamento: {modelo_3.rsquared:.6f}")

assert np.isclose(
    modelo_3.rsquared,
    0.6511289354898803,
    atol=0.0001
)

# ==========================================
# 3. CALCULAR PREVISOES
# ==========================================

y_previsto_train = modelo_3.predict(
    X_train[explicativas_3]
)

# ==========================================
# 4. OBTER RESIDUOS
# ==========================================

residuos = modelo_3.resid

print("\n===== PRIMEIROS RESIDUOS =====")
print(residuos.head(10).to_string())

print("\n===== ESTATISTICAS DOS RESIDUOS =====")
print(f"Media: R$ {residuos.mean():,.2f}")
print(f"Desvio padrao: R$ {residuos.std():,.2f}")
print(f"Menor residuo: R$ {residuos.min():,.2f}")
print(f"Maior residuo: R$ {residuos.max():,.2f}")

# ==========================================
# 5. CRIAR GRAFICO COM SEABORN
# ==========================================

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 7))

sns.scatterplot(
    x=y_previsto_train.to_numpy(),
    y=residuos.to_numpy(),
    s=55,
    alpha=0.6,
    ax=ax
)

ax.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Residuo zero"
)

ax.set_title(
    "Residuos X Previsao - Modelo 3",
    fontsize=18
)

ax.set_xlabel(
    "Preco da casa - Previsao (R$)",
    fontsize=13
)

ax.set_ylabel(
    "Residuos (R$)",
    fontsize=13
)

ax.ticklabel_format(
    style="plain",
    axis="both",
    useOffset=False
)

ax.legend()

fig.tight_layout()

grafico = (
    resultados / "aula_03_residuos_vs_previsao.png"
)

fig.savefig(
    grafico,
    dpi=160
)

plt.close(fig)

# ==========================================
# 6. TESTE BREUSCH-PAGAN
# ==========================================

bp = het_breuschpagan(
    residuos,
    modelo_3.model.exog
)

estatistica_lm, p_valor_lm, estatistica_f, p_valor_f = bp

print("\n===== TESTE BREUSCH-PAGAN =====")

print(f"Estatistica LM: {estatistica_lm:.6f}")
print(f"P-valor LM: {p_valor_lm:.10g}")
print(f"Estatistica F: {estatistica_f:.6f}")
print(f"P-valor F: {p_valor_f:.10g}")

alpha = 0.05

if p_valor_lm < alpha:
    conclusao = (
        "Ha evidencia estatistica de "
        "heterocedasticidade ao nivel de 5%."
    )
else:
    conclusao = (
        "Nao houve evidencia estatistica suficiente "
        "para rejeitar homocedasticidade ao nivel de 5%."
    )

print("\n===== INTERPRETACAO =====")
print(conclusao)

# ==========================================
# 7. ANALISAR DISPERSAO POR FAIXA
# ==========================================

tabela = pd.DataFrame({
    "preco_real": y_train.to_numpy(),
    "preco_previsto": y_previsto_train.to_numpy(),
    "residuo": residuos.to_numpy()
})

tabela["erro_absoluto"] = tabela["residuo"].abs()

tabela["faixa_previsao"] = pd.qcut(
    tabela["preco_previsto"],
    q=4,
    labels=[
        "Faixa 1 - Menores",
        "Faixa 2",
        "Faixa 3",
        "Faixa 4 - Maiores"
    ],
    duplicates="drop"
)

resumo_faixas = tabela.groupby(
    "faixa_previsao",
    observed=True
).agg(
    quantidade=("residuo", "size"),
    media_erro_absoluto=("erro_absoluto", "mean"),
    desvio_residuos=("residuo", "std")
)

print("\n===== ERROS POR FAIXA DE PREVISAO =====")

print(resumo_faixas.to_string())

# ==========================================
# 8. SALVAR RESULTADOS
# ==========================================

arquivo_csv = (
    resultados / "aula_03_residuos_homocedasticidade.csv"
)

arquivo_txt = (
    resultados / "aula_03_diagnostico_homocedasticidade.txt"
)

for arquivo in (arquivo_csv, arquivo_txt):
    if arquivo.exists():
        raise FileExistsError(arquivo)

tabela.to_csv(
    arquivo_csv,
    index=False
)

relatorio = [
    "ALURA - AULA 03 - HOMOCEDASTICIDADE",
    "",
    f"Imoveis analisados: {len(dados)}",
    f"R2 treinamento: {modelo_3.rsquared:.6f}",
    "",
    "ESTATISTICAS DOS RESIDUOS",
    f"Media: {residuos.mean():.6f}",
    f"Desvio padrao: {residuos.std():.6f}",
    "",
    "TESTE BREUSCH-PAGAN",
    f"LM: {estatistica_lm:.6f}",
    f"P-valor LM: {p_valor_lm:.12g}",
    f"F: {estatistica_f:.6f}",
    f"P-valor F: {p_valor_f:.12g}",
    "",
    "CONCLUSAO",
    conclusao,
    "",
    "ERROS POR FAIXA DE PREVISAO",
    resumo_faixas.to_string(),
    "",
    "OBSERVACOES",
    "Homocedasticidade significa variancia",
    "condicional constante dos erros.",
    "Residuos sem padrao visual nao garantem",
    "todos os pressupostos da regressao.",
    "O teste nao demonstra causalidade."
]

arquivo_txt.write_text(
    "\n".join(relatorio),
    encoding="utf-8"
)

# ==========================================
# 9. VALIDAR ARQUIVOS
# ==========================================

print("\n===== ARQUIVOS GERADOS =====")

for arquivo in (grafico, arquivo_csv, arquivo_txt):

    assert arquivo.is_file()
    assert arquivo.stat().st_size > 0

    print(
        f"{arquivo.name}: "
        f"{arquivo.stat().st_size} bytes"
    )

print("\nAULA 03 - HOMOCEDASTICIDADE EXECUTADA COM SUCESSO")