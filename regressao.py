
import pandas as pd
import numpy as np

def calcular_regressao(arquivo):
    # Leitura do arquivo CSV
    df = pd.read_csv(arquivo)

    # Variáveis preditoras
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()

    # Variável dependente
    preco = df["preco_venda"].to_list()

    # Construção da matriz X
    X = np.column_stack((
        np.ones(len(area)),
        area,
        quartos
    ))

    # Construção do vetor Y
    Y = np.array(preco)

    # Cálculo dos coeficientes pelo método dos mínimos quadrados
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y

    # Retorno dos três coeficientes
    return beta.tolist()
