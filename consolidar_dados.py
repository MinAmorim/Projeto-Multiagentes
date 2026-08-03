import os
import pandas as pd

CAMINHO_BASE = os.path.dirname(os.path.abspath(__file__))
PASTA_RESULTADOS = os.path.join(CAMINHO_BASE, "data", "resultados")

CAMINHO_CUSTO = os.path.join(PASTA_RESULTADOS, "custo.csv")
CAMINHO_AVALIACAO = os.path.join(PASTA_RESULTADOS, "avaliacao.csv")


def carregar_dados():
    custo = pd.read_csv(CAMINHO_CUSTO)
    avaliacao = pd.read_csv(CAMINHO_AVALIACAO, sep=";", decimal=",")
    return custo, avaliacao


def fundir(custo, avaliacao):
    # Uma linha por (arquivo, abordagem), juntando as métricas de qualidade
    # (avaliacao.csv) com as de custo (custo.csv).
    return pd.merge(avaliacao, custo, on=["arquivo", "abordagem"], how="inner")


def calcular_medias(dados_fundidos):
    colunas = [
        "rouge1", "rouge2", "rougeL", "bertscore_f1",
        "cobertura_score", "fidelidade_nota", "fidelidade_qtd_nao_suportadas",
        "tempo_segundos", "tokens_total", "numero_chamadas",
    ]
    colunas = [c for c in colunas if c in dados_fundidos.columns]
    return dados_fundidos.groupby("abordagem")[colunas].agg(["mean", "std"]).round(3)


def executar():
    custo, avaliacao = carregar_dados()
    dados_fundidos = fundir(custo, avaliacao)

    caminho_fundido = os.path.join(PASTA_RESULTADOS, "dados_fundidos.csv")
    dados_fundidos.to_csv(caminho_fundido, index=False)
    print(f"Dados fundidos salvos em {caminho_fundido} ({len(dados_fundidos)} linhas)")

    medias = calcular_medias(dados_fundidos)
    caminho_medias = os.path.join(PASTA_RESULTADOS, "medias_por_abordagem.csv")
    medias.to_csv(caminho_medias)
    print(f"Médias por abordagem salvas em {caminho_medias}")

    print()
    print(medias.to_string())


if __name__ == "__main__":
    executar()