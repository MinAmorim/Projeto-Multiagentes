import os
import re

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed")
PASTA_GOLD = os.path.join(CAMINHO_BASE, "data", "gold")

os.makedirs(PASTA_GOLD, exist_ok=True)

for arquivo in os.listdir(PASTA_PROCESSADOS):

    if not arquivo.endswith(".txt"):
        continue

    caminho = os.path.join(PASTA_PROCESSADOS, arquivo)

    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()

    resumo = None

    match = re.search(
        r"Abstract\.?\s*(.*?)(?=\n\s*Resumo|\n\s*Keywords|\n\s*Index Terms|\n\s*1\.?\s*Introduction|\n\s*Introduction)",
        texto,
        flags=re.IGNORECASE | re.DOTALL
    )

    if match:
        resumo = match.group(1).strip()

    if resumo is None:
        match = re.search(
            r"Resumo\.?\s*(.*?)(?=\n\s*1\.?\s*Introdução|\n\s*Introdução)",
            texto,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            resumo = match.group(1).strip()

    if resumo:
        caminho_saida = os.path.join(PASTA_GOLD, arquivo)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(resumo)

        print(f"{arquivo}")

    else:
        print(f"{arquivo} não possui resumo.")