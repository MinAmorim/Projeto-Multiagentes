import os
import re

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed")
PASTA_GOLD = os.path.join(CAMINHO_BASE, "data", "gold")

os.makedirs(PASTA_GOLD, exist_ok=True)


def extrair_resumo(texto):
    inicio = re.search(r"\bResumo\b|\bAbstract\b", texto, re.IGNORECASE)

    if not inicio:
        return None

    texto = texto[inicio.end():]

    fim = re.search(
        r"\n\s*(?:1\s*[\.\)]?|I\s*[\.\)]?)\s+",
        texto,
        re.IGNORECASE
    )

    if fim:
        resumo = texto[:fim.start()]
    else:
        resumo = texto

    resumo = re.sub(
        r"(?im)^.*(?:Palavras[- ]?Chave|Keywords?|Key-words?).*$",
        "",
        resumo
    )

    resumo = re.sub(
        r"(?im)^Abstract\s*$",
        "",
        resumo
    )

    resumo = re.sub(r"\n{2,}", "\n\n", resumo)

    return resumo.strip()


for arquivo in os.listdir(PASTA_PROCESSADOS):

    if not arquivo.endswith(".txt"):
        continue

    caminho = os.path.join(PASTA_PROCESSADOS, arquivo)

    with open(caminho, "r", encoding="utf-8") as f:
        texto = f.read()

    resumo = extrair_resumo(texto)

    if resumo is None:
        print(f"Ignorado: {arquivo}")
        continue

    with open(os.path.join(PASTA_GOLD, arquivo), "w", encoding="utf-8") as f:
        f.write(resumo)

    print(f"Resumo salvo: {arquivo}")