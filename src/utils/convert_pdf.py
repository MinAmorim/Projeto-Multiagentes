import os
import re
import pdfplumber

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PASTA_ENTRADA = os.path.join(CAMINHO_BASE, "data", "raw")
PASTA_SAIDA = os.path.join(CAMINHO_BASE, "data", "processed")


def limpar_texto(texto):
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r" +\n", "\n", texto)
    return texto.strip()


def converter_pdf():

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    for arquivo in os.listdir(PASTA_ENTRADA):

        if not arquivo.endswith(".pdf"):
            continue

        caminho_pdf = os.path.join(PASTA_ENTRADA, arquivo)
        caminho_txt = os.path.join(
            PASTA_SAIDA,
            arquivo.replace(".pdf", ".txt")
        )

        texto = ""

        with pdfplumber.open(caminho_pdf) as pdf:

            for pagina in pdf.pages:

                texto_pagina = pagina.extract_text()

                if texto_pagina:
                    texto += texto_pagina + "\n\n"

        texto = limpar_texto(texto)

        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(texto)

        print(f"Convertido: {arquivo}")


if __name__ == "__main__":
    converter_pdf()