import pdfplumber
import os

caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pasta_entrada = os.path.join(caminho_base, "data", "raw")
pasta_saida = os.path.join(caminho_base, "data", "processed")


def eh_duas_colunas(pagina, margem_frac=0.02, limiar=0.01):
    largura = pagina.width
    meio = largura / 2
    margem = largura * margem_frac

    palavras = pagina.extract_words()
    if len(palavras) < 10:
        return False

    no_meio = sum(
        1 for p in palavras
        if abs((p["x0"] + p["x1"]) / 2 - meio) < margem
    )
    proporcao = no_meio / len(palavras)
    return proporcao < limiar


def extrair_texto_pagina(pagina):
    if not eh_duas_colunas(pagina):
        return pagina.extract_text() or ""

    largura = pagina.width
    meio = largura / 2
    coluna_esquerda = pagina.crop((0, 0, meio, pagina.height))
    coluna_direita = pagina.crop((meio, 0, largura, pagina.height))

    texto_esquerda = coluna_esquerda.extract_text() or ""
    texto_direita = coluna_direita.extract_text() or ""

    return texto_esquerda + "\n" + texto_direita


def converter_pdf():
    for arquivo in os.listdir(pasta_entrada):
        if arquivo.endswith(".pdf"):
            caminho_pdf = os.path.join(pasta_entrada, arquivo)
            nome_txt = arquivo.replace(".pdf", ".txt")
            caminho_txt = os.path.join(pasta_saida, nome_txt)

            with pdfplumber.open(caminho_pdf) as pdf:
                texto = ""
                for pagina in pdf.pages:
                    texto += extrair_texto_pagina(pagina) + "\n"
                with open(caminho_txt, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"convertido {arquivo}")


if __name__ == "__main__":
    converter_pdf()