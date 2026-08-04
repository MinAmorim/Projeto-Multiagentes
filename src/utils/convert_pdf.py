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


def extrair_tabelas_pagina(pagina):
    blocos = []
    for tabela in pagina.extract_tables():
        linhas_formatadas = []
        for linha in tabela:
            celulas = [(c or "").strip().replace("\n", " ") for c in linha]
            linhas_formatadas.append(" | ".join(celulas))
        blocos.append("\n".join(linhas_formatadas))
    return blocos

def extrair_texto_pagina(pagina):
    if eh_duas_colunas(pagina):
        largura = pagina.width
        meio = largura / 2
        coluna_esquerda = pagina.crop((0, 0, meio, pagina.height))
        coluna_direita = pagina.crop((meio, 0, largura, pagina.height))
        texto = (coluna_esquerda.extract_text() or "") + "\n" + (coluna_direita.extract_text() or "")
    else:
        texto = pagina.extract_text() or ""

    tabelas = extrair_tabelas_pagina(pagina)
    if tabelas:
        texto += "\n\n[TABELAS DA PÁGINA]\n" + "\n\n".join(tabelas)

    return texto

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