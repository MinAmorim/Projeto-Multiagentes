import pdfplumber
import os

caminho_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pasta_entrada = os.path.join(caminho_base, "data", "raw")
pasta_saida = os.path.join(caminho_base, "data", "processed")

def converter_pdf ():


    for arquivo in os.listdir(pasta_entrada):
        if arquivo.endswith(".pdf"):
            caminho_pdf = os.path.join(pasta_entrada, arquivo)
            nome_txt = arquivo.replace(".pdf", ".txt")
            caminho_txt = os.path.join(pasta_saida, nome_txt)

            with pdfplumber.open(caminho_pdf) as pdf:
                texto = ""
                for pagina in pdf.pages:
                    texto += pagina.extract_text() + "\n"
                with open(caminho_txt, "w", encoding="utf-8") as f:
                    f.write(texto)
                print(f"convertido{arquivo}")


if __name__ == "__main__":
    converter_pdf()
