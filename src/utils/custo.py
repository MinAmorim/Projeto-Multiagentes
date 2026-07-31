import os
import csv


def registrar_custo(caminho_csv, linha):
    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    existe = os.path.exists(caminho_csv)

    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linha.keys()))
        if not existe:
            escritor.writeheader()
        escritor.writerow(linha)