import os
import csv
from collections import defaultdict

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAMINHO_CSV = os.path.join(CAMINHO_BASE, "data", "resultados", "avaliacao.csv")

ELEMENTOS_COBERTURA = ["problema", "objetivo", "metodo", "resultados", "contribuicao"]


def _para_bool(valor):
    return str(valor).strip().lower() == "true"


def analisar(caminho_csv=CAMINHO_CSV):
    with open(caminho_csv, "r", encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f, delimiter=";")
        linhas = list(leitor)

    if not linhas:
        print("CSV vazio ou não encontrado.")
        return

    abordagens = sorted(set(linha["abordagem"] for linha in linhas))

    print(f"Total de linhas: {len(linhas)}\n")

    # --- Taxa geral (qualquer elemento suspeito na linha) por abordagem ---
    print("=== Taxa de linhas com ao menos 1 elemento de evidência suspeita ===")
    for abordagem in abordagens:
        linhas_abordagem = [l for l in linhas if l["abordagem"] == abordagem]
        suspeitas = sum(1 for l in linhas_abordagem if _para_bool(l.get("cobertura_evidencia_suspeita")))
        total = len(linhas_abordagem)
        pct = 100 * suspeitas / total if total else 0
        print(f"  {abordagem:20s}: {suspeitas:3d} / {total:3d} linhas  ({pct:5.1f}%)")

    # --- Taxa por elemento individual, por abordagem ---
    print("\n=== Taxa de evidência suspeita por elemento, por abordagem ===")
    for abordagem in abordagens:
        linhas_abordagem = [l for l in linhas if l["abordagem"] == abordagem]
        total = len(linhas_abordagem)
        print(f"\n  {abordagem}:")
        for elemento in ELEMENTOS_COBERTURA:
            col = f"cobertura_{elemento}_evidencia_suspeita"
            suspeitas = sum(1 for l in linhas_abordagem if _para_bool(l.get(col)))
            pct = 100 * suspeitas / total if total else 0
            print(f"    {elemento:15s}: {suspeitas:3d} / {total:3d}  ({pct:5.1f}%)")

    # --- Taxa geral (todas as linhas, todas as abordagens) ---
    total_geral = len(linhas)
    suspeitas_geral = sum(1 for l in linhas if _para_bool(l.get("cobertura_evidencia_suspeita")))
    pct_geral = 100 * suspeitas_geral / total_geral if total_geral else 0
    print(f"\n=== Taxa geral (todas as abordagens juntas) ===")
    print(f"  {suspeitas_geral} / {total_geral} linhas  ({pct_geral:.1f}%)")

    # --- Falhas de parsing, por garantia ---
    falhas_parsing = sum(1 for l in linhas if _para_bool(l.get("cobertura_falha_parsing")))
    print(f"\n=== Falhas de parsing de cobertura ===")
    print(f"  {falhas_parsing} / {total_geral} linhas")


if __name__ == "__main__":
    analisar()