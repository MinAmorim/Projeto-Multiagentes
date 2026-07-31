import os
import sys
import subprocess

CAMINHO_BASE = os.path.dirname(os.path.abspath(__file__))

ETAPAS = [
    os.path.join(CAMINHO_BASE, "src", "utils", "convert_pdf.py"),
    os.path.join(CAMINHO_BASE, "src", "utils", "extract_gold.py"),
    os.path.join(CAMINHO_BASE, "src", "single_agent.py"),
    os.path.join(CAMINHO_BASE, "src", "reflection_agent.py"),
    os.path.join(CAMINHO_BASE, "src", "multi_agents.py"),
    os.path.join(CAMINHO_BASE, "src", "evaluate.py"),
]


def executar():
    for script in ETAPAS:
        nome_etapa = os.path.relpath(script, CAMINHO_BASE)
        print(f"\n{'=' * 60}")
        print(f"Executando: {nome_etapa}")
        print('=' * 60)

        resultado = subprocess.run([sys.executable, script])

        if resultado.returncode != 0:
            print(f"\nErro ao executar {nome_etapa} (código {resultado.returncode}).")
            print("Pipeline interrompido.")
            sys.exit(resultado.returncode)

    print(f"\n{'=' * 60}")
    print("Pipeline completo executado com sucesso.")
    print(f"Resultados em: {os.path.join(CAMINHO_BASE, 'data', 'resultados')}")
    print('=' * 60)


if __name__ == "__main__":
    executar()