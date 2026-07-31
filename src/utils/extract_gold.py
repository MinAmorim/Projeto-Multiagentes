import os
import re

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed")
PASTA_GOLD = os.path.join(CAMINHO_BASE, "data", "gold")
PASTA_SEM_ABSTRACT = os.path.join(CAMINHO_BASE, "data", "processed_sem_abstract")

os.makedirs(PASTA_GOLD, exist_ok=True)
os.makedirs(PASTA_SEM_ABSTRACT, exist_ok=True)


def extrair_resumo_e_posicao(texto, termo_inicio_regex, termo_parada_extra):
    inicio = re.search(termo_inicio_regex, texto, re.IGNORECASE)

    if not inicio:
        return None, None, None

    restante = texto[inicio.end():]

    fim_secao = re.search(
        r"\n\s*(?:1\s*[\.\)]?|I\s*[\.\)]?)\s+",
        restante,
        re.IGNORECASE
    )
    fim_outro_idioma = re.search(termo_parada_extra, restante, re.IGNORECASE) if termo_parada_extra else None

    candidatos = [m.start() for m in (fim_secao, fim_outro_idioma) if m]

    if candidatos:
        posicao_final = inicio.end() + min(candidatos)
        resumo_bruto = restante[:min(candidatos)]
    else:
        posicao_final = len(texto)
        resumo_bruto = restante

    resumo = re.sub(
        r"(?im)^.*(?:Palavras[- ]?Chave|Keywords?|Key-words?).*$",
        "",
        resumo_bruto
    )
    resumo = re.sub(r"(?im)^Abstract\s*$", "", resumo)
    resumo = re.sub(r"\n{2,}", "\n\n", resumo).strip()

    return resumo, inicio.start(), posicao_final


def executar():
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]

    for arquivo in arquivos:
        caminho = os.path.join(PASTA_PROCESSADOS, arquivo)

        with open(caminho, "r", encoding="utf-8") as f:
            texto = f.read()

        resumo_pt, ini_pt, fim_pt = extrair_resumo_e_posicao(texto, r"\bResumo\b", r"\bAbstract\b")
        resumo_en, ini_en, fim_en = extrair_resumo_e_posicao(texto, r"\bAbstract\b", r"\bResumo\b")

        if resumo_pt:
            gold, idioma = resumo_pt, "pt"
        elif resumo_en:
            gold, idioma = resumo_en, "en"
        else:
            gold, idioma = None, None

        spans = sorted(
            (
                (ini, fim) for ini, fim in [(ini_pt, fim_pt), (ini_en, fim_en)]
                if ini is not None
            ),
            key=lambda par: par[0]
        )

        mesclados = []
        for ini, fim in spans:
            if mesclados and ini <= mesclados[-1][1]:
                mesclados[-1] = (mesclados[-1][0], max(mesclados[-1][1], fim))
            else:
                mesclados.append((ini, fim))

        texto_limpo = ""
        cursor = 0
        for ini, fim in mesclados:
            texto_limpo += texto[cursor:ini]
            cursor = fim
        texto_limpo += texto[cursor:]

        caminho_limpo = os.path.join(PASTA_SEM_ABSTRACT, arquivo)
        with open(caminho_limpo, "w", encoding="utf-8") as f:
            f.write(texto_limpo)

        if gold is None:
            print(f"Ignorado (sem resumo/abstract identificável): {arquivo}")
            continue

        with open(os.path.join(PASTA_GOLD, arquivo), "w", encoding="utf-8") as f:
            f.write(gold)

        print(f"Resumo salvo ({idioma}): {arquivo}")


if __name__ == "__main__":
    executar()