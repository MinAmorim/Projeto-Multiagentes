import os
import json
import csv
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CAMINHO_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed")
PASTA_GOLD = os.path.join(CAMINHO_BASE, "data", "gold")
PASTA_OUTPUTS = os.path.join(CAMINHO_BASE, "data", "outputs")
PASTA_RESULTADOS = os.path.join(CAMINHO_BASE, "data", "resultados")

ABORDAGENS = ["single_agent", "reflection_agent", "multi_agent"]

ELEMENTOS_COBERTURA = ["problema", "objetivo", "metodo", "resultados", "contribuicao"]

scorer_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def _limpar_json(resposta_llm):
    texto = resposta_llm.strip()
    if texto.startswith("```"):
        texto = texto.strip("`")
        if texto.lower().startswith("json"):
            texto = texto[4:]
    return texto.strip()

def calcular_rouge(gold, gerado):
    scores = scorer_rouge.score(gold, gerado)
    return {chave: valor.fmeasure for chave, valor in scores.items()}

def calcular_bertscore(gold, gerado):
    precisao, recall, f1 = bert_score([gerado], [gold], lang="pt", verbose=False)
    return {
        "precisao": precisao.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }

def avaliar_cobertura_semantica(texto_original, abstract_gerado):
    prompt = f"""
Analise o resumo gerado com base no trecho do artigo original fornecido.
Verifique objetivamente a presença de cinco elementos estruturais no resumo.
Para cada elemento, atribua 1 se estiver claramente identificável, ou 0 caso contrário.
Forneça o resultado exclusivamente no formato JSON abaixo, sem texto adicional:
{{"problema": 0 ou 1, "objetivo": 0 ou 1, "metodo": 0 ou 1, "resultados": 0 ou 1, "contribuicao": 0 ou 1}}

Artigo original: {texto_original[:6000]}

Abstract gerado: {abstract_gerado}
""".strip()

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content

    try:
        cobertura = json.loads(_limpar_json(resposta))
    except json.JSONDecodeError:
        cobertura = {elemento: None for elemento in ELEMENTOS_COBERTURA}
    return cobertura

def avaliar_fidelidade(texto_original, abstract_gerado):
    prompt = f"""
Compare o resumo gerado com o artigo original.
Identifique qualquer afirmação, dado numérico, método ou conclusão presente no abstract que não tenha suporte direto no texto original.
Avalie a fidelidade geral com uma nota de 1 (invenção ou distorção de dados cruciais) a 5 (totalmente fiel e ancorado no artigo).
Liste as afirmações não suportadas, se existirem. Se o resumo for totalmente fiel, retorne uma lista vazia.
Retorne estritamente um JSON neste formato, sem explicações adicionais:
{{"nota_fidelidade": <1 a 5>, "afirmacoes_nao_suportadas": ["...", "..."]}}

Artigo original: {texto_original[:6000]}

Abstract gerado: {abstract_gerado}
""".strip()

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content

    try:
        fidelidade = json.loads(_limpar_json(resposta))
    except json.JSONDecodeError:
        fidelidade = {"nota_fidelidade": None, "afirmacoes_nao_suportadas": []}
    return fidelidade

def avaliar_arquivo(nome_arquivo, abordagem):
    caminho_gold = os.path.join(PASTA_GOLD, nome_arquivo)
    caminho_gerado = os.path.join(PASTA_OUTPUTS, abordagem, nome_arquivo)
    caminho_original = os.path.join(PASTA_PROCESSADOS, nome_arquivo)

    if not (os.path.exists(caminho_gold) and os.path.exists(caminho_gerado)):
        return None

    with open(caminho_gold, "r", encoding="utf-8") as f:
        gold = f.read()
    with open(caminho_gerado, "r", encoding="utf-8") as f:
        gerado = f.read()
    with open(caminho_original, "r", encoding="utf-8") as f:
        original = f.read()

    rouge = calcular_rouge(gold, gerado)
    bert = calcular_bertscore(gold, gerado)
    cobertura = avaliar_cobertura_semantica(original, gerado)
    fidelidade = avaliar_fidelidade(original, gerado)

    linha = {
        "arquivo": nome_arquivo,
        "abordagem": abordagem,
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bertscore_precisao": round(bert["precisao"], 4),
        "bertscore_recall": round(bert["recall"], 4),
        "bertscore_f1": round(bert["f1"], 4),
        "cobertura_score": sum(v for v in cobertura.values() if v is not None),
        "fidelidade_nota": fidelidade["nota_fidelidade"],
        "fidelidade_qtd_nao_suportadas": len(fidelidade["afirmacoes_nao_suportadas"]),
    }
    for elemento in ELEMENTOS_COBERTURA:
        linha[f"cobertura_{elemento}"] = cobertura.get(elemento)

    return linha

def executar():
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    if not os.path.isdir(PASTA_GOLD):
        print(f"Pasta de gold standard não encontrada: {PASTA_GOLD}")
        return

    arquivos_gold = [f for f in os.listdir(PASTA_GOLD) if f.endswith(".txt")]
    if not arquivos_gold:
        print("Nenhum arquivo txt encontrado ")
        return

    resultados = []
    for abordagem in ABORDAGENS:
        pasta_abordagem = os.path.join(PASTA_OUTPUTS, abordagem)
        if not os.path.isdir(pasta_abordagem):
            print(f"Aviso: pasta de outputs '{abordagem}' não encontrada, pulando essa abordagem.")
            continue

        for nome in arquivos_gold:
            print(f"Avaliando {nome} ({abordagem})...")
            resultado = avaliar_arquivo(nome, abordagem)
            if resultado:
                resultados.append(resultado)
            else:
                print(f" resumo gerado ou gold ausente para {nome}")

    if not resultados:
        print("\nNenhum resultado calculado")
        return

    resultados = sorted(resultados, key=lambda x: (x["arquivo"], x["abordagem"]))

    caminho_csv = os.path.join(PASTA_RESULTADOS, "avaliacao.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.DictWriter(f, fieldnames=resultados[0].keys(), delimiter=";")
        escritor.writeheader()
        
        for linha in resultados:
            for k, v in linha.items():
                if isinstance(v, float):
                    linha[k] = str(v).replace(".", ",")
                    
        escritor.writerows(resultados)

    print(f"\nResultados salvos em {caminho_csv}")

if __name__ == "__main__":
    executar()