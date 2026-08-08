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

def avaliar_cobertura_semantica(texto_original, resumo_gerado):
    prompt = f"""
Analise o resumo gerado com base no trecho do artigo original fornecido.
Verifique a presença de cinco elementos estruturais no resumo: problema,
objetivo, método, resultados e contribuição.

Seja rigoroso: só marque presente=1 se o elemento estiver explicitamente e
inequivocamente no resumo, com um trecho concreto que comprove isso. Não
marque 1 por inferência, insinuação, ou porque "parece que está implícito".
Um resumo bem escrito ainda pode deixar de contemplar um ou mais desses
elementos com clareza, não hesite em marcar 0 nesses casos.

Para cada elemento, retorne presente (0 ou 1) e evidencia (uma citação curta
e literal do resumo gerado que comprove a marcação; string vazia "" se
presente=0).

Retorne exclusivamente este JSON, sem texto adicional:
{{
  "problema": {{"presente": 0 ou 1, "evidencia": "..."}},
  "objetivo": {{"presente": 0 ou 1, "evidencia": "..."}},
  "metodo": {{"presente": 0 ou 1, "evidencia": "..."}},
  "resultados": {{"presente": 0 ou 1, "evidencia": "..."}},
  "contribuicao": {{"presente": 0 ou 1, "evidencia": "..."}}
}}

Artigo original: {texto_original}

Resumo gerado: {resumo_gerado}
""".strip()

    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content

    try:
        bruto = json.loads(_limpar_json(resposta))
        cobertura = {
            elemento: bruto.get(elemento, {}).get("presente")
            for elemento in ELEMENTOS_COBERTURA
        }
        evidencias = {
            elemento: bruto.get(elemento, {}).get("evidencia", "")
            for elemento in ELEMENTOS_COBERTURA
        }
    except (json.JSONDecodeError, AttributeError):
        cobertura = {elemento: None for elemento in ELEMENTOS_COBERTURA}
        evidencias = {elemento: "" for elemento in ELEMENTOS_COBERTURA}

    return cobertura, evidencias

def _rodar_avaliacao_fidelidade_uma_vez(texto_original, resumo_gerado):
    resposta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _montar_prompt_fidelidade(texto_original, resumo_gerado)}],
        temperature=0,
    ).choices[0].message.content

    try:
        return json.loads(_limpar_json(resposta))
    except json.JSONDecodeError:
        return {"nota_fidelidade": None, "afirmacoes_nao_suportadas": []}


def _montar_prompt_fidelidade(texto_original, resumo_gerado):
    return f"""
Compare o resumo gerado com o artigo original.
Primeiro, identifique qualquer afirmação, dado numérico, método ou conclusão
presente no resumo que não tenha suporte direto no texto original.

Depois, atribua a nota_fidelidade seguindo ESTRITAMENTE esta régua, para
manter a nota consistente com a lista de afirmações não suportadas que você
levantou:
- 5: nenhuma afirmação não suportada
- 4: 1 afirmação não suportada, de importância secundária
- 3: 2 a 3 afirmações não suportadas, ou 1 de importância central
- 2: 4 ou mais afirmações não suportadas, ou distorção de um resultado central
- 1: invenção de dado central (número, conclusão ou método incorretos)

A nota_fidelidade deve ser sempre coerente com o tamanho e a gravidade da
lista de afirmacoes_nao_suportadas -- não atribua uma nota alta se a lista
tiver várias afirmações, nem uma nota baixa se a lista estiver vazia.

Se o resumo for totalmente fiel, retorne uma lista vazia.
Retorne estritamente um JSON neste formato, sem explicações adicionais:
{{"afirmacoes_nao_suportadas": ["...", "..."], "nota_fidelidade": <1 a 5>}}

Artigo original: {texto_original}

resumo gerado: {resumo_gerado}
""".strip()


def avaliar_fidelidade(texto_original, resumo_gerado):
    """
    Roda a auditoria de fidelidade duas vezes de forma independente e consolida
    por consenso, para reduzir o ruído do próprio avaliador (mesmo em
    temperature=0 o LLM-juiz tem variância residual entre chamadas, o que pode
    mascarar diferenças reais e pequenas entre abordagens).

    - nota_fidelidade final: média das duas notas, arredondada para o inteiro
      mais próximo (ties arredondam para cima), mantendo a régua 1-5.
    - afirmacoes_nao_suportadas final: apenas as afirmações que a auditoria
      considerou problemáticas nas DUAS rodadas (interseção aproximada por
      similaridade textual simples), para não penalizar por um julgamento
      espúrio isolado de uma única rodada.
    """
    rodada_1 = _rodar_avaliacao_fidelidade_uma_vez(texto_original, resumo_gerado)
    rodada_2 = _rodar_avaliacao_fidelidade_uma_vez(texto_original, resumo_gerado)

    notas = [r["nota_fidelidade"] for r in (rodada_1, rodada_2) if r.get("nota_fidelidade") is not None]
    if notas:
        nota_final = int(round(sum(notas) / len(notas)))
    else:
        nota_final = None

    afirmacoes_1 = rodada_1.get("afirmacoes_nao_suportadas", []) or []
    afirmacoes_2 = rodada_2.get("afirmacoes_nao_suportadas", []) or []

    def _normalizar(texto):
        return " ".join(texto.lower().split())

    normalizadas_2 = [_normalizar(a) for a in afirmacoes_2]
    consenso = []
    for afirmacao in afirmacoes_1:
        alvo = _normalizar(afirmacao)
        # match aproximado: considera consenso se uma afirmação da rodada 1
        # compartilha uma fração substancial de palavras com alguma da rodada 2
        for outra in normalizadas_2:
            palavras_alvo = set(alvo.split())
            palavras_outra = set(outra.split())
            if not palavras_alvo or not palavras_outra:
                continue
            sobreposicao = len(palavras_alvo & palavras_outra) / min(len(palavras_alvo), len(palavras_outra))
            if sobreposicao >= 0.5:
                consenso.append(afirmacao)
                break

    return {
        "nota_fidelidade": nota_final,
        "afirmacoes_nao_suportadas": consenso,
        "nota_fidelidade_rodada_1": rodada_1.get("nota_fidelidade"),
        "nota_fidelidade_rodada_2": rodada_2.get("nota_fidelidade"),
        "concordancia_notas": (
            rodada_1.get("nota_fidelidade") == rodada_2.get("nota_fidelidade")
            if notas and len(notas) == 2 else None
        ),
    }

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
    cobertura, evidencias_cobertura = avaliar_cobertura_semantica(original, gerado)
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
        "fidelidade_afirmacoes_nao_suportadas": " | ".join(fidelidade["afirmacoes_nao_suportadas"]),
        "fidelidade_nota_rodada_1": fidelidade.get("nota_fidelidade_rodada_1"),
        "fidelidade_nota_rodada_2": fidelidade.get("nota_fidelidade_rodada_2"),
        "fidelidade_concordancia_rodadas": fidelidade.get("concordancia_notas"),
    }
    for elemento in ELEMENTOS_COBERTURA:
        linha[f"cobertura_{elemento}"] = cobertura.get(elemento)
        linha[f"cobertura_{elemento}_evidencia"] = evidencias_cobertura.get(elemento, "")

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