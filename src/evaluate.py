import os
import json
import csv
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score

load_dotenv()
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODELO = "qwen2.5-16k"

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


def _normalizar_texto(texto):
    return " ".join(texto.lower().split())


def _evidencia_esta_no_resumo(evidencia, resumo_gerado):
    """
    Checagem programática simples: confirma se a evidência citada pelo juiz
    realmente aparece (literalmente) no resumo gerado, e não no artigo
    original. Uma evidência vazia (elemento marcado como ausente) não
    precisa ser validada.
    """
    if not evidencia:
        return True
    evidencia_norm = _normalizar_texto(evidencia)
    resumo_norm = _normalizar_texto(resumo_gerado)
    return evidencia_norm in resumo_norm


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
Você vai analisar dois textos: um ARTIGO ORIGINAL (mais longo) e um RESUMO GERADO (mais
curto, entre 150 e 250 palavras). Sua tarefa é avaliar apenas o RESUMO GERADO.

Verifique a presença de cinco elementos estruturais no RESUMO GERADO: problema, objetivo,
método, resultados e contribuição.

Seja rigoroso: só marque presente=1 se o elemento estiver explicitamente e
inequivocamente no RESUMO GERADO (não no artigo original), com um trecho concreto do
RESUMO GERADO que comprove isso. Não marque 1 por inferência, insinuação, ou porque
"parece que está implícito". Um resumo bem escrito ainda pode deixar de contemplar um ou
mais desses elementos com clareza, não hesite em marcar 0 nesses casos.

REGRA CRÍTICA SOBRE A EVIDÊNCIA: a evidência DEVE ser uma frase copiada literalmente do
RESUMO GERADO (o texto fornecido depois de "RESUMO GERADO:" abaixo), nunca do ARTIGO
ORIGINAL. O artigo original é mais longo, costuma ter citações bibliográficas como
"[Autor, ano]" e frases diferentes do resumo. Se você citar uma frase que só existe no
artigo original e não no resumo gerado, isso é um erro grave. Antes de escrever cada
evidência, confirme mentalmente que aquela frase exata aparece no texto do RESUMO GERADO.

Retorne exclusivamente este JSON plano, sem aninhamento e sem texto adicional:
{{
  "problema_presente": 0 ou 1,
  "problema_evidencia": "...",
  "objetivo_presente": 0 ou 1,
  "objetivo_evidencia": "...",
  "metodo_presente": 0 ou 1,
  "metodo_evidencia": "...",
  "resultados_presente": 0 ou 1,
  "resultados_evidencia": "...",
  "contribuicao_presente": 0 ou 1,
  "contribuicao_evidencia": "..."
}}

ARTIGO ORIGINAL: {texto_original}

RESUMO GERADO: {resumo_gerado}
""".strip()

    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        extra_body={"options": {"num_ctx": 16384}},
        response_format={"type": "json_object"},
    ).choices[0].message.content

    try:
        bruto = json.loads(_limpar_json(resposta))
        cobertura = {
            elemento: bruto.get(f"{elemento}_presente")
            for elemento in ELEMENTOS_COBERTURA
        }
        evidencias = {
            elemento: bruto.get(f"{elemento}_evidencia", "")
            for elemento in ELEMENTOS_COBERTURA
        }
    except (json.JSONDecodeError, AttributeError):
        print(f"  [AVISO] Falha ao parsear JSON de cobertura. Resposta bruta: {resposta[:300]}")
        cobertura = {elemento: None for elemento in ELEMENTOS_COBERTURA}
        evidencias = {elemento: "" for elemento in ELEMENTOS_COBERTURA}

    evidencias_suspeitas = {
        elemento: not _evidencia_esta_no_resumo(evidencias.get(elemento, ""), resumo_gerado)
        for elemento in ELEMENTOS_COBERTURA
    }

    return cobertura, evidencias, evidencias_suspeitas


def _rodar_avaliacao_fidelidade_uma_vez(texto_original, resumo_gerado):
    resposta = client.chat.completions.create(
        model=MODELO,
        messages=[{"role": "user", "content": _montar_prompt_fidelidade(texto_original, resumo_gerado)}],
        temperature=0,
        extra_body={"options": {"num_ctx": 16384}},
        response_format={"type": "json_object"},
    ).choices[0].message.content

    try:
        return json.loads(_limpar_json(resposta))
    except json.JSONDecodeError:
        print(f"  [AVISO] Falha ao parsear JSON de fidelidade. Resposta bruta: {resposta[:300]}")
        return {"nota_fidelidade": None, "afirmacoes_nao_suportadas": []}


def _montar_prompt_fidelidade(texto_original, resumo_gerado):
    return f"""
Compare o resumo gerado com o artigo original.
Primeiro, identifique qualquer afirmação, dado numérico, método ou conclusão
presente no resumo que não tenha suporte no texto original.

IMPORTANTE sobre o que conta como "suporte" (leia com atenção antes de avaliar):
- O resumo pode e deve parafrasear o artigo original com outras palavras. NÃO marque
  uma afirmação como não suportada apenas porque a frase não é idêntica ou não usa as
  mesmas palavras do texto original, verifique o SIGNIFICADO da afirmação contra o
  significado do texto original, não a correspondência literal de string.
- O resumo pode combinar, em uma única frase, informações que aparecem em frases ou
  parágrafos diferentes do artigo original (ex.: unir o objetivo do estudo com o
  contexto/problema, ou um resultado com o método que o gerou). Isso é esperado e não
  é, por si só, motivo para marcar como não suportado, desde que cada fato individual
  esteja presente no artigo e a combinação não crie uma relação de causa e efeito,
  generalização ou atribuição de escopo que o artigo não estabelece explicitamente.
- Números, datas, percentuais, nomes de ferramentas/sistemas e conclusões devem
  corresponder ao que está no artigo original (aqui a precisão do dado em si importa),
  mas a frase ao redor deles pode estar totalmente reescrita, resumida ou reorganizada.
- Antes de marcar uma afirmação como não suportada, procure ativamente pelo SIGNIFICADO
  dela em qualquer parte do artigo original (não apenas por correspondência textual
  próxima). Só marque como não suportada se, mesmo assim, você não encontrar essa
  informação, ou encontrar uma informação diferente, mais restrita, ou com escopo
  distinto do que o resumo apresenta.

Depois, atribua a nota_fidelidade seguindo ESTRITAMENTE esta régua, para
manter a nota consistente com a lista de afirmações não suportadas que você
levantou:
- 5: nenhuma afirmação não suportada
- 4: 1 afirmação não suportada, de importância secundária
- 3: 2 a 3 afirmações não suportadas, ou 1 de importância central
- 2: 4 ou mais afirmações não suportadas, ou distorção de um resultado central
- 1: invenção de dado central (número, conclusão ou método incorretos)

A nota_fidelidade deve ser sempre coerente com o tamanho e a gravidade da
lista de afirmacoes_nao_suportadas: não atribua uma nota alta se a lista
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

    normalizadas_2 = [_normalizar_texto(a) for a in afirmacoes_2]
    consenso = []
    for afirmacao in afirmacoes_1:
        alvo = _normalizar_texto(afirmacao)
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
    cobertura, evidencias_cobertura, evidencias_suspeitas = avaliar_cobertura_semantica(original, gerado)
    fidelidade = avaliar_fidelidade(original, gerado)

    valores_cobertura_validos = [v for v in cobertura.values() if v is not None]
    cobertura_falhou = len(valores_cobertura_validos) == 0

    linha = {
        "arquivo": nome_arquivo,
        "abordagem": abordagem,
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bertscore_precisao": round(bert["precisao"], 4),
        "bertscore_recall": round(bert["recall"], 4),
        "bertscore_f1": round(bert["f1"], 4),
        "cobertura_score": sum(valores_cobertura_validos) if not cobertura_falhou else None,
        "cobertura_falha_parsing": cobertura_falhou,
        "cobertura_evidencia_suspeita": any(evidencias_suspeitas.values()),
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
        linha[f"cobertura_{elemento}_evidencia_suspeita"] = evidencias_suspeitas.get(elemento, False)

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