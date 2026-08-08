import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_resumo")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "reflection_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")


def resumo_reflection_agent(nome_arquivo, texto_artigo):
    prompt_sistema =  """
    Você é um pesquisador acadêmico.
    Sua tarefa é ler o corpo de um artigo científico e escrever um resumo científico claro e conciso.
    O resumo deve ser escrito obrigatoriamente em português do Brasil.

    O resumo deve conter:
    - problema
    - objetivo
    - método
    - resultados (se houver)
    - contribuição

    EXEMPLOS DE COMPORTAMENTO ESPERADO:

    Se o texto original diz: "Os testes no servidor Alpha reduziram o tempo de execução em 20%."
    CORRETO: "Houve redução de 20% no tempo de execução no servidor Alpha."
    ERRADO: "A automação reduziu o tempo em 20%." (generalizou o servidor Alpha para todo o estudo)
    ERRADO: "Reduziu o tempo significativamente." (omitiu o dado real e adicionou um qualificador vago)

    Se o texto original diz: "Os autores sugerem que a automação pode melhorar a manutenibilidade."
    CORRETO: "Os autores sugerem que a automação pode melhorar a manutenibilidade."
    ERRADO: "A automação melhora a manutenibilidade." (transformou uma sugestão em fato estabelecido)

    Se o texto original diz: "A ferramenta identificou 12 defeitos no módulo de login.":
    CORRETO: "A ferramenta identificou 12 defeitos no módulo de login."
    ERRADO: "A ferramenta identificou uma quantidade significativa de defeitos." (trocou o dado
    concreto por um qualificador vago que o texto não usa)
    """

    inicio = time.time()

    primeira_resposta_obj = client.chat.completions.create(        
        model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": texto_artigo}
            ],
            temperature=0.0
    )
    primeira_resposta = primeira_resposta_obj.choices[0].message.content

    

    prompt_auditoria = f"""
    Você é um auditor de fidelidade factual, não um editor de texto.
    Texto original: {texto_artigo}
    resumo atual: {primeira_resposta}

    Liste, em tópicos curtos, apenas os problemas encontrados no resumo atual, organizados nas
    seguintes categorias (pule a categoria se não houver problema nela):

    1. DADOS SEM SUPORTE: números, métodos, ferramentas ou conclusões que NÃO estão explicitamente
       no texto original.
    2. PERDA DE ESCOPO: um resultado restrito a um cenário/sistema/amostra/ferramenta específico
       foi generalizado como se fosse válido para todo o estudo.
    3. QUALIFICADOR SEM SUPORTE: palavras como "significativa", "essencial", "considerável",
       "demonstra que" foram adicionadas sem que o texto original expresse esse grau de certeza.
    4. INTERPRETAÇÃO COMO FATO: o artigo apresenta algo como sugestão, hipótese ou discussão, mas
       o resumo o escreve como conclusão definitiva.
    5. FUSÃO INDEVIDA: duas afirmações ou resultados distintos do artigo foram combinados criando
       uma relação de causa e efeito que o texto original não estabelece.

    Regras:
    - Não reescreva o resumo aqui. Apenas liste os problemas encontrados, citando a frase problemática.
    - Se não encontrar nenhum problema, responda apenas: "Nenhum problema encontrado."
    - Não questione nem altere números que já estão corretos e com o escopo correto, liste apenas o que 
      está genuinamente incorreto ou sem suporte.
    """

    auditoria_obj = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um auditor de fidelidade factual. Sua função é apenas apontar "
                                            "problemas, nunca reescrever texto."},
            {"role": "user", "content": prompt_auditoria}
        ],
        temperature=0.0
    )
    lista_problemas = auditoria_obj.choices[0].message.content

    prompt_correcao = f"""
    Texto original: {texto_artigo}
    resumo atual: {primeira_resposta}
    problemas identificados pela auditoria: {lista_problemas}

    Reescreva o resumo em português do Brasil, aplicando somente as correções necessárias para resolver os 
    problemas listados pela auditoria (removendo ou corrigindo essas informações específicas).
    não altere, reescreva ou reformule nenhuma outra parte do resumo que não tenha sido apontada como problema,
    preserve o texto original nessas partes, inclusive números e escopos já corretos.
    Se a auditoria não encontrou nenhum problema, retorne o resumo atual exatamente como está, sem nenhuma alteração.

    Retorne apenas o resumo final.
    """

    resposta_final_obj = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você aplica apenas as correções pontuais indicadas, preservando o "
                                            "restante do texto exatamente como está."},
            {"role": "user", "content": prompt_correcao}
        ],
        temperature=0.0
    )
    resposta_final = resposta_final_obj.choices[0].message.content

    tempo = time.time() - inicio

    uso1 = primeira_resposta_obj.usage
    uso2 = auditoria_obj.usage
    uso3 = resposta_final_obj.usage
    tokens_total = uso1.total_tokens + uso2.total_tokens + uso3.total_tokens
    registrar_custo(CAMINHO_CUSTO, {
        "arquivo": nome_arquivo,
        "abordagem": "reflection_agent",
        "tempo_segundos": round(tempo, 2),
        "tokens_prompt": uso1.prompt_tokens + uso2.prompt_tokens + uso3.prompt_tokens,
        "tokens_completion": uso1.completion_tokens + uso2.completion_tokens + uso3.completion_tokens,
        "tokens_total": tokens_total,
        "numero_chamadas": 3,
    })
    print(f"  {nome_arquivo}: {tempo:.2f}s, {tokens_total} tokens, 3 chamadas")

    return resposta_final

def executar():
    os.makedirs(PASTA_OUTPUT, exist_ok=True)
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]

    for nome in arquivos:
        print(f"Gerando resumo (reflection): {nome}")
        with open(os.path.join(PASTA_PROCESSADOS, nome), "r", encoding="utf-8") as f:
            conteudo = f.read()
        
        resultado = resumo_reflection_agent(nome, conteudo)
        
        with open(os.path.join(PASTA_OUTPUT, nome), "w", encoding="utf-8") as f:
            f.write(resultado)
    print("resumo concluída")

if __name__ == "__main__":
    executar()