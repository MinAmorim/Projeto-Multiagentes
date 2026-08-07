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

    Liste, em tópicos curtos, apenas os problemas encontrados no resumo atual:
    - Afirmações numéricas, métodos, ferramentas ou conclusões que NÃO estão explicitamente no texto original.
    - Afirmações que generalizam um resultado restrito a um cenário/sistema/amostra específico como se fosse geral.
    
    Regras:
    - NÃO reescreva o resumo aqui. Apenas liste os problemas encontrados, citando a frase problemática.
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
    uso2 = resposta_final_obj.usage
    tokens_total = uso1.total_tokens + uso2.total_tokens
    registrar_custo(CAMINHO_CUSTO, {
        "arquivo": nome_arquivo,
        "abordagem": "reflection_agent",
        "tempo_segundos": round(tempo, 2),
        "tokens_prompt": uso1.prompt_tokens + uso2.prompt_tokens,
        "tokens_completion": uso1.completion_tokens + uso2.completion_tokens,
        "tokens_total": tokens_total,
        "numero_chamadas": 2,
    })
    print(f"  {nome_arquivo}: {tempo:.2f}s, {tokens_total} tokens, 2 chamadas")

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