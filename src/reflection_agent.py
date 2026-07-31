import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_abstract")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "reflection_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")


def abstract_reflection_agent(nome_arquivo, texto_artigo):
    prompt_sistema =  """
    Você é um pesquisador acadêmico.
    Sua tarefa é ler o corpo de um artigo científico e escrever um abstract científico claro e conciso.
    O resumo deve ser escrito obrigatoriamente em português do Brasil.

    O abstract deve conter:
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
            temperature=0
    )
    primeira_resposta = primeira_resposta_obj.choices[0].message.content

    prompt_reflexao = f"""
    Você é um revisor científico.
    Analise o abstract abaixo comparando com o texto original.
    Verifique: fidelidade, clareza, completude, presença de problema, método e resultados
    Texto original: {texto_artigo}
    Abstract atual: {primeira_resposta}
    Agora, reescreva o abstract em PORTUGUÊS corrigindo falhas e tornando-o mais técnico.
    """

    resposta_final_obj = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um revisor sênior."},
            {"role": "user", "content": prompt_reflexao}
        ],
        temperature=0
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
        
        resultado = abstract_reflection_agent(nome, conteudo)
        
        with open(os.path.join(PASTA_OUTPUT, nome), "w", encoding="utf-8") as f:
            f.write(resultado)
    print("resumo concluída")

if __name__ == "__main__":
    executar()