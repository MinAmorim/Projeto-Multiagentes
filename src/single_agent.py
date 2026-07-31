import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_abstract")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "single_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")



def abstract_single_agent(nome_arquivo, texto_artigo):
    
    prompt_sistema = """
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
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Artigo para processar: {texto_artigo}"}
        ],
        temperature=0.0
    )
    tempo = time.time() - inicio

    uso = response.usage
    registrar_custo(CAMINHO_CUSTO, {
        "arquivo": nome_arquivo,
        "abordagem": "single_agent",
        "tempo_segundos": round(tempo, 2),
        "tokens_prompt": uso.prompt_tokens,
        "tokens_completion": uso.completion_tokens,
        "tokens_total": uso.total_tokens,
        "numero_chamadas": 1,
    })
    print(f"  {nome_arquivo}: {tempo:.2f}s, {uso.total_tokens} tokens, 1 chamada")

    return response.choices[0].message.content

def executar ():
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]
    if not os.path.exists(PASTA_OUTPUT):
        os.makedirs(PASTA_OUTPUT, exist_ok=True)
        print(f"Pasta criada: {PASTA_OUTPUT}")

    for nome in arquivos:
        print(f"Gerando resumo (single agent): {nome}")
        caminho_leitura = os.path.join(PASTA_PROCESSADOS, nome)
        with open(caminho_leitura, "r", encoding="utf-8") as f:
            conteudo = f.read()
        
        resumo_gerado = abstract_single_agent(nome, conteudo)

        caminho_salvamento = os.path.join(PASTA_OUTPUT, nome)
        with open(caminho_salvamento, "w", encoding="utf-8") as f:
            f.write(resumo_gerado)
    print(" resumo concluído")


if __name__ == "__main__":
    executar()