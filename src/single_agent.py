import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PASTA_PROCESSADOS = os.path.join("..", "data", "processed")
PASTA_OUTPUT = os.path.join("..", "data", "outputs", "single_agent")



def abstract_single_agent(nome_arquivo, texto_artigo):
    
    prompt_sistema = """
    Você é um pesquisador acadêmico.
    Sua tarefa é ler o corpo de um artigo científico e escrever um abstract científico claro e conciso.
    O resumo deve ser escrito obrigatoriamente em PORTUGUÊS do Brasil.

    O abstract deve conter:
    - problema
    - objetivo
    - método
    - resultados (se houver)
    - contribuição

    """

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Artigo para processar: {texto_artigo}"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content



def executar ():
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]
    if not os.path.exists(PASTA_OUTPUT):
        os.makedirs(PASTA_OUTPUT, exist_ok=True)
        print(f"Pasta criada: {PASTA_OUTPUT}")

    for nome in arquivos:
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