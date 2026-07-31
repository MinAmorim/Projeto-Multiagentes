import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PASTA_PROCESSADOS = os.path.join("..", "data", "processed")
PASTA_OUTPUT = os.path.join("..", "data", "outputs", "reflection_agent")


def abstract_reflection_agent (texto_artigo):
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

    primeira_resposta = client.chat.completions.create(        
        model = "gpt-4o-mini",
            messages = [
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": texto_artigo}
            ],
            temperature=0
    
    ).choices[0].message.content

    prompt_reflexao = f"""
    Você é um revisor científico.
    Analise o abstract abaixo comparando com o texto original.
    Verifique: fidelidade, clareza, completude, presença de problema, método e resultados
    Texto original: {texto_artigo}
    Abstract atual: {primeira_resposta}
    Agora, reescreva o abstract em PORTUGUÊS corrigindo falhas e tornando-o mais técnico.
    """

    resposta_final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um revisor sênior."},
            {"role": "user", "content": prompt_reflexao}
        ],
        temperature=0

    ).choices[0].message.content
    
    return resposta_final

def executar():
    os.makedirs(PASTA_OUTPUT, exist_ok=True)
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]

    for nome in arquivos:
        print(f"Refletindo sobre: {nome}")
        with open(os.path.join(PASTA_PROCESSADOS, nome), "r", encoding="utf-8") as f:
            conteudo = f.read()
        
        resultado = abstract_reflection_agent(conteudo)
        
        with open(os.path.join(PASTA_OUTPUT, nome), "w", encoding="utf-8") as f:
            f.write(resultado)
    print("abordagem de reflexão concluída")

if __name__ == "__main__":
    executar()







