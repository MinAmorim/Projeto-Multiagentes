import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_resumo")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "single_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")



def resumo_single_agent(nome_arquivo, texto_artigo):
    
    prompt_sistema = """
    Você é um pesquisador acadêmico.
    Sua tarefa é ler o corpo de um artigo científico e escrever um resumo científico claro e conciso.
    O resumo deve ser escrito obrigatoriamente em português do Brasil.

    Regras obrigatórias:
    - Utilize apenas as informações presentes no texto fornecido.
    - Não acrescente informações externas, nem faça suposições
    - Não invente dados, métricas ou conclusões
    - Ao citar qualquer número, percentual, taxa ou resultado quantitativo, inclua explicitamente 
      a condição, cenário, sistema, amostra ou ferramenta a que ele se refere, exatamente como no 
      texto original. Nunca generalize um resultado válido apenas para um caso específico como se 
      fosse um resultado geral do estudo.
    - Não transforme sugestões, hipóteses ou discussões dos autores em afirmações categóricas.
      Se o artigo diz "os autores sugerem" ou "os resultados indicam", preserve esse grau de 
      certeza no resumo, não escreva como fato definitivo.
    - Não funda duas frases ou resultados distintos do artigo em uma única afirmação que crie
      uma relação de causa e efeito que não está explícita no texto original.
    - Caso alguma das informações solicitadas não esteja presente ou não esteja clara no texto original, simplesmente não a inclua na resposta

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

    Se o texto original diz: "A cobertura passou de 15% em 2020 para 43% em 2023." e, em outro
    trecho, "a equipe relatou maior confiança no processo de deploy":
    CORRETO: manter as duas informações como observações separadas.
    ERRADO: "O aumento da cobertura de 15% para 43% resultou em maior confiança da equipe no
    deploy." (criou uma relação causal entre dois fatos que o artigo não conecta explicitamente)

    Se o texto original diz: "A ferramenta identificou 12 defeitos no módulo de login.":
    CORRETO: "A ferramenta identificou 12 defeitos no módulo de login."
    ERRADO: "A ferramenta identificou uma quantidade significativa de defeitos." (trocou o dado
    concreto por um qualificador vago que o texto não usa)
    ERRADO: "A ferramenta demonstrou grande eficácia na detecção de defeitos." (inseriu uma
    avaliação de mérito, "grande eficácia", que o artigo não faz)

    Antes de finalizar sua resposta, revise mentalmente cada frase do resumo e confirme que ela
    tem suporte direto e no mesmo escopo do texto original.
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
        
        resumo_gerado = resumo_single_agent(nome, conteudo)

        caminho_salvamento = os.path.join(PASTA_OUTPUT, nome)
        with open(caminho_salvamento, "w", encoding="utf-8") as f:
            f.write(resumo_gerado)
    print(" resumo concluído")


if __name__ == "__main__":
    executar()