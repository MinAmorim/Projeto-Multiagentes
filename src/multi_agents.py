import os
import time
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PASTA_PROCESSADOS = os.path.join("..", "data", "processed")
PASTA_OUTPUT = os.path.join("..", "data", "outputs", "multi_agent")

os.makedirs(PASTA_OUTPUT, exist_ok=True)

llm_padrao = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def gerar_abstract_multiagente(texto_artigo):
   
    agente_problema = Agent(
        role='Extrator de Problema e Lacuna',
        goal='Identificar de forma precisa o problema de pesquisa e a lacuna científica no texto fornecido.',
        backstory='Você é um pesquisador acadêmico experiente, focado em compreender a motivação e o problema central de artigos científicos.',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    agente_metodo = Agent(
        role='Extrator de Método',
        goal='Identificar a metodologia, o desenho do estudo, datasets e métricas utilizadas no texto.',
        backstory='Você é um especialista em metodologia científica, com olhar apurado para como a pesquisa foi executada.',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    agente_resultados = Agent(
        role='Extrator de Resultados e Contribuições',
        goal='Extrair os principais resultados, conclusões e contribuições empíricas do texto.',
        backstory='Você é um analista de dados acadêmicos especializado em sintetizar as descobertas e impactos de pesquisas.',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    agente_sintetizador = Agent(
        role='Sintetizador de Abstract Final',
        goal='Escrever um abstract científico claro, conciso e completo em PORTUGUÊS do Brasil, unindo os achados dos outros especialistas.',
        backstory='Você é um redator chefe e editor de uma renomada revista científica. Sua especialidade é redigir resumos perfeitos que englobem problema, objetivo, método e resultados, sem inventar informações.',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    tarefa_problema = Task(
        description=f'Leia o seguinte artigo e extraia o problema e a lacuna. Artigo: {texto_artigo}',
        expected_output='Um parágrafo descrevendo o problema de pesquisa.',
        agent=agente_problema
    )

    tarefa_metodo = Task(
        description=f'Leia o seguinte artigo e extraia o método e materiais. Artigo: {texto_artigo}',
        expected_output='Um parágrafo descrevendo a metodologia e o desenho do estudo.',
        agent=agente_metodo
    )

    tarefa_resultados = Task(
        description=f'Leia o seguinte artigo e extraia os resultados e contribuições. Artigo: {texto_artigo}',
        expected_output='Um parágrafo descrevendo os resultados e contribuições.',
        agent=agente_resultados
    )

    tarefa_sintese = Task(
        description='A partir do problema, método e resultados extraídos, redija o abstract final. O resumo deve ser escrito obrigatoriamente em PORTUGUÊS do Brasil.',
        expected_output='Um abstract científico completo e fluido, escrito em português do Brasil.',
        agent=agente_sintetizador,
        context=[tarefa_problema, tarefa_metodo, tarefa_resultados] # Agente 4 recebe o output dos Agentes 1, 2 e 3
    )

    equipe = Crew(
        agents=[agente_problema, agente_metodo, agente_resultados, agente_sintetizador],
        tasks=[tarefa_problema, tarefa_metodo, tarefa_resultados, tarefa_sintese],
        process=Process.sequential 
    )

    resultado = equipe.kickoff()
    return resultado

def executar():
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]
    
    for nome in arquivos:
        print(f"Processando com multiagentes: {nome}...")
        caminho_leitura = os.path.join(PASTA_PROCESSADOS, nome)
        
        with open(caminho_leitura, "r", encoding="utf-8") as f:
            conteudo = f.read()
            
        inicio = time.time()
        resumo_gerado = gerar_abstract_multiagente(conteudo)
        fim = time.time()
        
        print(f"Tempo de execução para {nome}: {fim - inicio:.2f} segundos")
        
        caminho_salvamento = os.path.join(PASTA_OUTPUT, nome)
        with open(caminho_salvamento, "w", encoding="utf-8") as f:
            f.write(str(resumo_gerado))
            
    print("Geração de abstracts concluida")

if __name__ == "__main__":
    executar()