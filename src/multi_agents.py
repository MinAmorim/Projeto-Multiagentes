import os
import time
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_resumo")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "multi_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")

os.makedirs(PASTA_OUTPUT, exist_ok=True)

def gerar_resumo_multiagente(nome_arquivo, texto_artigo):

    llm_padrao = LLM(model="gpt-4o-mini", temperature=0.0)
   
    agente_problema = Agent(
    role='Extrator de Problema e Lacuna',
    goal='Identificar de forma precisa o problema de pesquisa e a lacuna científica no texto fornecido, '
         'sem afirmar a existência da lacuna com mais certeza do que o próprio artigo expressa.',
    backstory='Você é um pesquisador acadêmico experiente. Você nunca infere ou reforça uma lacuna '
              'científica além do que os autores dizem explicitamente. Se os autores não afirmam '
              'categoricamente que algo é uma lacuna, você não afirma isso categoricamente também.',
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
        goal='Extrair os principais resultados, conclusões e contribuições empíricas do texto, preservando '
             'exatamente as condições, cenários ou sistemas a que cada resultado se refere, e sem apresentar '
             'a contribuição do estudo com mais certeza, generalidade ou impacto do que os autores expressam '
             'explicitamente no artigo.',
        backstory='Você é um analista de dados acadêmicos especializado em sintetizar as descobertas e impactos '
                  'de pesquisas. Você nunca amplia o alcance de uma contribuição além do que os próprios autores '
                  'afirmam, se os autores apresentam algo como um resultado preliminar ou específico de um '
                  'cenário, você preserva essa limitação em vez de generalizar.',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    agente_sintetizador = Agent(
        role='Sintetizador de Resumo Final',
        goal='Escrever um resumo científico claro, conciso e completo em português do Brasil, unindo os achados dos outros especialistas e mantendo fidelidade absoluta ao artigo.',
        backstory='Você é um redator chefe e editor de uma renomada revista científica. Você nunca inventa dados. Você usa as notas dos especialistas apenas como um roteiro estrutural, mas sempre valida e extrai os termos, números e conclusões exatas do artigo original fornecido. Antes de entregar qualquer texto, você sempreconfere cada número e conclusão contra o artigo original, palavra por palavra',
        verbose=False,
        allow_delegation=False,
        llm=llm_padrao
    )

    tarefa_problema = Task(
        description=f'Leia o seguinte artigo e extraia o problema e a lacuna. Regra: cite apenas o que os autores afirmam explicitamente sobre o problema/lacuna. Não reforce, generalize nem torne a afirmação mais categórica do que o texto original permite. Artigo: {texto_artigo}',
        expected_output='Um parágrafo descrevendo o problema de pesquisa.',
        agent=agente_problema
    )

    tarefa_metodo = Task(
        description=f'Leia o seguinte artigo e extraia o método e materiais. Artigo: {texto_artigo}',
        expected_output='Um parágrafo descrevendo a metodologia e o desenho do estudo.',
        agent=agente_metodo
    )

    tarefa_resultados = Task(
        description=f'''Leia o seguinte artigo e extraia os resultados e contribuições.
        Regra: preserve o cenário/sistema/condição exata a que cada resultado numérico se refere. Ao descrever 
        a contribuição do estudo, não amplie seu alcance nem a apresente como mais certa ou geral do que os 
        autores expressam, cite apenas o que está explicitamente afirmado.
        Artigo: {texto_artigo}''',
        expected_output='Um parágrafo descrevendo os resultados e contribuições.',
        agent=agente_resultados
    )

    tarefa_sintese = Task(
        description=f'''A partir do problema, método e resultados extraídos pelos outros especialistas, redija o resumo final.
        Siga este processo em duas etapas:
        1 - Utilize as notas produzidas pelos especialistas como base para estruturar o resumo, tomando sempre o texto original como principal referência.
        2 - Revise o resumo elaborado e verifique se todas as afirmações numéricas, métodos, ferramentas e conclusões apresentados estão explicitamente fundamentados no texto original. Confirme também se cada informação preserva o mesmo contexto e escopo em que foi apresentada no artigo. Quando um resultado estiver restrito a um cenário, método, ferramenta ou módulo específico, essa restrição deve permanecer explícita no resumo. Caso identifique informações sem suporte direto, interpretações ou generalizações indevidas, corrija-as ou remova-as. Em situações de dúvida, prefira uma formulação mais conservadora, compatível com as evidências disponíveis no texto.
        3 - Verifique também o tom das frases de abertura (problema/lacuna) e fechamento (contribuição): elas devem refletir o mesmo grau de certeza que os autores expressam no artigo. Se os autores sugerem ou discutem algo, não reescreva como se fosse um fato estabelecido. Prefira formulações como "os autores argumentam que..." quando a afirmação for interpretação, não fato direto.
        Retorne apenas a versão final do resumo, sem incluir o rascunho ou o processo de revisão.
        Texto original do artigo: {texto_artigo}
        O resumo deve ser escrito obrigatoriamente em português do Brasil e ser um texto coeso e fluido.''',
        expected_output='Um resumo científico completo e fluido, 100% fiel ao texto original, escrito em português do Brasil.',
        agent=agente_sintetizador,
        context=[tarefa_problema, tarefa_metodo, tarefa_resultados] 
    )

    equipe = Crew(
        agents=[agente_problema, agente_metodo, agente_resultados, agente_sintetizador],
        tasks=[tarefa_problema, tarefa_metodo, tarefa_resultados, tarefa_sintese],
        process=Process.sequential 
    )

    inicio = time.time()
    resultado = equipe.kickoff()
    tempo = time.time() - inicio

    metricas = equipe.calculate_usage_metrics()
    registrar_custo(CAMINHO_CUSTO, {
        "arquivo": nome_arquivo,
        "abordagem": "multi_agent",
        "tempo_segundos": round(tempo, 2),
        "tokens_prompt": metricas.prompt_tokens,
        "tokens_completion": metricas.completion_tokens,
        "tokens_total": metricas.total_tokens,
        "numero_chamadas": metricas.successful_requests,
    })
    print(f"  {nome_arquivo}: {tempo:.2f}s, {metricas.total_tokens} tokens, {metricas.successful_requests} chamadas")

    return resultado

def executar():
    arquivos = [f for f in os.listdir(PASTA_PROCESSADOS) if f.endswith(".txt")]
    
    for nome in arquivos:
        print(f"Processando com multiagentes: {nome}...")
        caminho_leitura = os.path.join(PASTA_PROCESSADOS, nome)
        
        with open(caminho_leitura, "r", encoding="utf-8") as f:
            conteudo = f.read()
            
        resumo_gerado = gerar_resumo_multiagente(nome, conteudo)
        
        caminho_salvamento = os.path.join(PASTA_OUTPUT, nome)
        with open(caminho_salvamento, "w", encoding="utf-8") as f:
            f.write(str(resumo_gerado))
            
    print("Geração de resumos concluido")

if __name__ == "__main__":
    executar()