import os
import time
import textwrap
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()

CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_resumo")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "multi_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")

os.makedirs(PASTA_OUTPUT, exist_ok=True)


def novo_agente(role, goal, backstory, llm):
    return Agent(role=role, goal=goal, backstory=backstory, llm=llm, verbose=False, allow_delegation=False)


def montar_agentes(llm):
    problema = novo_agente(
        role='Extrator de Problema e Lacuna',
        goal='Identificar o problema de pesquisa e a lacuna científica do texto, sem afirmar a lacuna com '
             'mais certeza do que o artigo expressa.',
        backstory='Você é pesquisador acadêmico experiente e não infere uma lacuna além do que os autores '
                   'dizem explicitamente. Se eles não afirmam algo com certeza, você também não afirma.',
        llm=llm,
    )

    metodo = novo_agente(
        role='Extrator de Método',
        goal='Identificar a metodologia, o desenho do estudo, datasets e métricas usadas no texto.',
        backstory='Você entende bem de metodologia científica e presta atenção em como a pesquisa foi '
                   'conduzida.',
        llm=llm,
    )

    resultados = novo_agente(
        role='Extrator de Resultados e Contribuições',
        goal='Extrair os principais resultados, conclusões e contribuições do texto, preservando o cenário '
             'exato de cada resultado e sem apresentar a contribuição do estudo como mais certa, geral ou '
             'impactante do que os autores realmente afirmam.',
        backstory='Você sintetiza descobertas e impactos de pesquisas acadêmicas. Nunca amplia uma '
                   'contribuição além do que os autores afirmam. Se o resultado é preliminar ou específico '
                   'de um cenário, você mantém essa limitação em vez de generalizar.',
        llm=llm,
    )

    sintetizador = novo_agente(
        role='Sintetizador de Resumo Final',
        goal='Escrever um resumo científico claro e conciso em português do Brasil, unindo o que os outros '
             'agentes extraíram do artigo com fidelidade total ao texto original.',
        backstory='Você é editor-chefe de uma revista científica e nunca inventa dados. Usa as extrações '
                   'anteriores (problema, método, resultados) como roteiro, mas sempre confere números e '
                   'conclusões contra o artigo original antes de entregar o texto.',
        llm=llm,
    )

    return problema, metodo, resultados, sintetizador


TEMPLATE_PROBLEMA = textwrap.dedent('''
    Leia o artigo abaixo e diga qual é o problema e a lacuna de pesquisa. Use só o que os
    autores dizem de forma clara no texto, sem deixar a frase mais forte ou mais certa do
    que o artigo realmente permite. Para cada ideia, coloque entre aspas um trecho curto
    do artigo que comprove o que você escreveu. Não use referências bibliográficas
    formatadas (ex: "[Autor, ano]") nem mencione "a tabela X" ou "a figura X"; descreva a
    informação diretamente.
    Artigo: {texto_artigo}
    ''').strip()

TEMPLATE_METODO = textwrap.dedent('''
    Leia o artigo abaixo e diga qual foi o método e os materiais usados. Para cada ideia,
    coloque entre aspas um trecho curto do artigo que comprove o que você escreveu. Não use
    referências bibliográficas formatadas (ex: "[Autor, ano]") nem mencione "a tabela X" ou
    "a figura X"; descreva a informação diretamente.
    Artigo: {texto_artigo}
    ''').strip()

TEMPLATE_RESULTADOS = textwrap.dedent('''
    Leia o artigo abaixo e diga quais foram os resultados e as contribuições. Mantenha o
    cenário exato de cada número: se um resultado vale só pra uma parte do estudo, não
    escreva como se valesse pro estudo inteiro. Para cada ideia, coloque entre aspas um
    trecho curto do artigo que comprove o que você escreveu. Não use referências
    bibliográficas formatadas (ex: "[Autor, ano]") nem mencione "a tabela X", "a figura X"
    ou "o gráfico X"; descreva a informação diretamente.

    Exemplo: se o artigo diz "Os testes no servidor Alpha reduziram o tempo de execução em
    20%", escreva "Houve redução de 20% no tempo de execução no servidor Alpha". Não
    escreva "A automação reduziu o tempo em 20%", porque isso faria parecer que vale pra
    tudo, quando na verdade só vale pro servidor Alpha.

    Artigo: {texto_artigo}
    ''').strip()

TEMPLATE_SINTESE = textwrap.dedent('''
    Você tem em mãos o problema, o método e os resultados já extraídos do artigo nas etapas
    anteriores. As citações entre aspas nessas notas são só pra você conferir se bate com o
    artigo, não precisam aparecer no texto final.

    FORMATO OBRIGATÓRIO: escreva o resumo em português do Brasil, em um único parágrafo
    corrido, entre 150 e 250 palavras. Nunca use títulos, seções, listas numeradas ou
    tópicos. Nunca cite o nome dos autores, o título do artigo, nem a instituição ou
    universidade a que os autores são vinculados. Nunca use aspas. Nunca mencione "a
    tabela X", "a figura X" ou "o gráfico X"; descreva a informação diretamente.

    REGRA DE COESÃO: o resumo precisa ter começo, meio e fim, como um texto corrido e
    contínuo, não um amontoado de frases desconexas. Combine problema, método, resultado e
    contribuição numa narrativa única, sem separar um pedaço fixo pra cada etapa nem repetir
    sempre a mesma ordem ("o objetivo foi...", "o método envolveu...").

    REGRA DE SOBRIEDADE: evite adjetivos e advérbios de valor sem suporte direto no artigo
    (ex: "significativa", "essencial", "considerável") se o artigo não expressa esse grau.

    REGRAS DE FIDELIDADE: fique fiel ao artigo o tempo todo, use só o que está nas notas e
    no texto original. Se um resultado vale só pra um cenário específico, mantenha essa
    restrição, não vire uma afirmação geral. Se algo era só uma sugestão dos autores,
    continue tratando como sugestão. Não crie uma relação de causa e efeito entre dois fatos
    que o artigo não liga diretamente. Evite "embora", "apesar de", "no entanto", "contudo"
    a não ser que o artigo genuinamente discuta aquela contradição ali. Se usar uma sigla,
    escreva por extenso entre parênteses na primeira aparição, mesmo que as notas já tenham
    explicado antes.

    REGRA DE REGISTRO (ABNT NBR 6028): escreva na voz ativa e na 3ª pessoa do singular (ex:
    "o estudo identificou", "os autores desenvolveram"), nomeando quem pratica a ação em vez
    de usar voz passiva. Evite comentar o artigo de fora ("o artigo apresenta", "este
    trabalho discute"); vá direto ao conteúdo.

    REGRA DE ABERTURA: a primeira frase precisa dizer, de forma direta e específica, do que
    o artigo trata. Não comece com "Foi desenvolvida", "Foi proposto", "O presente trabalho
    trata" ou qualquer frase genérica que serviria pra qualquer artigo da área só trocando
    o nome.

    EXEMPLO DE RESUMO NO FORMATO CORRETO:
    "Os movimentos sociais que surgiram a partir da década de 1970 imprimiram uma nova noção de 
cidadania através da participação popular para a ampliação de espaços públicos. Propõe-se 
a observar a possibilidade de reconhecimento de um espaço público de interlocução e deliberação, 
segundo um modelo de atenção pública não estatal, focalizando o caso da Organização Social de Saúde 
Hospital Geral do Grajaú. Trata-se de entidade instituída com base na proposta de parcerias entre 
Estado e sociedade civil, do governo federal, de reforma do aparelho de Estado, com características 
próprias no Estado de São Paulo - exclusividade para o Sistema Único de Saúde, serviço novo e controle 
da Secretaria Estadual de Saúde. Por meio de estudo da legislação pertinente e com uso de metodologia
qualitativa, procedeu-se à observação participante e a entrevistas semi-estruturadas, com lideranças 
de movimentos sociais e de gerentes do Estado na região das sub-Prefeituras de Capela da Socorro e 
Parelheiros. O estudo recuperou a história de participação popular na região por recursos que 
possibilitassem condições de vida e saúde, caracterizando atores que se mantêm atuantes, e buscam o 
diálogo institucional no sistema de saúde e, em especial, na organização social. Constatou a carência 
de recursos para atender à demanda de saúde na região, para a qual a organização social vem dando respostas, 
e as dificuldades em estabelecer um sistema referenciado. Observou possibilidades de interlocução entre a 
população organizada e a organização social. Concluiu que parcerias reguladas se efetivam no cotidiano e 
que para tal, é necessário também, postura participativa, bem como, permeabilidade para relações democráticas."

    Antes de terminar, releia o texto e confira: é um único parágrafo, tem no máximo 250
    palavras, não tem aspas, não cita autor/instituição, a abertura não é genérica, e cada
    frase tem apoio direto no artigo.

    Texto original do artigo: {texto_artigo}
    ''').strip()


def montar_tarefas(agentes, texto_artigo):
    agente_problema, agente_metodo, agente_resultados, agente_sintetizador = agentes

    tarefa_problema = Task(
        agent=agente_problema,
        description=TEMPLATE_PROBLEMA.format(texto_artigo=texto_artigo),
        expected_output='Um parágrafo descrevendo o problema de pesquisa, com cada afirmação acompanhada de '
                         'uma citação literal curta entre aspas que a comprove no texto original.',
    )

    tarefa_metodo = Task(
        agent=agente_metodo,
        description=TEMPLATE_METODO.format(texto_artigo=texto_artigo),
        expected_output='Um parágrafo descrevendo a metodologia e o desenho do estudo, com cada afirmação '
                         'acompanhada de uma citação literal curta entre aspas que a comprove no texto original.',
    )

    tarefa_resultados = Task(
        agent=agente_resultados,
        description=TEMPLATE_RESULTADOS.format(texto_artigo=texto_artigo),
        expected_output='Um parágrafo descrevendo os resultados e contribuições, com cada afirmação '
                         'acompanhada de uma citação literal curta entre aspas que a comprove no texto original.',
    )

    tarefa_sintese = Task(
        agent=agente_sintetizador,
        context=[tarefa_problema, tarefa_metodo, tarefa_resultados],
        description=TEMPLATE_SINTESE.format(texto_artigo=texto_artigo),
        expected_output='Um resumo científico: um único parágrafo corrido, entre 150 e 250 palavras, 100% '
                         'fiel ao texto original, escrito em português do Brasil, sem aspas, sem nome de '
                         'autores/instituição/título, sem comentar o artigo de fora, e com abertura '
                         'específica do artigo (não genérica).',
    )

    return tarefa_problema, tarefa_metodo, tarefa_resultados, tarefa_sintese


def gerar_resumo_multiagente(nome_arquivo, texto_artigo):
    llm_padrao = LLM(
        model="ollama/qwen2.5-16k",
        base_url="http://localhost:11434",
        temperature=0.0,
    )
    agentes = montar_agentes(llm_padrao)
    tarefas = montar_tarefas(agentes, texto_artigo)

    equipe = Crew(agents=list(agentes), tasks=list(tarefas), process=Process.sequential)

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

    print("Geração de resumos concluído")


if __name__ == "__main__":
    executar()