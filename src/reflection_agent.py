import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from utils.custo import registrar_custo

load_dotenv()
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODELO = "qwen2.5-16k"
CAMINHO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_PROCESSADOS = os.path.join(CAMINHO_BASE, "data", "processed_sem_resumo")
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "reflection_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")


def resumo_reflection_agent(nome_arquivo, texto_artigo):
    prompt_sistema = """
    Você é um pesquisador acadêmico. Leia o corpo de um artigo científico e escreva um resumo
    em português do Brasil.
    FORMATO OBRIGATÓRIO: um único parágrafo corrido, entre 150 e 250 palavras. Nunca use
    títulos, seções, listas numeradas ou tópicos. Nunca cite o nome dos autores, o título do
    artigo, nem a instituição ou universidade a que os autores são vinculados. Nunca use
    aspas com trechos literais do texto original, reescreva com suas próprias palavras. Nunca
    mencione "a tabela X", "a figura X" ou "o gráfico X"; descreva a informação diretamente,
    sem referenciar o elemento visual de onde ela veio.


    REGRA DE COESÃO: o resumo precisa ter começo, meio e fim, como um texto corrido e
    contínuo, não um amontoado de frases desconexas em que cada uma trata de um tópico
    isolado sem ligação com a anterior. Uma frase deve levar naturalmente à próxima. Não
    transforme problema/objetivo/método/resultados/contribuição em uma lista mental de
    frases separadas, cada uma sempre abrindo com o nome do item ("o objetivo foi...", "o
    método envolveu..."). Combine itens relacionados na mesma frase quando fizer sentido, e
    deixe a ordem seguir a ênfase real do artigo, não um roteiro fixo.

    REGRA DE SOBRIEDADE: evite o uso excessivo de adjetivos e advérbios que não tenham
    suporte direto no texto original. Prefira descrever o fato de forma direta a qualificá-lo
    com termos de valor (ex: em vez de "resultados extremamente positivos", descreva o
    resultado concreto que o artigo apresenta).

    REGRAS DE FIDELIDADE:
    - Use só o que está no texto fornecido. Não invente dados, métricas ou conclusões.
    - Ao citar qualquer número ou resultado, inclua o cenário/sistema/amostra exato a que ele
      se refere. Nunca generalize um resultado específico como se valesse para o estudo todo.
    - Se o artigo trata algo como sugestão ou hipótese ("os autores sugerem que..."), preserve
      esse grau de incerteza no resumo. Não transforme em afirmação categórica.
    - Não funda dois resultados distintos numa relação de causa e efeito que o artigo não
      estabelece explicitamente.
    - Se usar uma sigla, escreva por extenso entre parênteses na primeira aparição.
    - Não adicione ressalvas como "embora", "no entanto", "apesar de" a não ser que o artigo
      genuinamente discuta aquele contraponto naquele ponto específico.

    REGRA DE REGISTRO (norma ABNT NBR 6028): escreva na voz ativa, terceira pessoa do
    singular ("o estudo identificou", "os autores desenvolveram"), nunca na voz passiva
    ("foi desenvolvida"). Evite comentar o artigo de fora ("o artigo apresenta", "este
    trabalho discute"); vá direto ao conteúdo.

    REGRA DE ABERTURA: não comece com uma frase genérica que serviria para qualquer artigo da
    área (ex: "a automação de testes é importante para a qualidade do software"). Comece pelo
    problema, ferramenta ou achado específico deste artigo.

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

    CASOS CERTO/ERRADO:
    - Original: "Os testes no servidor Alpha reduziram o tempo de execução em 20%."
      CERTO: "Houve redução de 20% no tempo de execução no servidor Alpha."
      ERRADO: "A automação reduziu o tempo em 20%." (generalizou um cenário específico)
    - Original: "Os autores sugerem que a automação pode melhorar a manutenibilidade."
      CERTO: manter como sugestão.
      ERRADO: "A automação melhora a manutenibilidade." (virou fato definitivo)
    - ERRADO: "O artigo de João Silva apresenta uma ferramenta..." (cita nome de autor)
      ERRADO: "Os autores afirmam que 'a ferramenta reduziu o tempo em 40%'." (usa aspas)

    Antes de responder, confira: o texto é um único parágrafo, tem entre 150 e 250 palavras,
    não tem headers nem listas, não cita autor, tem fluxo contínuo entre as frases, e cada
    frase tem suporte direto no artigo.
    """

    inicio = time.time()

    resposta_geracao = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_artigo}
        ],
        temperature=0.0,
        extra_body={"options": {"num_ctx": 16384}}
    )
    resumo_gerado = resposta_geracao.choices[0].message.content

    prompt_auditoria = f"""
    Você é um auditor de fidelidade factual, não um editor de texto.
    Texto original: {texto_artigo}
    resumo atual: {resumo_gerado}

    Liste, em tópicos curtos, apenas os problemas encontrados no resumo atual, organizados
    nas categorias abaixo (pule a categoria se não houver problema nela):

    1. DADOS SEM SUPORTE: números, métodos, ferramentas ou conclusões que não estão
       explicitamente no texto original.
    2. PERDA DE ESCOPO: um resultado restrito a um cenário/sistema/amostra específico foi
       generalizado como se valesse para todo o estudo.
    3. QUALIFICADOR SEM SUPORTE: palavras como "significativa", "essencial", "considerável"
       foram usadas sem que o texto original expresse esse grau de certeza.
    4. INTERPRETAÇÃO COMO FATO: o artigo apresenta algo como sugestão ou hipótese, mas o
       resumo escreve como conclusão definitiva.
    5. FUSÃO INDEVIDA: dois resultados distintos foram combinados numa relação de causa e
       efeito que o artigo não estabelece.
    6. REGISTRO INADEQUADO: o resumo comenta o artigo de fora ("o presente trabalho trata",
       "o artigo apresenta") em vez de ir direto ao conteúdo; ou usa voz passiva em vez de
       voz ativa na 3ª pessoa; ou cita nome de autor/título; ou usa aspas com trecho literal.
    7. ESTRUTURA ENGESSADA: o resumo trata problema/objetivo/método/resultados como uma
       lista mental de frases separadas, sem fluxo entre elas.
    8. ABERTURA GENÉRICA: a primeira frase reafirma o campo geral da pesquisa de forma
       genérica, em vez de começar pelo problema ou achado específico do artigo.
    9. RESSALVA SEM SUPORTE: o resumo insere "embora", "no entanto", "apesar de" sem que o
       artigo discuta explicitamente essa limitação naquele ponto.
    10. SIGLA NÃO EXPLICADA: uma sigla aparece sem o significado por extenso na primeira vez.
    11. REFERÊNCIA A ELEMENTO VISUAL: o resumo menciona "a tabela X", "a figura X" ou "o
        gráfico X" em vez de descrever a informação diretamente.
    12. ADJETIVAÇÃO EXCESSIVA: adjetivos ou advérbios de valor sem suporte direto no artigo.

    Regras:
    - Não reescreva o resumo aqui, apenas liste os problemas, citando a frase problemática.
    - Se não encontrar nenhum problema, responda apenas: "Nenhum problema encontrado."
    - Não questione números que já estão corretos e com o escopo correto.
    """

    resposta_auditoria = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": "Você é um auditor de fidelidade factual. Sua função é apenas apontar "
                                            "problemas, nunca reescrever texto."},
            {"role": "user", "content": prompt_auditoria}
        ],
        temperature=0.0,
        extra_body={"options": {"num_ctx": 16384}}
    )
    lista_problemas = resposta_auditoria.choices[0].message.content

    prompt_correcao = f"""
    Texto original: {texto_artigo}
    resumo atual: {resumo_gerado}
    problemas identificados pela auditoria: {lista_problemas}

    Reescreva o resumo em português do Brasil, aplicando somente as correções necessárias para
    resolver os problemas listados pela auditoria (removendo ou corrigindo essas informações
    específicas). Não altere, reescreva ou reformule nenhuma outra parte do resumo que não
    tenha sido apontada como problema, preserve o texto original nessas partes, inclusive
    números e escopos já corretos. Se a auditoria não encontrou nenhum problema, retorne o
    resumo atual exatamente como está, sem nenhuma alteração.

    Retorne apenas o resumo final, em um único parágrafo.
    """

    resposta_correcao = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": "Você aplica apenas as correções pontuais indicadas, preservando o "
                                            "restante do texto exatamente como está."},
            {"role": "user", "content": prompt_correcao}
        ],
        temperature=0.0,
        extra_body={"options": {"num_ctx": 16384}}
    )
    resumo_corrigido = resposta_correcao.choices[0].message.content

    tempo = time.time() - inicio

    uso_geracao = resposta_geracao.usage
    uso_auditoria = resposta_auditoria.usage
    uso_correcao = resposta_correcao.usage
    tokens_total = uso_geracao.total_tokens + uso_auditoria.total_tokens + uso_correcao.total_tokens
    registrar_custo(CAMINHO_CUSTO, {
        "arquivo": nome_arquivo,
        "abordagem": "reflection_agent",
        "tempo_segundos": round(tempo, 2),
        "tokens_prompt": uso_geracao.prompt_tokens + uso_auditoria.prompt_tokens + uso_correcao.prompt_tokens,
        "tokens_completion": uso_geracao.completion_tokens + uso_auditoria.completion_tokens + uso_correcao.completion_tokens,
        "tokens_total": tokens_total,
        "numero_chamadas": 3,
    })
    print(f"  {nome_arquivo}: {tempo:.2f}s, {tokens_total} tokens, 3 chamadas")

    return resumo_corrigido

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