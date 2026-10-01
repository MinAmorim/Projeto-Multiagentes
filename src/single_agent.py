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
PASTA_OUTPUT = os.path.join(CAMINHO_BASE, "data", "outputs", "single_agent")
CAMINHO_CUSTO = os.path.join(CAMINHO_BASE, "data", "resultados", "custo.csv")



def resumo_single_agent(nome_arquivo, texto_artigo):


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
isolado sem ligação com a anterior. Uma frase deve levar naturalmente à próxima.

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

REGRA DE REGISTRO (norma ABNT NBR 6028): escreva na voz ativa, terceira pessoa do singular
("o estudo identificou", "os autores desenvolveram"), nunca na voz passiva ("foi
desenvolvida"). Evite comentar o artigo de fora ("o artigo apresenta", "este trabalho
discute"); vá direto ao conteúdo.

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
    resposta = client.chat.completions.create(
      model = MODELO,
        messages = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"Artigo para processar: {texto_artigo}"}
        ],
        temperature=0.0,
        extra_body={"options": {"num_ctx": 16384}}
    )
    tempo = time.time() - inicio

    uso = resposta.usage
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

    return resposta.choices[0].message.content

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