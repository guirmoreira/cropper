"""Templates de prompt (constantes), em portugues."""

from __future__ import annotations

SISTEMA_MELHORA_DESCRICAO = """\
Voce reformula a descricao vaga de um objeto visual como faria com uma query de busca: \
mais clara e especifica, sem adicionar informacao que o usuario nao forneceu.

Regras:
- Preserve a intencao original. Nao invente atributos, formato, cor, posicao ou qualquer \
outro detalhe visual que o usuario nao tenha mencionado.
- Nao faca suposicoes sobre a imagem: voce nao tem acesso a ela e nao deve descrever como \
se a tivesse visto.
- Apenas reescreva a descricao para remover ambiguidade de linguagem (sinonimos vagos, \
referencias soltas, termos genericos), da mesma forma que se reescreve uma busca para \
torna-la mais eficaz.
- Liste criterios de exclusao apenas quando decorrerem explicitamente do texto do usuario \
(ex.: "nao o botao de cancelar" quando o usuario pede "o botao de confirmar").
- Maximo de 60 palavras na descricao.
- Se a descricao original ja for clara, devolva-a praticamente inalterada.\
"""

SISTEMA_DESCREVE_IMAGEM = """\
Voce analisa um recorte de uma imagem maior. Descreva objetivamente o que esta visivel.

Regras:
- Liste todos os elementos visuais distinguiveis (componentes de UI, blocos de texto, \
imagens, icones, graficos, areas de conteudo).
- Transcreva textos legiveis, sem interpreta-los.
- Nao especule sobre o que esta fora do recorte.
- Nao avalie qualidade nem faca sugestoes.\
"""

SISTEMA_ESCOLHE_IMAGEM = """\
Voce recebe a descricao de um objeto-alvo e as descricoes de varios recortes numerados de \
uma mesma imagem. Ordene os recortes pela probabilidade de conterem o objeto-alvo inteiro.

Regras:
- Atribua pontuacao de 0.0 a 1.0 a cada recorte.
- Prefira recortes que contenham o objeto completo, nao fragmentos.
- Inclua todos os recortes no resultado, mesmo os de pontuacao zero.
- Justifique cada pontuacao em no maximo 20 palavras.\
"""

SISTEMA_ENCONTRA_OBJETO = """\
Voce localiza objetos em imagens e devolve suas coordenadas.

Sistema de coordenadas (obrigatorio): a imagem tem um espaco normalizado onde o canto \
superior esquerdo e (0, 0) e o canto inferior direito e (1000, 1000), independentemente \
das dimensoes reais em pixels. Devolva sempre numeros inteiros nesse espaco.

Regras:
- A caixa deve conter o objeto completo, com margem minima. Nao inclua elementos vizinhos \
que nao fazem parte dele.
- x1 > x0 e y1 > y0, sempre.
- Se o objeto nao estiver presente ou aparecer apenas parcialmente cortado na borda, devolva \
encontrado: false e explique em observacao.
- Nao devolva a imagem inteira como caixa. Se a unica resposta possivel for a imagem toda, \
isso significa que voce nao localizou o objeto.
- Se houver feedback de tentativa anterior, ele descreve o que estava errado no recorte que \
voce propos -- corrija exatamente aquilo.\
"""

SISTEMA_JULGA_RESULTADO = """\
Voce e um revisor rigoroso. Recebe (a) a descricao de um objeto que deveria ter sido \
recortado de uma imagem e (b) o recorte produzido. Julgue se o recorte corresponde ao \
objeto pedido.

Reprove quando:
- o objeto aparece cortado nas bordas;
- o recorte contem muito conteudo extra alem do objeto;
- o recorte mostra um elemento diferente do pedido;
- o recorte esta vazio, ilegivel ou e apenas fundo.

Aprove quando o objeto pedido esta inteiro e o enquadramento e justo.

O campo `feedback` e sempre obrigatorio. Se reprovar, descreva de forma acionavel em que \
direcao e quanto ajustar (ex.: "o cabecalho do card esta cortado no topo; expanda cerca de \
10% para cima e reduza 15% a direita, onde ha conteudo do card vizinho"). Nunca escreva \
apenas "nao corresponde".\
"""

SISTEMA_TRADUCAO = """\
Traduza o texto a seguir do portugues para o ingles. Devolva apenas a traducao, sem \
comentarios, no campo `texto_ingles`. Preserve o sentido literal -- este texto sera usado \
como prompt de um detector de objetos open-vocabulary, nao como texto para um leitor humano.\
"""

INSTRUCAO_JSON = """\
Responda exclusivamente com um objeto JSON valido que satisfaca o schema abaixo. Nao \
escreva texto antes ou depois. Nao use blocos de codigo markdown.

Schema: {schema_json}\
"""


def mensagem_usuario_encontra_objeto(
    *,
    largura: int,
    altura: int,
    descricao_melhorada: str,
    criterios_exclusao: list[str],
    bloco_feedback: str = "",
) -> str:
    criterios = "\n".join(f"- {c}" for c in criterios_exclusao) if criterios_exclusao else "(nenhum)"
    partes = [
        f"Dimensoes reais do recorte: {largura}x{altura} px (use ainda assim o espaco 0..1000).",
        "",
        "OBJETO A LOCALIZAR:",
        descricao_melhorada,
        "",
        "NAO CONFUNDIR COM:",
        criterios,
    ]
    if bloco_feedback:
        partes.extend(["", bloco_feedback])
    return "\n".join(partes)


def bloco_feedback_tentativa_anterior(
    *, x0: int, y0: int, x1: int, y1: int, feedback: str
) -> str:
    return (
        f"TENTATIVA ANTERIOR -- coordenadas propostas: [{x0}, {y0}, {x1}, {y1}]\n"
        f"AVALIACAO DO REVISOR: {feedback}\n"
        "Ajuste as coordenadas para corrigir o problema apontado."
    )
