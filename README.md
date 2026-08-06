# detector-objetos

Detector de objetos em imagem hibrido: um detector open-vocabulary local
(Florence-2 ou Grounding DINO) localiza o objeto e uma LLM multimodal
(Claude Haiku 4.5, via LiteLLM) julga se o recorte encontrado corresponde
ao pedido. Um motor 100% via LLM (`--motor llm`) e mantido como
alternativa/fallback.

Recebe o caminho de uma imagem e uma descricao em linguagem natural de um
objeto contido nela (ex.: `"layout do card da tela inicial"`) e devolve o
recorte da imagem correspondente a esse objeto, com metricas de custo/uso
da LLM e de compute local.

## Como funciona

1. A imagem e fatiada em tiles com sobreposicao se exceder `1568px` no
   maior lado (imagens menores viram um unico tile).
2. A descricao do usuario e reformulada por uma LLM como uma query de
   busca mais clara, sem olhar a imagem nem supor detalhes visuais que o
   usuario nao mencionou.
3. **Motor local (padrao, `--motor local`):** a descricao e traduzida
   para ingles e cada tile e submetido ao detector local configurado
   (`--backend-local florence2` ou `grounding_dino`), que devolve
   candidatos com score. Os candidatos de todos os tiles sao ranqueados
   globalmente por score (sem chamada LLM) e, na ordem do ranking, cada
   um e recortado do original em resolucao plena e submetido a um
   revisor LLM (unica chamada multimodal do fluxo local). Como o
   detector local nao aceita feedback textual, uma reprovacao apenas
   avanca para o proximo candidato do ranking -- nao ha refino iterativo
   dentro do motor local. Se nenhum candidato for aprovado e
   `--fallback-para-llm` estiver ativo (padrao), o fluxo cai
   automaticamente para o motor LLM abaixo.
4. **Motor LLM (`--motor llm`, ou fallback do motor local):** cada tile e
   descrito por uma LLM (em paralelo) e depois ranqueado pela
   probabilidade de conter o objeto-alvo; para os melhores candidatos, um
   laco localiza o objeto no tile (coordenadas normalizadas 0..1000),
   recorta do original e submete o recorte a um revisor. Reprovacoes
   geram feedback acionavel usado para refinar a proxima tentativa, ate
   um numero maximo de tentativas por candidato.
5. O primeiro recorte aprovado e promovido para o diretorio de saida,
   junto com um `.json` com o resultado e as metricas de uso da LLM e de
   compute local (chamadas, tokens, custo, tempo, dispositivo).

No motor local, o detector devolve caixas ja em pixels absolutos do tile
recebido (sem normalizacao 0..1000); no motor LLM, a LLM devolve
coordenadas normalizadas no espaco `0..1000` relativas ao tile enviado.
Em ambos os casos a conversao para pixels do arquivo original e feita
inteiramente em codigo Python (`imagem.py`), nunca pelo modelo.

## Instalacao

Requer Python 3.12+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# edite .env e defina ANTHROPIC_API_KEY
```

O motor local baixa os pesos do backend escolhido (Florence-2 ou
Grounding DINO) do Hugging Face Hub no primeiro uso, em
`~/.cache/huggingface` (ou em `DETECTOR_CACHE_DIR_MODELOS`, se definido).
Isso requer acesso a `huggingface.co`; em ambientes sem esse acesso,
pre-baixe os pesos e aponte `HF_HOME`, ou use `--motor llm`.

## Uso

```bash
uv run detector \
  --imagem ./telas/home.png \
  --descricao "layout do card da tela inicial" \
  --dir-saida ./saida \
  --max-tentativas 3 \
  --json
```

Codigos de saida: `0` sucesso, `1` objeto nao detectado, `2` erro de
entrada (arquivo/formato), `3` erro de configuracao (ex.: chave de API
ausente), `4` erro de rede/provedor apos esgotar retries.

Use `--debug` para salvar todas as imagens intermediarias do processo
(tiles, versoes reduzidas enviadas a LLM e recortes candidatos) em
`<dir-saida>/debug/<run_id>/` -- os caminhos aparecem em
`caminho_debug` e `imagens_intermediarias` no `ResultadoDeteccao`.

Use `--motor llm` para desativar o detector local e usar apenas a LLM
(comportamento identico ao v1), ou `--backend-local grounding_dino` para
trocar o backend local sem sair do motor padrao.

Veja todas as opcoes com `uv run detector --help`.

## Uso como biblioteca

```python
from detector.config import Configuracao
from detector.pipeline import detecta
from pathlib import Path

resultado = detecta(Path("home.png"), "o card azul do topo", Configuracao())
print(resultado.sucesso, resultado.caminho_imagem, resultado.metricas.custo_usd)
```

## Desenvolvimento

```bash
uv run pytest                 # testes rapidos (sem chamadas de rede nem modelo local)
uv run pytest -m integracao   # testes de integracao, fazem chamadas reais a LLM
uv run pytest -m modelo_local # testes que baixam/rodam um backend local real (pesado)
uv run ruff check .
uv run mypy src/
```

## Estrutura

```
src/detector/
├── cli.py                    # interface Typer
├── config.py                 # Configuracao (pydantic-settings)
├── modelos.py                 # modelos Pydantic
├── telemetria.py               # acumulador de metricas, thread-safe
├── llm.py                       # unico ponto de chamada ao litellm
├── imagem.py                    # tiling, recorte, conversao de coordenadas
├── prompts.py                   # templates de prompt
├── pipeline.py                  # orquestracao das etapas, selecao de motor
├── deteccao_local/               # detector open-vocabulary local (motor padrao)
│   ├── base.py                    # interface DetectorLocal (ABC)
│   ├── florence2.py                # backend Florence-2
│   ├── grounding_dino.py           # backend Grounding DINO
│   └── tradutor.py                 # traducao PT->EN do prompt de deteccao
└── etapas/                        # cada etapa do pipeline
    ├── melhora_descricao.py
    ├── encontra_objeto_local.py    # motor local (padrao)
    ├── encontra_objeto_llm.py      # motor LLM (--motor llm, ou fallback)
    ├── descreve_imagem.py          # usado apenas pelo motor LLM
    ├── escolhe_imagem.py           # usado apenas pelo motor LLM
    └── julga_resultado.py          # revisor final, usado por ambos os motores
```
