# detector-objetos

Detector de objetos em imagem via LLM multimodal (Claude Haiku 4.5, via LiteLLM).

Recebe o caminho de uma imagem e uma descricao em linguagem natural de um
objeto contido nela (ex.: `"layout do card da tela inicial"`) e devolve o
recorte da imagem correspondente a esse objeto, com metricas de custo e
uso da LLM.

## Como funciona

1. A imagem e fatiada em tiles com sobreposicao se exceder `1568px` no
   maior lado (imagens menores viram um unico tile).
2. A descricao do usuario e refinada por uma LLM em uma descricao mais
   precisa e verificavel.
3. Cada tile e descrito por uma LLM (em paralelo) e depois ranqueado pela
   probabilidade de conter o objeto-alvo.
4. Para os melhores candidatos do ranking, um laco localiza o objeto no
   tile (coordenadas normalizadas 0..1000), recorta do original em
   resolucao plena e submete o recorte a um revisor. Reprovacoes geram
   feedback acionavel usado para refinar a proxima tentativa, ate um
   numero maximo de tentativas por candidato.
5. O primeiro recorte aprovado e promovido para o diretorio de saida,
   junto com um `.json` com o resultado e as metricas de uso da LLM
   (chamadas, tokens, custo, tempo).

Todas as coordenadas retornadas pela LLM sao normalizadas no espaco
`0..1000` relativas ao tile enviado -- a conversao para pixels do
arquivo original e feita inteiramente em codigo Python (`imagem.py`),
nunca pela LLM.

## Instalacao

Requer Python 3.12+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
# edite .env e defina ANTHROPIC_API_KEY
```

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
uv run pytest              # testes rapidos (sem chamadas de rede)
uv run pytest -m integracao  # testes de integracao, fazem chamadas reais a LLM
uv run ruff check .
uv run mypy src/
```

## Estrutura

```
src/detector/
├── cli.py              # interface Typer
├── config.py           # Configuracao (pydantic-settings)
├── modelos.py           # modelos Pydantic
├── telemetria.py        # acumulador de metricas, thread-safe
├── llm.py                # unico ponto de chamada ao litellm
├── imagem.py             # tiling, recorte, conversao de coordenadas
├── prompts.py            # templates de prompt
├── pipeline.py           # orquestracao das etapas
└── etapas/                # cada etapa do pipeline
```
