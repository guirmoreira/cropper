"""Traducao PT->EN da descricao melhorada, usada como prompt do detector local.

Grounding DINO/Florence-2 sao majoritariamente treinados em ingles; sem esta traducao,
a qualidade da deteccao cai para descricoes em portugues.
"""

from __future__ import annotations

from detector.config import Configuracao
from detector.llm import chamar_llm
from detector.modelos import Traducao
from detector.prompts import SISTEMA_TRADUCAO
from detector.telemetria import Telemetria


def traduz_para_ingles(texto_pt: str, config: Configuracao, telemetria: Telemetria) -> str:
    """Traduz via LLM (chamada de texto puro, barata) ou devolve o texto original.

    `config.traduzir_prompt=False` desativa a traducao. `config.modelo_traducao="local"` e um
    hook documentado para um tradutor offline (ex.: argostranslate) -- nao implementado nesta
    fase; devolve o texto original sem custo de API.
    """
    if not config.traduzir_prompt:
        return texto_pt
    if config.modelo_traducao == "local":
        return texto_pt

    resultado = chamar_llm(
        etapa="traduz_para_ingles",
        modelo=config.modelo_traducao,
        sistema=SISTEMA_TRADUCAO,
        conteudo=[{"type": "text", "text": texto_pt}],
        schema=Traducao,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )
    return resultado.texto_ingles
