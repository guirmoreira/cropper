"""Etapa 3: transforma a descricao vaga do usuario em uma descricao verificavel."""

from __future__ import annotations

import logging
from pathlib import Path

from detector.config import Configuracao
from detector.llm import bloco_imagem, chamar_llm
from detector.modelos import DescricaoMelhorada
from detector.prompts import SISTEMA_MELHORA_DESCRICAO
from detector.telemetria import Telemetria

logger = logging.getLogger(__name__)


def melhora_descricao(
    descricao: str,
    caminho_imagem_reduzida: Path | None,
    config: Configuracao,
    telemetria: Telemetria,
) -> DescricaoMelhorada:
    conteudo: list[dict] = [
        {"type": "text", "text": f'Descricao original do usuario: "{descricao}"'}
    ]

    if caminho_imagem_reduzida is not None:
        try:
            conteudo.append(bloco_imagem(caminho_imagem_reduzida))
        except OSError:
            logger.warning(
                "Falha ao ler imagem reduzida em %s; prosseguindo apenas com texto.",
                caminho_imagem_reduzida,
                exc_info=True,
            )

    return chamar_llm(
        etapa="melhora_descricao",
        modelo=config.modelo_texto,
        sistema=SISTEMA_MELHORA_DESCRICAO,
        conteudo=conteudo,
        schema=DescricaoMelhorada,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )
