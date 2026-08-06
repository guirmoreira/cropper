"""Etapa 4: descreve o conteudo de cada tile, em paralelo."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from detector.config import Configuracao
from detector.imagem import prepara_para_llm
from detector.llm import bloco_imagem, chamar_llm
from detector.modelos import DescricaoTile, Tile
from detector.prompts import SISTEMA_DESCREVE_IMAGEM
from detector.telemetria import Telemetria

logger = logging.getLogger(__name__)

RESUMO_FALHA = "[falha na descricao]"


def _descreve_tile(tile: Tile, config: Configuracao, telemetria: Telemetria) -> DescricaoTile:
    caminho_llm = prepara_para_llm(tile.caminho, config.limiar_dimensao)
    conteudo = [
        {"type": "text", "text": f"Tile numero {tile.indice}."},
        bloco_imagem(caminho_llm),
    ]
    return chamar_llm(
        etapa="descreve_imagem",
        modelo=config.modelo_visao,
        sistema=SISTEMA_DESCREVE_IMAGEM,
        conteudo=conteudo,
        schema=DescricaoTile,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )


def descreve_tiles(
    tiles: list[Tile], config: Configuracao, telemetria: Telemetria
) -> list[DescricaoTile]:
    """Descreve cada tile em paralelo. Falha individual nao aborta o lote."""
    resultados: list[DescricaoTile | None] = [None] * len(tiles)

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futuros = {
            executor.submit(_descreve_tile, tile, config, telemetria): tile for tile in tiles
        }
        for futuro in as_completed(futuros):
            tile = futuros[futuro]
            try:
                resultados[tile.indice] = futuro.result()
            except Exception:
                logger.exception("Falha ao descrever o tile %s", tile.indice)
                resultados[tile.indice] = DescricaoTile(indice_tile=tile.indice, resumo=RESUMO_FALHA)

    return [r for r in resultados if r is not None]
