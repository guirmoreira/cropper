"""Orquestracao das etapas do pipeline de deteccao."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from detector.config import Configuracao
from detector.etapas.descreve_imagem import RESUMO_FALHA, descreve_tiles
from detector.etapas.encontra_objeto import encontra_objeto
from detector.etapas.escolhe_imagem import escolher_imagem
from detector.etapas.julga_resultado import julga_resultado
from detector.etapas.melhora_descricao import melhora_descricao
from detector.excecoes import DetectorErro, ErroEntrada, ErroProvedor
from detector.imagem import (
    caixa_plausivel,
    carrega_imagem,
    gera_tiles,
    normalizado_para_original,
    recorta_e_salva,
    tamanho_da_imagem,
)
from detector.modelos import (
    DescricaoMelhorada,
    ItemRanking,
    Ranking,
    ResultadoDeteccao,
    Retangulo,
    RetanguloNormalizado,
    Tamanho,
    Tile,
    Veredito,
)
from detector.telemetria import Telemetria

logger = logging.getLogger(__name__)


@dataclass
class _CandidatoRefino:
    caminho: Path
    retangulo: Retangulo
    veredito: Veredito


def _slugifica(texto: str, tamanho_max: int = 40) -> str:
    normalizado = re.sub(r"[^a-z0-9]+", "-", texto.lower()).strip("-")
    return normalizado[:tamanho_max] or "objeto"


def _promove_arquivo(
    caminho_original: Path, caminho_temp: Path, descricao: str, config: Configuracao, run_id: str
) -> Path:
    config.dir_saida.mkdir(parents=True, exist_ok=True)
    slug = _slugifica(descricao)
    destino = config.dir_saida / f"{caminho_original.stem}_{slug}_{run_id}.png"
    shutil.copy2(caminho_temp, destino)
    return destino


def _grava_json_resultado(resultado: ResultadoDeteccao) -> None:
    if resultado.caminho_imagem is None:
        return
    caminho_json = resultado.caminho_imagem.with_suffix(".json")
    caminho_json.write_text(resultado.model_dump_json(indent=2), encoding="utf-8")


def _copia_imagens_debug(
    dir_temp: Path, config: Configuracao, run_id: str
) -> tuple[Path, list[Path]]:
    """Copia todas as imagens intermediarias do diretorio temporario para dir_saida/debug.

    Inclui tiles, versoes reduzidas preparadas para a LLM e recortes candidatos.
    """
    dir_debug = config.dir_saida / "debug" / run_id
    dir_debug.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dir_temp, dir_debug, dirs_exist_ok=True)
    imagens = sorted(dir_debug.rglob("*.png"))
    return dir_debug, imagens


def _executa_refino(
    *,
    imagem: Image.Image,
    tiles: list[Tile],
    ranking: Ranking,
    descricao_melhorada: DescricaoMelhorada,
    config: Configuracao,
    telemetria: Telemetria,
    dir_temp: Path,
    tamanho_original: Tamanho,
) -> tuple[_CandidatoRefino | None, int, int]:
    """Executa o laco de localizacao/recorte/julgamento sobre o ranking de tiles.

    Devolve (candidato final ou None, tentativas_totais, tiles_avaliados).
    """
    tiles_por_indice = {tile.indice: tile for tile in tiles}
    tentativas_totais = 0
    tiles_avaliados = 0
    melhor_esforco: _CandidatoRefino | None = None

    for item in ranking.itens:
        tile = tiles_por_indice[item.indice_tile]
        tiles_avaliados += 1
        feedback: str | None = None
        caixa_anterior: RetanguloNormalizado | None = None

        for tentativa in range(1, config.max_tentativas_refino + 1):
            tentativas_totais += 1

            loc = encontra_objeto(
                tile=tile,
                descricao=descricao_melhorada,
                feedback=feedback,
                caixa_anterior=caixa_anterior,
                config=config,
                telemetria=telemetria,
            )
            if not loc.encontrado or loc.caixa is None:
                logger.info(
                    "Tile %d: objeto nao encontrado (%s)", tile.indice, loc.observacao
                )
                break

            ret_original = normalizado_para_original(loc.caixa, tile)
            ret_original = ret_original.expandir(config.margem_recorte, tamanho_original)
            if not caixa_plausivel(ret_original, tile):
                logger.info("Tile %d: caixa implausivel, descartando", tile.indice)
                break

            caminho_candidato = dir_temp / f"candidato_{item.indice_tile}_{tentativa}.png"
            recorta_e_salva(imagem, ret_original, caminho_candidato)

            veredito = julga_resultado(caminho_candidato, descricao_melhorada, config, telemetria)
            candidato = _CandidatoRefino(
                caminho=caminho_candidato, retangulo=ret_original, veredito=veredito
            )
            if melhor_esforco is None or veredito.confianca > melhor_esforco.veredito.confianca:
                melhor_esforco = candidato

            logger.info(
                "Tile %d, tentativa %d: aprovado=%s confianca=%.2f",
                tile.indice,
                tentativa,
                veredito.aprovado,
                veredito.confianca,
            )
            if veredito.aprovado:
                return candidato, tentativas_totais, tiles_avaliados

            feedback = veredito.feedback
            caixa_anterior = loc.caixa

        if config.parar_no_primeiro_candidato:
            break

    if config.devolver_melhor_esforco and melhor_esforco is not None:
        return melhor_esforco, tentativas_totais, tiles_avaliados
    return None, tentativas_totais, tiles_avaliados


def detecta(caminho_imagem: Path, descricao: str, config: Configuracao) -> ResultadoDeteccao:
    run_id = f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    dir_temp = Path(tempfile.mkdtemp(prefix=f"detector-{run_id}-"))
    telemetria = Telemetria(taxa_usd_brl=config.taxa_usd_brl)
    descricao_melhorada_texto = ""

    try:
        if len(descricao.strip()) < 3:
            raise ErroEntrada("A descricao deve ter pelo menos 3 caracteres.")

        imagem = carrega_imagem(caminho_imagem)
        tamanho_original = tamanho_da_imagem(imagem)
        logger.info(
            "Imagem carregada: %dx%d px (%s)",
            tamanho_original.largura,
            tamanho_original.altura,
            caminho_imagem,
        )

        tiles = gera_tiles(imagem, config, dir_temp)
        logger.info("Fatiamento gerou %d tile(s)", len(tiles))

        descricao_melhorada = melhora_descricao(descricao, config, telemetria)
        descricao_melhorada_texto = descricao_melhorada.descricao_melhorada
        logger.info("Descricao melhorada: %s", descricao_melhorada_texto)

        if len(tiles) == 1:
            ranking = Ranking(
                itens=[ItemRanking(indice_tile=0, pontuacao=1.0, justificativa="tile unico")]
            )
        else:
            descricoes_tiles = descreve_tiles(tiles, config, telemetria)
            if all(d.resumo == RESUMO_FALHA for d in descricoes_tiles):
                raise ErroProvedor("Falha ao descrever todos os tiles gerados a partir da imagem.")
            ranking = escolher_imagem(descricao_melhorada, descricoes_tiles, config, telemetria)
            logger.info(
                "Ranking: %s",
                [(item.indice_tile, round(item.pontuacao, 2)) for item in ranking.itens],
            )

        candidato_final, tentativas, tiles_avaliados = _executa_refino(
            imagem=imagem,
            tiles=tiles,
            ranking=ranking,
            descricao_melhorada=descricao_melhorada,
            config=config,
            telemetria=telemetria,
            dir_temp=dir_temp,
            tamanho_original=tamanho_original,
        )

        metricas = telemetria.finaliza()

        caminho_debug: Path | None = None
        imagens_debug: list[Path] = []
        if config.debug:
            caminho_debug, imagens_debug = _copia_imagens_debug(dir_temp, config, run_id)

        if candidato_final is None:
            return ResultadoDeteccao(
                sucesso=False,
                descricao_original=descricao,
                descricao_melhorada=descricao_melhorada_texto,
                tiles_avaliados=tiles_avaliados,
                tentativas_refino=tentativas,
                mensagem="Objeto nao localizado com confianca suficiente apos as tentativas configuradas.",
                metricas=metricas,
                run_id=run_id,
                caminho_debug=caminho_debug,
                imagens_intermediarias=imagens_debug,
            )

        caminho_promovido = _promove_arquivo(
            caminho_imagem, candidato_final.caminho, descricao, config, run_id
        )
        sucesso = candidato_final.veredito.aprovado
        mensagem = (
            ""
            if sucesso
            else f"Melhor esforco (nao aprovado pelo revisor): {candidato_final.veredito.feedback}"
        )

        resultado = ResultadoDeteccao(
            sucesso=sucesso,
            caminho_imagem=caminho_promovido,
            caixa_no_original=candidato_final.retangulo,
            descricao_original=descricao,
            descricao_melhorada=descricao_melhorada_texto,
            tiles_avaliados=tiles_avaliados,
            tentativas_refino=tentativas,
            confianca_final=candidato_final.veredito.confianca,
            mensagem=mensagem,
            metricas=metricas,
            run_id=run_id,
            caminho_debug=caminho_debug,
            imagens_intermediarias=imagens_debug,
        )
        _grava_json_resultado(resultado)
        return resultado

    except DetectorErro as erro:
        erro.metricas = telemetria.finaliza()
        raise
    finally:
        if not config.manter_temporarios:
            shutil.rmtree(dir_temp, ignore_errors=True)
