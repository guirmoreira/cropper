"""Manipulacao de imagem: carga, tiling, recorte e conversao de coordenadas.

Ponto critico do pipeline: a LLM sempre devolve coordenadas normalizadas no
espaco 0..1000, relativas ao tile que ela recebeu. A conversao para pixels do
arquivo original e feita inteiramente aqui, nunca pela LLM.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from detector.config import Configuracao
from detector.excecoes import ErroEntrada
from detector.modelos import Retangulo, RetanguloNormalizado, Tamanho, Tile

FORMATOS_SUPORTADOS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}


def carrega_imagem(caminho: Path) -> Image.Image:
    """Carrega, normaliza orientacao EXIF e converte para RGB sobre fundo branco."""
    if not caminho.exists():
        raise ErroEntrada(f"Arquivo nao encontrado: {caminho}")

    try:
        imagem: Image.Image = Image.open(caminho)
        imagem.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ErroEntrada(f"Nao foi possivel abrir a imagem: {caminho} ({exc})") from exc

    if imagem.format not in FORMATOS_SUPORTADOS:
        raise ErroEntrada(f"Formato de imagem nao suportado: {imagem.format} ({caminho})")

    imagem = ImageOps.exif_transpose(imagem)

    if imagem.mode in ("RGBA", "LA", "P"):
        rgba = imagem.convert("RGBA")
        fundo = Image.new("RGB", imagem.size, (255, 255, 255))
        fundo.paste(rgba, mask=rgba.split()[-1])
        imagem = fundo
    else:
        imagem = imagem.convert("RGB")

    return imagem


def _posicoes_eixo(dimensao: int, tamanho_tile: int, passo: int) -> tuple[list[int], int]:
    """Posicoes de origem ao longo de um eixo e o tamanho do tile nesse eixo."""
    if dimensao <= tamanho_tile:
        return [0], dimensao

    n = max(1, math.ceil((dimensao - tamanho_tile) / passo) + 1)
    posicoes: list[int] = []
    for i in range(n):
        pos = min(i * passo, dimensao - tamanho_tile)
        if not posicoes or pos != posicoes[-1]:
            posicoes.append(pos)
    return posicoes, tamanho_tile


def gera_tiles(imagem: Image.Image, config: Configuracao, dir_temp: Path) -> list[Tile]:
    """Fatia a imagem em tiles com sobreposicao, ou devolve um unico tile.

    Se a imagem for menor ou igual a `config.limiar_dimensao` no maior lado,
    devolve uma lista com um unico Tile cobrindo a imagem inteira. O restante
    do pipeline nao distingue esse caso: e sempre uma lista de tiles.
    """
    largura, altura = imagem.size
    dir_tiles = dir_temp / "tiles"
    dir_tiles.mkdir(parents=True, exist_ok=True)

    if max(largura, altura) <= config.limiar_dimensao:
        caminho = dir_tiles / "tile_000.png"
        imagem.save(caminho)
        return [
            Tile(
                indice=0,
                caminho=caminho,
                origem=(0, 0),
                tamanho=(largura, altura),
                linha=0,
                coluna=0,
            )
        ]

    passo = max(1, int(config.tamanho_tile * (1 - config.sobreposicao)))
    xs, largura_tile = _posicoes_eixo(largura, config.tamanho_tile, passo)
    ys, altura_tile = _posicoes_eixo(altura, config.tamanho_tile, passo)

    tiles: list[Tile] = []
    indice = 0
    for linha, y in enumerate(ys):
        for coluna, x in enumerate(xs):
            caixa = (x, y, x + largura_tile, y + altura_tile)
            recorte = imagem.crop(caixa)
            caminho = dir_tiles / f"tile_{indice:03d}.png"
            recorte.save(caminho)
            tiles.append(
                Tile(
                    indice=indice,
                    caminho=caminho,
                    origem=(x, y),
                    tamanho=(largura_tile, altura_tile),
                    linha=linha,
                    coluna=coluna,
                )
            )
            indice += 1
    return tiles


def prepara_para_llm(caminho: Path, limite: int = 1568) -> Path:
    """Redimensiona a imagem se exceder `limite` no maior lado, para reduzir tokens.

    As coordenadas sao normalizadas (0..1000), entao nenhuma correcao adicional
    e necessaria ao usar essa versao reduzida para a chamada a LLM.
    """
    imagem = Image.open(caminho)
    largura, altura = imagem.size
    maior_lado = max(largura, altura)
    if maior_lado <= limite:
        return caminho

    fator = limite / maior_lado
    nova_largura = round(largura * fator)
    nova_altura = round(altura * fator)
    redimensionada = imagem.convert("RGB").resize(
        (nova_largura, nova_altura), Image.Resampling.LANCZOS
    )
    destino = caminho.with_name(f"{caminho.stem}_llm.png")
    redimensionada.save(destino)
    return destino


def normalizado_para_original(caixa: RetanguloNormalizado, tile: Tile) -> Retangulo:
    """0..1000 relativo ao tile -> pixels absolutos do ORIGINAL."""
    largura_tile, altura_tile = tile.tamanho
    x0 = round(caixa.x0 / 1000 * largura_tile) + tile.origem[0]
    y0 = round(caixa.y0 / 1000 * altura_tile) + tile.origem[1]
    x1 = round(caixa.x1 / 1000 * largura_tile) + tile.origem[0]
    y1 = round(caixa.y1 / 1000 * altura_tile) + tile.origem[1]
    return Retangulo(x0=x0, y0=y0, x1=x1, y1=y1)


def caixa_plausivel(
    retangulo: Retangulo,
    tile: Tile,
    largura_min: int = 8,
    altura_min: int = 8,
    fracao_max_area: float = 0.98,
) -> bool:
    """Rejeita caixas degeneradas ou que cobrem quase todo o tile (modelo "desistiu")."""
    if retangulo.largura < largura_min or retangulo.altura < altura_min:
        return False
    area_tile = tile.tamanho[0] * tile.tamanho[1]
    return not retangulo.area > fracao_max_area * area_tile


def recorta_e_salva(imagem: Image.Image, retangulo: Retangulo, destino: Path) -> Path:
    """Recorta do original em resolucao plena e salva em PNG."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    recorte = imagem.crop((retangulo.x0, retangulo.y0, retangulo.x1, retangulo.y1))
    recorte.save(destino)
    return destino


def tamanho_da_imagem(imagem: Image.Image) -> Tamanho:
    largura, altura = imagem.size
    return Tamanho(largura=largura, altura=altura)
