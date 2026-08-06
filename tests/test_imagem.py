"""Testes de tiling e conversao de coordenadas -- sem nenhuma chamada a LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from PIL import Image
from pydantic import ValidationError

from detector.config import Configuracao
from detector.imagem import (
    caixa_plausivel,
    carrega_imagem,
    gera_tiles,
    normalizado_para_original,
)
from detector.modelos import Retangulo, RetanguloNormalizado, Tile


def _cobertura_completa(tiles: list[Tile], largura: int, altura: int) -> bool:
    """Verifica que a uniao dos retangulos dos tiles cobre 100% da area original."""
    mascara = Image.new("L", (largura, altura), 0)
    for tile in tiles:
        bloco = Image.new("L", tile.tamanho, 255)
        mascara.paste(bloco, tile.origem)
    minimo, _maximo = mascara.getextrema()
    return minimo == 255


@pytest.mark.parametrize(
    "largura,altura",
    [
        (3000, 3000),  # proporcao 1:1
        (3600, 1200),  # proporcao 3:1
        (1200, 4800),  # proporcao 1:4
    ],
)
def test_tiles_cobrem_area_total(
    largura: int, altura: int, criar_imagem: Callable[[int, int, str], Path], tmp_path: Path
) -> None:
    config = Configuracao(dir_saida=tmp_path / "saida")
    caminho = criar_imagem(largura, altura, "gray")
    imagem = carrega_imagem(caminho)

    tiles = gera_tiles(imagem, config, tmp_path / "run")

    assert _cobertura_completa(tiles, largura, altura)


def test_tile_unico_abaixo_do_limiar(
    criar_imagem: Callable[[int, int, str], Path], config_padrao: Configuracao, tmp_path: Path
) -> None:
    caminho = criar_imagem(800, 600, "blue")
    imagem = carrega_imagem(caminho)

    tiles = gera_tiles(imagem, config_padrao, tmp_path / "run")

    assert len(tiles) == 1
    assert tiles[0].origem == (0, 0)
    assert tiles[0].tamanho == (800, 600)


def test_sobreposicao_efetiva(
    criar_imagem: Callable[[int, int, str], Path], tmp_path: Path
) -> None:
    config = Configuracao(tamanho_tile=1400, sobreposicao=0.20, limiar_dimensao=1568)
    caminho = criar_imagem(3600, 1400, "green")
    imagem = carrega_imagem(caminho)

    tiles = gera_tiles(imagem, config, tmp_path / "run")
    tiles_linha0 = sorted((t for t in tiles if t.linha == 0), key=lambda t: t.coluna)
    assert len(tiles_linha0) >= 2

    overlap_minimo_esperado = config.tamanho_tile - int(
        config.tamanho_tile * (1 - config.sobreposicao)
    )
    for anterior, atual in zip(tiles_linha0, tiles_linha0[1:]):
        fim_anterior = anterior.origem[0] + anterior.tamanho[0]
        overlap = fim_anterior - atual.origem[0]
        assert overlap >= overlap_minimo_esperado


def test_conversao_coordenadas(tmp_path: Path) -> None:
    tile = Tile(
        indice=0,
        caminho=tmp_path / "tile_000.png",
        origem=(200, 300),
        tamanho=(400, 400),
        linha=0,
        coluna=0,
    )
    caixa = RetanguloNormalizado(x0=0, y0=0, x1=1000, y1=1000)

    resultado = normalizado_para_original(caixa, tile)

    assert resultado == Retangulo(x0=200, y0=300, x1=600, y1=700)


def test_caixa_implausivel_rejeitada(tmp_path: Path) -> None:
    tile = Tile(
        indice=0,
        caminho=tmp_path / "tile_000.png",
        origem=(0, 0),
        tamanho=(400, 400),
        linha=0,
        coluna=0,
    )
    caixa_quase_toda = Retangulo(x0=0, y0=0, x1=400, y1=396)  # 99% da area do tile

    assert caixa_plausivel(caixa_quase_toda, tile) is False


def test_caixa_pequena_e_implausivel(tmp_path: Path) -> None:
    tile = Tile(
        indice=0,
        caminho=tmp_path / "tile_000.png",
        origem=(0, 0),
        tamanho=(400, 400),
        linha=0,
        coluna=0,
    )
    caixa_minuscula = Retangulo(x0=0, y0=0, x1=4, y1=4)

    assert caixa_plausivel(caixa_minuscula, tile) is False


def test_retangulo_invalido() -> None:
    with pytest.raises(ValidationError):
        Retangulo(x0=10, y0=0, x1=10, y1=5)
