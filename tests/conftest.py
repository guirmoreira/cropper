"""Fixtures compartilhadas entre os testes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from detector.config import Configuracao


@pytest.fixture
def config_padrao() -> Configuracao:
    return Configuracao()


@pytest.fixture
def criar_imagem(tmp_path: Path) -> Callable[[int, int, str], Path]:
    """Fabrica uma imagem RGB solida de tamanho arbitrario e devolve o caminho."""

    contador = {"n": 0}

    def _criar(largura: int, altura: int, cor: str = "gray") -> Path:
        contador["n"] += 1
        caminho = tmp_path / f"imagem_{contador['n']}.png"
        Image.new("RGB", (largura, altura), color=cor).save(caminho)
        return caminho

    return _criar
