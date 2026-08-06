"""Teste de integracao: chamada real a LLM. Nao roda no conjunto padrao.

Execute com: uv run pytest -m integracao
Requer ANTHROPIC_API_KEY definido no ambiente.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from detector.config import Configuracao
from detector.modelos import Retangulo
from detector.pipeline import detecta

pytestmark = pytest.mark.integracao

CAIXA_ESPERADA = Retangulo(x0=520, y0=180, x1=920, y1=480)


def _imagem_com_retangulo_vermelho(caminho: Path) -> None:
    imagem = Image.new("RGB", (1200, 800), color="white")
    desenho = ImageDraw.Draw(imagem)
    desenho.rectangle(
        (CAIXA_ESPERADA.x0, CAIXA_ESPERADA.y0, CAIXA_ESPERADA.x1, CAIXA_ESPERADA.y1),
        fill="red",
    )
    imagem.save(caminho)


def _iou(a: Retangulo, b: Retangulo) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    intersecao = max(0, x1 - x0) * max(0, y1 - y0)
    uniao = a.area + b.area - intersecao
    return intersecao / uniao if uniao else 0.0


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY nao definido"
)
def test_localiza_retangulo_vermelho_com_iou_minima(tmp_path: Path) -> None:
    caminho = tmp_path / "retangulo.png"
    _imagem_com_retangulo_vermelho(caminho)
    config = Configuracao(dir_saida=tmp_path / "saida")

    resultado = detecta(caminho, "o retangulo vermelho solido sobre fundo branco", config)

    assert resultado.sucesso is True
    assert resultado.caixa_no_original is not None
    assert _iou(resultado.caixa_no_original, CAIXA_ESPERADA) >= 0.7
