"""Pipeline end-to-end com a LLM mockada (nenhuma chamada de rede)."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from detector import pipeline
from detector.config import Configuracao
from detector.etapas import (
    descreve_imagem,
    encontra_objeto_llm,
    escolhe_imagem,
    julga_resultado,
    melhora_descricao,
)
from detector.modelos import (
    DescricaoMelhorada,
    DescricaoTile,
    ItemRanking,
    Ranking,
    ResultadoLocalizacao,
    RetanguloNormalizado,
    Veredito,
)

MODULOS_COM_CHAMAR_LLM = [
    melhora_descricao,
    descreve_imagem,
    escolhe_imagem,
    encontra_objeto_llm,
    julga_resultado,
]


def _indice_do_conteudo(conteudo: list[dict[str, Any]]) -> int:
    texto = conteudo[0]["text"]
    m = re.search(r"Tile numero (\d+)", texto)
    assert m is not None, f"conteudo sem indice de tile: {texto!r}"
    return int(m.group(1))


@pytest.fixture
def fila_respostas(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fila de respostas mockadas por etapa: lista (FIFO) ou callable(conteudo=...)."""
    fila: dict[str, Any] = {}
    chamadas: dict[str, list[list[dict[str, Any]]]] = {}
    lock = threading.Lock()

    def fake_chamar_llm(*, etapa: str, telemetria: Any, conteudo: Any = None, **kwargs: Any) -> Any:
        with lock:
            chamadas.setdefault(etapa, []).append(conteudo)
            fonte = fila.get(etapa)
            if fonte is None:
                raise AssertionError(f"Nenhuma resposta mockada para a etapa '{etapa}'")
            if callable(fonte):
                resposta = fonte(conteudo=conteudo)
            else:
                if not fonte:
                    raise AssertionError(f"Fila de respostas vazia para a etapa '{etapa}'")
                resposta = fonte.pop(0)

        telemetria.registra(
            etapa=etapa, tokens_entrada=100, tokens_saida=20, custo_usd=0.0005, duracao_s=0.01
        )
        if isinstance(resposta, Exception):
            raise resposta
        return resposta

    for modulo in MODULOS_COM_CHAMAR_LLM:
        monkeypatch.setattr(modulo, "chamar_llm", fake_chamar_llm)

    fila["_chamadas"] = chamadas  # exposto para inspecao nos testes
    return fila


def _imagem_pequena(tmp_path: Path, largura: int = 800, altura: int = 600) -> Path:
    caminho = tmp_path / "entrada.png"
    Image.new("RGB", (largura, altura), color="white").save(caminho)
    return caminho


def _config(tmp_path: Path, **overrides: Any) -> Configuracao:
    """Configuracao padrao destes testes: motor_localizacao='llm' (fluxo v1, exercitado aqui)."""
    overrides.setdefault("motor_localizacao", "llm")
    return Configuracao(dir_saida=tmp_path / "saida", **overrides)


def test_sucesso_primeira_tentativa_tile_unico(
    tmp_path: Path, fila_respostas: dict[str, Any]
) -> None:
    caminho = _imagem_pequena(tmp_path)
    config = _config(tmp_path)

    fila_respostas["melhora_descricao"] = [
        DescricaoMelhorada(descricao_melhorada="um card azul", criterios_exclusao=[], termos_chave=[])
    ]
    fila_respostas["encontra_objeto"] = [
        ResultadoLocalizacao(
            encontrado=True,
            caixa=RetanguloNormalizado(x0=100, y0=100, x1=700, y1=500),
            confianca=0.9,
        )
    ]
    fila_respostas["julga_resultado"] = [Veredito(aprovado=True, confianca=0.95, feedback="ok")]

    resultado = pipeline.detecta(caminho, "o card azul", config)

    assert resultado.sucesso is True
    assert resultado.tentativas_refino == 1
    assert resultado.tiles_avaliados == 1
    assert resultado.caminho_imagem is not None
    assert resultado.caminho_imagem.exists()
    assert resultado.metricas.chamadas_llm == 3


def test_aprovacao_na_terceira_tentativa(tmp_path: Path, fila_respostas: dict[str, Any]) -> None:
    caminho = _imagem_pequena(tmp_path)
    config = _config(tmp_path)

    caixa = RetanguloNormalizado(x0=100, y0=100, x1=700, y1=500)
    fila_respostas["melhora_descricao"] = [
        DescricaoMelhorada(descricao_melhorada="um card azul", criterios_exclusao=[], termos_chave=[])
    ]
    fila_respostas["encontra_objeto"] = [
        ResultadoLocalizacao(encontrado=True, caixa=caixa, confianca=0.7) for _ in range(3)
    ]
    fila_respostas["julga_resultado"] = [
        Veredito(aprovado=False, confianca=0.4, feedback="cortado no topo"),
        Veredito(aprovado=False, confianca=0.6, feedback="ainda cortado"),
        Veredito(aprovado=True, confianca=0.9, feedback="ok"),
    ]

    resultado = pipeline.detecta(caminho, "o card azul", config)

    assert resultado.sucesso is True
    assert resultado.tentativas_refino == 3

    chamadas_encontra = fila_respostas["_chamadas"]["encontra_objeto"]
    assert len(chamadas_encontra) == 3
    assert "cortado no topo" in chamadas_encontra[1][0]["text"]
    assert "ainda cortado" in chamadas_encontra[2][0]["text"]


def test_primeiro_tile_sem_objeto_segundo_com_objeto(
    tmp_path: Path, fila_respostas: dict[str, Any]
) -> None:
    caminho = _imagem_pequena(tmp_path, largura=2000, altura=1000)
    config = _config(tmp_path)

    fila_respostas["melhora_descricao"] = [
        DescricaoMelhorada(descricao_melhorada="o logo", criterios_exclusao=[], termos_chave=[])
    ]

    def _descreve(conteudo: list[dict[str, Any]]) -> DescricaoTile:
        indice = _indice_do_conteudo(conteudo)
        return DescricaoTile(indice_tile=indice, resumo=f"tile {indice}", elementos=[])

    fila_respostas["descreve_imagem"] = _descreve
    fila_respostas["escolhe_imagem"] = [
        Ranking(
            itens=[
                ItemRanking(indice_tile=0, pontuacao=0.9, justificativa="parece ter o logo"),
                ItemRanking(indice_tile=1, pontuacao=0.5, justificativa="talvez"),
            ]
        )
    ]
    fila_respostas["encontra_objeto"] = [
        ResultadoLocalizacao(encontrado=False, observacao="nao esta neste tile"),
        ResultadoLocalizacao(
            encontrado=True,
            caixa=RetanguloNormalizado(x0=100, y0=100, x1=700, y1=500),
            confianca=0.9,
        ),
    ]
    fila_respostas["julga_resultado"] = [Veredito(aprovado=True, confianca=0.9, feedback="ok")]

    resultado = pipeline.detecta(caminho, "o logo", config)

    assert resultado.sucesso is True
    assert resultado.tiles_avaliados == 2
    assert resultado.tentativas_refino == 2


def test_falha_total_apos_esgotar_tentativas(
    tmp_path: Path, fila_respostas: dict[str, Any]
) -> None:
    caminho = _imagem_pequena(tmp_path)
    config = _config(tmp_path, max_tentativas_refino=3)

    caixa = RetanguloNormalizado(x0=100, y0=100, x1=700, y1=500)
    fila_respostas["melhora_descricao"] = [
        DescricaoMelhorada(descricao_melhorada="algo que nao existe", criterios_exclusao=[], termos_chave=[])
    ]
    fila_respostas["encontra_objeto"] = [
        ResultadoLocalizacao(encontrado=True, caixa=caixa, confianca=0.5) for _ in range(3)
    ]
    fila_respostas["julga_resultado"] = [
        Veredito(aprovado=False, confianca=0.2, feedback="nao e o objeto pedido") for _ in range(3)
    ]

    resultado = pipeline.detecta(caminho, "algo que nao existe", config)

    assert resultado.sucesso is False
    assert resultado.caminho_imagem is None
    assert resultado.tentativas_refino == 3
    assert "confianca suficiente" in resultado.mensagem


def test_descricao_vazia_levanta_erro_entrada(tmp_path: Path, fila_respostas: dict[str, Any]) -> None:
    from detector.excecoes import ErroEntrada

    caminho = _imagem_pequena(tmp_path)
    config = _config(tmp_path)

    with pytest.raises(ErroEntrada) as exc_info:
        pipeline.detecta(caminho, "ab", config)

    assert exc_info.value.metricas is not None


def test_dir_temporario_removido_apos_execucao(
    tmp_path: Path, fila_respostas: dict[str, Any]
) -> None:
    import tempfile

    caminho = _imagem_pequena(tmp_path)
    config = _config(tmp_path)

    fila_respostas["melhora_descricao"] = [
        DescricaoMelhorada(descricao_melhorada="um card azul", criterios_exclusao=[], termos_chave=[])
    ]
    fila_respostas["encontra_objeto"] = [
        ResultadoLocalizacao(
            encontrado=True,
            caixa=RetanguloNormalizado(x0=100, y0=100, x1=700, y1=500),
            confianca=0.9,
        )
    ]
    fila_respostas["julga_resultado"] = [Veredito(aprovado=True, confianca=0.95, feedback="ok")]

    resultado = pipeline.detecta(caminho, "o card azul", config)

    restantes = list(Path(tempfile.gettempdir()).glob(f"detector-{resultado.run_id}-*"))
    assert restantes == []
