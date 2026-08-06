"""Testes do motor de deteccao local (§11.1 da spec v2) -- sem baixar nenhum modelo real."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from detector import pipeline
from detector.config import Configuracao
from detector.deteccao_local.base import DetectorLocal
from detector.etapas.encontra_objeto_local import encontra_objeto_local
from detector.excecoes import ErroConfiguracao
from detector.imagem import pixels_tile_para_original
from detector.modelos import CandidatoLocal, DescricaoMelhorada, Retangulo, Tamanho, Tile, Veredito
from detector.telemetria import Telemetria


class FakeDetectorLocal(DetectorLocal):
    """DetectorLocal de teste: devolve candidatos definidos por um callable, sem carregar peso algum."""

    def __init__(
        self, respostas: Callable[[Image.Image, str], list[CandidatoLocal]]
    ) -> None:
        super().__init__()
        self.dispositivo = "cpu"
        self._respostas = respostas
        self.chamadas_carregar = 0

    def carregar(self, config: Configuracao) -> None:
        self.chamadas_carregar += 1

    def detectar(self, imagem_tile: Image.Image, prompt_texto: str) -> list[CandidatoLocal]:
        return self._respostas(imagem_tile, prompt_texto)

    @property
    def nome_modelo(self) -> str:
        return "fake-detector"


def _tile(tmp_path: Path, indice: int, origem: tuple[int, int], tamanho: tuple[int, int]) -> Tile:
    caminho = tmp_path / f"tile_{indice}.png"
    Image.new("RGB", tamanho, color="white").save(caminho)
    return Tile(indice=indice, caminho=caminho, origem=origem, tamanho=tamanho, linha=0, coluna=indice)


def _descricao() -> DescricaoMelhorada:
    return DescricaoMelhorada(descricao_melhorada="algo", criterios_exclusao=[], termos_chave=[])


def test_candidatos_filtrados_por_limiar_score(tmp_path: Path) -> None:
    tile = _tile(tmp_path, 0, (0, 0), (100, 100))
    config = Configuracao(dir_saida=tmp_path / "saida", limiar_score_local=0.5, max_candidatos_local=10)
    telemetria = Telemetria()

    candidatos = [
        CandidatoLocal(caixa=Retangulo(x0=0, y0=0, x1=10, y1=10), score=0.9),
        CandidatoLocal(caixa=Retangulo(x0=0, y0=0, x1=10, y1=10), score=0.3),
    ]
    detector = FakeDetectorLocal(lambda imagem, prompt: candidatos)

    resultado = encontra_objeto_local(tile, "a red button", detector, config, telemetria)

    assert [c.score for c in resultado] == [0.9]


def test_ranking_global_ordena_por_score_cross_tile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dir_temp = tmp_path / "temp"
    dir_temp.mkdir()
    tile0 = _tile(tmp_path, 0, (0, 0), (100, 100))
    tile1 = _tile(tmp_path, 1, (100, 0), (100, 100))

    respostas = iter(
        [
            [CandidatoLocal(caixa=Retangulo(x0=0, y0=0, x1=50, y1=50), score=0.4)],  # tile 0
            [CandidatoLocal(caixa=Retangulo(x0=0, y0=0, x1=50, y1=50), score=0.9)],  # tile 1
        ]
    )
    detector = FakeDetectorLocal(lambda imagem, prompt: next(respostas))

    ordem_avaliada: list[int] = []

    def fake_julga(caminho: Path, descricao: Any, config: Any, telemetria: Any, caminho_contexto: Any = None) -> Veredito:
        indice = int(caminho.stem.split("_")[2])
        ordem_avaliada.append(indice)
        return Veredito(aprovado=False, confianca=0.1, feedback="nunca aprova")

    monkeypatch.setattr(pipeline, "julga_resultado", fake_julga)

    imagem_original = Image.new("RGB", (200, 100), color="white")
    config = Configuracao(
        dir_saida=tmp_path / "saida",
        max_candidatos_local=2,
        limiar_score_local=0.0,
        fallback_para_llm=False,
        traduzir_prompt=False,
    )

    pipeline._fluxo_local(
        imagem=imagem_original,
        tiles=[tile0, tile1],
        descricao_melhorada=_descricao(),
        detector=detector,
        config=config,
        telemetria=Telemetria(),
        dir_temp=dir_temp,
        tamanho_original=Tamanho(largura=200, altura=100),
    )

    assert ordem_avaliada[0] == 1  # tile 1 (score 0.9) avaliado antes do tile 0 (score 0.4)


def test_fallback_acionado_quando_sem_candidato(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tile = _tile(tmp_path, 0, (0, 0), (100, 100))
    detector = FakeDetectorLocal(lambda imagem, prompt: [])

    chamado = {"ok": False}

    def fake_fluxo_llm(**kwargs: Any) -> tuple[None, int, int]:
        chamado["ok"] = True
        return None, 0, 0

    monkeypatch.setattr(pipeline, "_fluxo_llm", fake_fluxo_llm)

    imagem_original = Image.new("RGB", (100, 100), color="white")
    config = Configuracao(dir_saida=tmp_path / "saida", fallback_para_llm=True, traduzir_prompt=False)

    pipeline._fluxo_local(
        imagem=imagem_original,
        tiles=[tile],
        descricao_melhorada=_descricao(),
        detector=detector,
        config=config,
        telemetria=Telemetria(),
        dir_temp=tmp_path,
        tamanho_original=Tamanho(largura=100, altura=100),
    )

    assert chamado["ok"] is True


def test_fallback_desabilitado_retorna_falha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tile = _tile(tmp_path, 0, (0, 0), (100, 100))
    detector = FakeDetectorLocal(lambda imagem, prompt: [])

    def fake_fluxo_llm(**kwargs: Any) -> tuple[None, int, int]:
        raise AssertionError("fallback nao deveria ser chamado com fallback_para_llm=False")

    monkeypatch.setattr(pipeline, "_fluxo_llm", fake_fluxo_llm)

    imagem_original = Image.new("RGB", (100, 100), color="white")
    config = Configuracao(dir_saida=tmp_path / "saida", fallback_para_llm=False, traduzir_prompt=False)

    candidato, tentativas, _tiles_avaliados = pipeline._fluxo_local(
        imagem=imagem_original,
        tiles=[tile],
        descricao_melhorada=_descricao(),
        detector=detector,
        config=config,
        telemetria=Telemetria(),
        dir_temp=tmp_path,
        tamanho_original=Tamanho(largura=100, altura=100),
    )

    assert candidato is None
    assert tentativas == 0


def test_conversao_pixels_tile_sem_normalizacao(tmp_path: Path) -> None:
    tile = _tile(tmp_path, 0, (200, 100), (300, 300))
    caixa_tile = Retangulo(x0=10, y0=20, x1=110, y1=220)

    resultado = pixels_tile_para_original(caixa_tile, tile)

    assert resultado == Retangulo(x0=210, y0=120, x1=310, y1=320)


def test_singleton_carrega_uma_vez(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline._detector_cache.clear()
    contador = {"n": 0}

    class FakeBackend(DetectorLocal):
        def carregar(self, config: Configuracao) -> None:
            contador["n"] += 1

        def detectar(self, imagem_tile: Image.Image, prompt_texto: str) -> list[CandidatoLocal]:
            return []

        @property
        def nome_modelo(self) -> str:
            return "fake"

    monkeypatch.setattr("detector.deteccao_local.florence2.Florence2Detector", FakeBackend)

    config = Configuracao(dir_saida=tmp_path / "saida", backend_local="florence2")
    pipeline.obter_detector_local(config)
    pipeline.obter_detector_local(config)

    assert contador["n"] == 1
    pipeline._detector_cache.clear()


def test_metricas_compute_preenchidas(tmp_path: Path) -> None:
    tile = _tile(tmp_path, 0, (0, 0), (50, 50))
    config = Configuracao(dir_saida=tmp_path / "saida")
    telemetria = Telemetria()
    detector = FakeDetectorLocal(lambda imagem, prompt: [])

    encontra_objeto_local(tile, "algo", detector, config, telemetria)
    encontra_objeto_local(tile, "algo", detector, config, telemetria)

    metricas = telemetria.finaliza()
    assert metricas.compute_local.chamadas_detector_local == 2
    assert metricas.compute_local.tempo_inferencia_s >= 0.0


def test_dispositivo_cuda_invalido_aborta_cedo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ErroConfiguracao):
        Configuracao(dir_saida=tmp_path / "saida", dispositivo="cuda")
