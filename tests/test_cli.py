"""Testes da CLI: codigos de saida e formatacao, com o pipeline mockado."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from detector import cli, pipeline
from detector.excecoes import ErroEntrada, ErroProvedor
from detector.modelos import Metricas, ResultadoDeteccao

runner = CliRunner()


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-teste")


def _resultado_sucesso(tmp_path: Path) -> ResultadoDeteccao:
    from detector.modelos import Retangulo

    caminho = tmp_path / "saida.png"
    caminho.write_bytes(b"fake-png")
    return ResultadoDeteccao(
        sucesso=True,
        caminho_imagem=caminho,
        caixa_no_original=Retangulo(x0=10, y0=20, x1=110, y1=220),
        descricao_original="um botao azul",
        descricao_melhorada="um botao azul, retangular, no topo da tela",
        tiles_avaliados=1,
        tentativas_refino=1,
        confianca_final=0.92,
        metricas=Metricas(
            chamadas_llm=3, tokens_entrada=1000, tokens_saida=100, custo_usd=0.0123
        ),
        run_id="20260806-000000-abcdef",
    )


def test_sem_api_key_retorna_codigo_3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resultado = runner.invoke(cli.app, ["--imagem", str(tmp_path / "x.png"), "--descricao", "um botao"])
    assert resultado.exit_code == 3


def test_motor_invalido_retorna_codigo_3(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resultado = runner.invoke(
        cli.app,
        ["--imagem", str(tmp_path / "x.png"), "--descricao", "algo", "--motor", "invalido"],
    )
    assert resultado.exit_code == 3


def test_arquivo_inexistente_retorna_codigo_2(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_detecta(*args: Any, **kwargs: Any) -> ResultadoDeteccao:
        raise ErroEntrada("Arquivo nao encontrado: x.png")

    monkeypatch.setattr(pipeline, "detecta", fake_detecta)

    resultado = runner.invoke(
        cli.app, ["--imagem", str(tmp_path / "x.png"), "--descricao", "um botao"]
    )
    assert resultado.exit_code == 2


def test_erro_provedor_retorna_codigo_4(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_detecta(*args: Any, **kwargs: Any) -> ResultadoDeteccao:
        raise ErroProvedor("falha de rede apos retries")

    monkeypatch.setattr(pipeline, "detecta", fake_detecta)

    resultado = runner.invoke(
        cli.app, ["--imagem", str(tmp_path / "x.png"), "--descricao", "um botao"]
    )
    assert resultado.exit_code == 4


def test_sucesso_retorna_codigo_0_e_imprime_metricas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resultado_mock = _resultado_sucesso(tmp_path)
    monkeypatch.setattr(pipeline, "detecta", lambda *a, **k: resultado_mock)

    resultado = runner.invoke(
        cli.app, ["--imagem", str(tmp_path / "x.png"), "--descricao", "um botao azul"]
    )

    assert resultado.exit_code == 0
    assert "Objeto detectado" in resultado.stdout
    assert "0,92" in resultado.stdout
    assert "1.000" in resultado.stdout  # tokens_entrada formatado em pt-BR


def test_falha_sem_excecao_retorna_codigo_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resultado_falha = ResultadoDeteccao(
        sucesso=False,
        descricao_original="algo",
        descricao_melhorada="algo",
        tiles_avaliados=2,
        tentativas_refino=6,
        mensagem="Objeto nao localizado com confianca suficiente apos as tentativas configuradas.",
        metricas=Metricas(chamadas_llm=10, custo_usd=0.05),
        run_id="20260806-000001-abcdef",
    )
    monkeypatch.setattr(pipeline, "detecta", lambda *a, **k: resultado_falha)

    resultado = runner.invoke(
        cli.app, ["--imagem", str(tmp_path / "x.png"), "--descricao", "algo"]
    )

    assert resultado.exit_code == 1
    assert "nao detectado" in resultado.stdout.lower()


def test_flag_json_emite_json_valido(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    resultado_mock = _resultado_sucesso(tmp_path)
    monkeypatch.setattr(pipeline, "detecta", lambda *a, **k: resultado_mock)

    resultado = runner.invoke(
        cli.app,
        ["--imagem", str(tmp_path / "x.png"), "--descricao", "um botao azul", "--json"],
    )

    assert resultado.exit_code == 0
    dados = json.loads(resultado.stdout)
    assert dados["sucesso"] is True
    assert dados["run_id"] == "20260806-000000-abcdef"
