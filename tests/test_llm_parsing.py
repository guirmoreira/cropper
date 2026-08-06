"""Testes de llm.py com litellm.completion mockado -- sem chamadas de rede."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import pytest
from pydantic import BaseModel

from detector.excecoes import ErroParsing, ErroProvedor
from detector.llm import chamar_llm, extrair_json
from detector.telemetria import Telemetria

SRC = Path(__file__).resolve().parent.parent / "src" / "detector"


def test_litellm_completion_so_e_chamado_em_llm_py() -> None:
    """llm.py e o unico lugar do codigo que pode invocar litellm.completion."""
    padrao = re.compile(r"litellm\.completion\b")
    ofensores = []
    for caminho in SRC.rglob("*.py"):
        if caminho.name == "llm.py":
            continue
        if padrao.search(caminho.read_text(encoding="utf-8")):
            ofensores.append(str(caminho))
    assert ofensores == []


class SchemaTeste(BaseModel):
    valor: int
    rotulo: str


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    cache_read_input_tokens: int = 0


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)

    @classmethod
    def com_texto(cls, texto: str, usage: FakeUsage | None = None) -> FakeResponse:
        return cls(choices=[FakeChoice(FakeMessage(content=texto))], usage=usage or FakeUsage())


@pytest.fixture(autouse=True)
def _mock_custo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response=None: 0.001)
    monkeypatch.setattr(litellm, "supports_response_schema", lambda model=None: True)


def test_sucesso_primeira_tentativa(monkeypatch: pytest.MonkeyPatch) -> None:
    respostas = [FakeResponse.com_texto(json.dumps({"valor": 42, "rotulo": "ok"}))]

    def fake_completion(**kwargs: Any) -> FakeResponse:
        return respostas.pop(0)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    resultado = chamar_llm(
        etapa="teste",
        modelo="anthropic/claude-haiku-4-5-20251001",
        sistema="sistema",
        conteudo=[{"type": "text", "text": "ola"}],
        schema=SchemaTeste,
        telemetria=telemetria,
    )

    assert resultado == SchemaTeste(valor=42, rotulo="ok")
    metricas = telemetria.finaliza()
    assert metricas.chamadas_llm == 1


def test_correcao_apos_json_malformado(monkeypatch: pytest.MonkeyPatch) -> None:
    respostas = [
        FakeResponse.com_texto("isso nao e json valido"),
        FakeResponse.com_texto(json.dumps({"valor": 7, "rotulo": "corrigido"})),
    ]

    def fake_completion(**kwargs: Any) -> FakeResponse:
        return respostas.pop(0)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    resultado = chamar_llm(
        etapa="teste",
        modelo="anthropic/claude-haiku-4-5-20251001",
        sistema="sistema",
        conteudo=[{"type": "text", "text": "ola"}],
        schema=SchemaTeste,
        telemetria=telemetria,
    )

    assert resultado == SchemaTeste(valor=7, rotulo="corrigido")
    metricas = telemetria.finaliza()
    assert metricas.chamadas_llm == 2


def test_desembrulha_resposta_envolta_em_chave_unica(monkeypatch: pytest.MonkeyPatch) -> None:
    """Alguns modelos devolvem os campos aninhados sob uma chave extra (ex.: 'parameters')."""
    respostas = [
        FakeResponse.com_texto(json.dumps({"parameters": {"valor": 9, "rotulo": "aninhado"}}))
    ]

    def fake_completion(**kwargs: Any) -> FakeResponse:
        return respostas.pop(0)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    resultado = chamar_llm(
        etapa="teste",
        modelo="anthropic/claude-haiku-4-5-20251001",
        sistema="sistema",
        conteudo=[{"type": "text", "text": "ola"}],
        schema=SchemaTeste,
        telemetria=telemetria,
    )

    assert resultado == SchemaTeste(valor=9, rotulo="aninhado")
    metricas = telemetria.finaliza()
    assert metricas.chamadas_llm == 1


def test_falha_apos_correcao_levanta_erro_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    respostas = [
        FakeResponse.com_texto("nao e json"),
        FakeResponse.com_texto("ainda nao e json"),
    ]

    def fake_completion(**kwargs: Any) -> FakeResponse:
        return respostas.pop(0)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    with pytest.raises(ErroParsing):
        chamar_llm(
            etapa="teste",
            modelo="anthropic/claude-haiku-4-5-20251001",
            sistema="sistema",
            conteudo=[{"type": "text", "text": "ola"}],
            schema=SchemaTeste,
            telemetria=telemetria,
        )

    metricas = telemetria.finaliza()
    assert metricas.chamadas_llm == 2


def test_fallback_json_quando_response_format_nao_suportado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chamadas: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> FakeResponse:
        chamadas.append(kwargs)
        if "response_format" in kwargs:
            raise litellm.BadRequestError(
                message="response_format nao suportado",
                model=kwargs.get("model"),
                llm_provider="anthropic",
            )
        return FakeResponse.com_texto(json.dumps({"valor": 1, "rotulo": "fallback"}))

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    resultado = chamar_llm(
        etapa="teste",
        modelo="anthropic/claude-haiku-4-5-20251001",
        sistema="sistema",
        conteudo=[{"type": "text", "text": "ola"}],
        schema=SchemaTeste,
        telemetria=telemetria,
    )

    assert resultado == SchemaTeste(valor=1, rotulo="fallback")
    assert len(chamadas) == 2
    assert "response_format" in chamadas[0]
    assert "response_format" not in chamadas[1]
    assert "Schema:" in chamadas[1]["messages"][0]["content"]


def test_erro_de_rede_exaure_retries_e_levanta_erro_provedor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chamadas = {"n": 0}

    def fake_completion(**kwargs: Any) -> FakeResponse:
        chamadas["n"] += 1
        raise litellm.Timeout(message="timeout", model=kwargs.get("model"), llm_provider="anthropic")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    telemetria = Telemetria()

    with pytest.raises(ErroProvedor):
        chamar_llm(
            etapa="teste",
            modelo="anthropic/claude-haiku-4-5-20251001",
            sistema="sistema",
            conteudo=[{"type": "text", "text": "ola"}],
            schema=SchemaTeste,
            telemetria=telemetria,
            max_retries=2,
        )

    assert chamadas["n"] == 2
    metricas = telemetria.finaliza()
    assert metricas.chamadas_llm == 0


def test_extrair_json_ignora_texto_ao_redor() -> None:
    texto = 'Aqui esta: ```json\n{"a": 1, "b": {"c": [1, 2]}}\n``` Espero que ajude.'
    assert json.loads(extrair_json(texto)) == {"a": 1, "b": {"c": [1, 2]}}


def test_extrair_json_sem_objeto_levanta_erro_parsing() -> None:
    with pytest.raises(ErroParsing):
        extrair_json("nao ha json aqui")
