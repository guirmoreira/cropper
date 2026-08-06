"""Testes de acumulacao de metricas, incluindo concorrencia."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from detector.telemetria import Telemetria


def test_telemetria_concorrente() -> None:
    telemetria = Telemetria()
    n_threads = 100

    def _registrar(_: int) -> None:
        telemetria.registra(
            etapa="encontra_objeto",
            tokens_entrada=10,
            tokens_saida=5,
            tokens_cache=2,
            custo_usd=0.001,
            duracao_s=0.01,
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_registrar, range(n_threads)))

    metricas = telemetria.finaliza()

    assert metricas.chamadas_llm == n_threads
    assert metricas.tokens_entrada == 10 * n_threads
    assert metricas.tokens_saida == 5 * n_threads
    assert metricas.tokens_cache_leitura == 2 * n_threads
    assert metricas.custo_usd == pytest.approx(0.001 * n_threads)
    assert metricas.por_etapa["encontra_objeto"].chamadas == n_threads


def test_telemetria_custo_brl() -> None:
    telemetria = Telemetria(taxa_usd_brl=5.0)
    telemetria.registra(
        etapa="julga_resultado",
        tokens_entrada=100,
        tokens_saida=20,
        custo_usd=0.10,
        duracao_s=0.5,
    )

    metricas = telemetria.finaliza()

    assert metricas.custo_brl == pytest.approx(0.5)


def test_telemetria_sem_taxa_nao_calcula_brl() -> None:
    telemetria = Telemetria()
    telemetria.registra(
        etapa="julga_resultado", tokens_entrada=1, tokens_saida=1, custo_usd=0.01, duracao_s=0.1
    )

    metricas = telemetria.finaliza()

    assert metricas.custo_brl is None


def test_cronometra_acumula_tempo() -> None:
    telemetria = Telemetria()
    with telemetria.cronometra("escolhe_imagem"):
        time.sleep(0.01)

    metricas = telemetria.finaliza()

    assert metricas.por_etapa["escolhe_imagem"].tempo_s >= 0.01
