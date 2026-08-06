"""Acumulador thread-safe de metricas de uso da LLM."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from detector.modelos import Metricas, MetricasEtapa


class Telemetria:
    def __init__(self, taxa_usd_brl: float | None = None) -> None:
        self._lock = threading.Lock()
        self._taxa_usd_brl = taxa_usd_brl
        self._inicio = time.monotonic()

        self._chamadas_llm = 0
        self._tokens_entrada = 0
        self._tokens_saida = 0
        self._tokens_cache_leitura = 0
        self._custo_usd = 0.0
        self._por_etapa: dict[str, MetricasEtapa] = {}

    def registra(
        self,
        *,
        etapa: str,
        tokens_entrada: int,
        tokens_saida: int,
        tokens_cache: int = 0,
        custo_usd: float,
        duracao_s: float,
    ) -> None:
        """Registra o resultado de uma chamada a LLM. Thread-safe."""
        with self._lock:
            self._chamadas_llm += 1
            self._tokens_entrada += tokens_entrada
            self._tokens_saida += tokens_saida
            self._tokens_cache_leitura += tokens_cache
            self._custo_usd += custo_usd

            metricas_etapa = self._por_etapa.setdefault(etapa, MetricasEtapa())
            metricas_etapa.chamadas += 1
            metricas_etapa.tokens_entrada += tokens_entrada
            metricas_etapa.tokens_saida += tokens_saida
            metricas_etapa.custo_usd += custo_usd
            metricas_etapa.tempo_s += duracao_s

    @contextmanager
    def cronometra(self, etapa: str) -> Iterator[None]:
        """Acumula o tempo de parede do bloco em `por_etapa[etapa].tempo_s`."""
        inicio = time.monotonic()
        try:
            yield
        finally:
            duracao = time.monotonic() - inicio
            with self._lock:
                metricas_etapa = self._por_etapa.setdefault(etapa, MetricasEtapa())
                metricas_etapa.tempo_s += duracao

    def finaliza(self) -> Metricas:
        with self._lock:
            custo_brl = (
                self._custo_usd * self._taxa_usd_brl if self._taxa_usd_brl is not None else None
            )
            return Metricas(
                chamadas_llm=self._chamadas_llm,
                tokens_entrada=self._tokens_entrada,
                tokens_saida=self._tokens_saida,
                tokens_cache_leitura=self._tokens_cache_leitura,
                custo_usd=self._custo_usd,
                custo_brl=custo_brl,
                tempo_total_s=time.monotonic() - self._inicio,
                por_etapa=dict(self._por_etapa),
            )
