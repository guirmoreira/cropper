"""Configuracao da aplicacao, carregada de variaveis de ambiente / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from detector.excecoes import ErroConfiguracao


class Configuracao(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DETECTOR_", env_file=".env", extra="ignore")

    modelo_visao: str = "anthropic/claude-haiku-4-5-20251001"
    modelo_texto: str = "anthropic/claude-haiku-4-5-20251001"
    modelo_juiz: str = "anthropic/claude-haiku-4-5-20251001"

    limiar_dimensao: int = 1568
    tamanho_tile: int = 1400
    sobreposicao: float = 0.20
    max_tentativas_refino: int = 3
    max_candidatos: int = 5
    workers: int = 4
    temperatura: float = 0.0
    seed: int | None = None
    timeout_s: int = 120
    max_retries: int = 3

    margem_recorte: int = 4
    parar_no_primeiro_candidato: bool = False
    devolver_melhor_esforco: bool = False

    dir_saida: Path = Path("./saida")
    manter_temporarios: bool = False
    taxa_usd_brl: float | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def valida(self) -> Configuracao:
        if not (0.0 <= self.sobreposicao < 0.5):
            raise ErroConfiguracao(
                f"sobreposicao deve estar em [0.0, 0.5), recebido: {self.sobreposicao}"
            )
        if self.tamanho_tile > self.limiar_dimensao:
            raise ErroConfiguracao(
                "tamanho_tile nao pode ser maior que limiar_dimensao: "
                f"{self.tamanho_tile} > {self.limiar_dimensao}"
            )
        if self.max_tentativas_refino < 1:
            raise ErroConfiguracao(
                f"max_tentativas_refino deve ser >= 1, recebido: {self.max_tentativas_refino}"
            )
        if self.max_candidatos < 1:
            raise ErroConfiguracao(
                f"max_candidatos deve ser >= 1, recebido: {self.max_candidatos}"
            )
        if self.workers < 1:
            raise ErroConfiguracao(f"workers deve ser >= 1, recebido: {self.workers}")
        return self
