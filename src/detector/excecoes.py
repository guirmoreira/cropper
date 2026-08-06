"""Hierarquia de excecoes do detector."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detector.modelos import Metricas


class DetectorErro(Exception):
    """Erro base do detector de objetos.

    `metricas` e preenchido pelo pipeline antes de propagar, para que o
    chamador (CLI) sempre tenha acesso aos custos ja acumulados ate a falha.
    """

    def __init__(self, mensagem: str) -> None:
        super().__init__(mensagem)
        self.metricas: "Metricas | None" = None


class ErroEntrada(DetectorErro):
    """Arquivo inexistente, formato nao suportado ou descricao vazia."""


class ErroConfiguracao(DetectorErro):
    """ANTHROPIC_API_KEY ausente ou parametro de configuracao invalido."""


class ErroProvedor(DetectorErro):
    """Falha da API apos todos os retries de rede."""


class ErroParsing(DetectorErro):
    """JSON invalido apos o retry de correcao de uma etapa."""
