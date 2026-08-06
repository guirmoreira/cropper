"""Hierarquia de excecoes do detector."""

from __future__ import annotations


class DetectorErro(Exception):
    """Erro base do detector de objetos."""


class ErroEntrada(DetectorErro):
    """Arquivo inexistente, formato nao suportado ou descricao vazia."""


class ErroConfiguracao(DetectorErro):
    """ANTHROPIC_API_KEY ausente ou parametro de configuracao invalido."""


class ErroProvedor(DetectorErro):
    """Falha da API apos todos os retries de rede."""


class ErroParsing(DetectorErro):
    """JSON invalido apos o retry de correcao de uma etapa."""
