"""Interface DetectorLocal, implementada por cada backend em `deteccao_local/`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from detector.config import Configuracao
from detector.modelos import CandidatoLocal


class DetectorLocal(ABC):
    """Backend de deteccao open-vocabulary local (Florence-2, Grounding DINO, etc.)."""

    def __init__(self) -> None:
        self.dispositivo: str = "cpu"

    @abstractmethod
    def carregar(self, config: Configuracao) -> None:
        """Carrega pesos e move para o dispositivo. Idempotente -- chamado uma vez por processo."""

    @abstractmethod
    def detectar(self, imagem_tile: Image.Image, prompt_texto: str) -> list[CandidatoLocal]:
        """Devolve candidatos ORDENADOS por score decrescente, em pixels do tile recebido."""

    @property
    @abstractmethod
    def nome_modelo(self) -> str: ...


def resolve_dispositivo(config: Configuracao) -> str:
    """Resolve dispositivo='auto' para 'cuda', 'mps' ou 'cpu' conforme disponibilidade."""
    if config.dispositivo != "auto":
        return config.dispositivo

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
