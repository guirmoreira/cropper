"""Etapa 6 (motor local): localiza candidatos via DetectorLocal, sem chamada a LLM."""

from __future__ import annotations

from PIL import Image

from detector.config import Configuracao
from detector.deteccao_local.base import DetectorLocal
from detector.modelos import CandidatoLocal, Tile
from detector.telemetria import Telemetria


def encontra_objeto_local(
    tile: Tile,
    prompt_ingles: str,
    detector: DetectorLocal,
    config: Configuracao,
    telemetria: Telemetria,
) -> list[CandidatoLocal]:
    with telemetria.cronometra_compute("encontra_objeto_local"):
        imagem_tile = Image.open(tile.caminho).convert("RGB")
        candidatos = detector.detectar(imagem_tile, prompt_ingles)

    candidatos = [c for c in candidatos if c.score >= config.limiar_score_local]
    return candidatos[: config.max_candidatos_local]
