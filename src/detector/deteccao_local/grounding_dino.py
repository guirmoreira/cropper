"""Backend Grounding DINO (IDEA-Research/grounding-dino-base) para DetectorLocal.

Todos os imports de `torch`/`transformers` sao adiados para `carregar()`/`detectar()`, para
que importar este modulo (ou o pacote `deteccao_local`) nao force a dependencia pesada em
processos que usam apenas `motor_localizacao="llm"`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from PIL import Image

from detector.config import Configuracao
from detector.deteccao_local.base import DetectorLocal, resolve_dispositivo
from detector.modelos import CandidatoLocal, Retangulo

logger = logging.getLogger(__name__)

CHECKPOINT = "IDEA-Research/grounding-dino-base"
TEXT_THRESHOLD = 0.25


class GroundingDinoDetector(DetectorLocal):
    def __init__(self) -> None:
        super().__init__()
        self._modelo: Any = None
        self._processor: Any = None
        self._limiar_score: float = 0.30

    def carregar(self, config: Configuracao) -> None:
        if self._modelo is not None:
            return

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.dispositivo = resolve_dispositivo(config)
        self._limiar_score = config.limiar_score_local
        cache_dir = str(config.cache_dir_modelos) if config.cache_dir_modelos else None

        inicio = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=cache_dir)
        self._modelo = AutoModelForZeroShotObjectDetection.from_pretrained(
            CHECKPOINT, cache_dir=cache_dir
        ).to(self.dispositivo)
        self._modelo.eval()
        logger.info(
            "Grounding DINO carregado em %s (%.1fs)", self.dispositivo, time.monotonic() - inicio
        )

    def detectar(self, imagem_tile: Image.Image, prompt_texto: str) -> list[CandidatoLocal]:
        import torch

        if self._modelo is None or self._processor is None:
            raise RuntimeError("GroundingDinoDetector.detectar() chamado antes de carregar().")

        prompt = prompt_texto.lower().strip()
        if not prompt.endswith("."):
            prompt += "."

        inputs = self._processor(images=imagem_tile, text=prompt, return_tensors="pt").to(
            self.dispositivo
        )
        with torch.no_grad():
            saida = self._modelo(**inputs)

        resultados = self._processor.post_process_grounded_object_detection(
            saida,
            inputs.input_ids,
            box_threshold=self._limiar_score,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[imagem_tile.size[::-1]],
        )[0]

        caixas = resultados["boxes"].tolist()
        scores = resultados["scores"].tolist()
        rotulos = resultados.get("text_labels") or resultados.get("labels") or [""] * len(caixas)

        candidatos: list[CandidatoLocal] = []
        for caixa, score, rotulo in zip(caixas, scores, rotulos, strict=False):
            x0, y0, x1, y1 = caixa
            if x1 <= x0 or y1 <= y0:
                continue
            candidatos.append(
                CandidatoLocal(
                    caixa=Retangulo(x0=round(x0), y0=round(y0), x1=round(x1), y1=round(y1)),
                    score=float(score),
                    rotulo=str(rotulo),
                )
            )
        candidatos.sort(key=lambda c: c.score, reverse=True)
        return candidatos

    @property
    def nome_modelo(self) -> str:
        return "grounding-dino-base"
