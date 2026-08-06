"""Backend Florence-2 (microsoft/Florence-2-base) para DetectorLocal.

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

CHECKPOINT = "microsoft/Florence-2-base"
TAREFA = "<CAPTION_TO_PHRASE_GROUNDING>"


class Florence2Detector(DetectorLocal):
    def __init__(self) -> None:
        super().__init__()
        self._modelo: Any = None
        self._processor: Any = None

    def carregar(self, config: Configuracao) -> None:
        if self._modelo is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        self.dispositivo = resolve_dispositivo(config)
        cache_dir = str(config.cache_dir_modelos) if config.cache_dir_modelos else None

        inicio = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(
            CHECKPOINT, trust_remote_code=True, cache_dir=cache_dir
        )
        self._modelo = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT, trust_remote_code=True, cache_dir=cache_dir
        ).to(self.dispositivo)
        self._modelo.eval()
        logger.info(
            "Florence-2 carregado em %s (%.1fs)", self.dispositivo, time.monotonic() - inicio
        )

    def detectar(self, imagem_tile: Image.Image, prompt_texto: str) -> list[CandidatoLocal]:
        import torch

        if self._modelo is None or self._processor is None:
            raise RuntimeError("Florence2Detector.detectar() chamado antes de carregar().")

        prompt = f"{TAREFA}{prompt_texto}"
        inputs = self._processor(text=prompt, images=imagem_tile, return_tensors="pt").to(
            self.dispositivo
        )

        with torch.no_grad():
            saida = self._modelo.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
            )
        texto = self._processor.batch_decode(saida, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            texto, task=TAREFA, image_size=imagem_tile.size
        )
        dados = parsed.get(TAREFA, {})
        caixas = dados.get("bboxes", [])
        rotulos = dados.get("labels", [])

        # Florence-2 nao devolve score de confianca nativo nesta tarefa: usamos a ordem de
        # geracao como proxy de ranking posicional (1a frase = mais confiavel).
        candidatos: list[CandidatoLocal] = []
        for indice, caixa in enumerate(caixas):
            x0, y0, x1, y1 = caixa
            if x1 <= x0 or y1 <= y0:
                continue
            candidatos.append(
                CandidatoLocal(
                    caixa=Retangulo(x0=round(x0), y0=round(y0), x1=round(x1), y1=round(y1)),
                    score=1.0 / (indice + 1),
                    rotulo=rotulos[indice] if indice < len(rotulos) else "",
                )
            )
        return candidatos

    @property
    def nome_modelo(self) -> str:
        return "florence2-base"
