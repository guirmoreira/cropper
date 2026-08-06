"""Etapa 6 (motor LLM): localiza o objeto dentro de um tile via LLM multimodal.

Com `motor_localizacao="local"` (padrao), esta etapa so e usada como fallback quando o
detector local falha ou nao aprova nenhum candidato -- ver `pipeline._fluxo_llm`.
"""

from __future__ import annotations

from detector.config import Configuracao
from detector.imagem import prepara_para_llm
from detector.llm import bloco_imagem, chamar_llm
from detector.modelos import DescricaoMelhorada, ResultadoLocalizacao, RetanguloNormalizado, Tile
from detector.prompts import (
    SISTEMA_ENCONTRA_OBJETO,
    bloco_feedback_tentativa_anterior,
    mensagem_usuario_encontra_objeto,
)
from detector.telemetria import Telemetria


def encontra_objeto(
    tile: Tile,
    descricao: DescricaoMelhorada,
    feedback: str | None,
    caixa_anterior: RetanguloNormalizado | None,
    config: Configuracao,
    telemetria: Telemetria,
) -> ResultadoLocalizacao:
    largura_tile, altura_tile = tile.tamanho

    bloco_feedback = ""
    if feedback is not None and caixa_anterior is not None:
        bloco_feedback = bloco_feedback_tentativa_anterior(
            x0=caixa_anterior.x0,
            y0=caixa_anterior.y0,
            x1=caixa_anterior.x1,
            y1=caixa_anterior.y1,
            feedback=feedback,
        )

    texto_usuario = mensagem_usuario_encontra_objeto(
        largura=largura_tile,
        altura=altura_tile,
        descricao_melhorada=descricao.descricao_melhorada,
        criterios_exclusao=descricao.criterios_exclusao,
        bloco_feedback=bloco_feedback,
    )

    caminho_llm = prepara_para_llm(tile.caminho, config.limiar_dimensao)
    conteudo = [
        {"type": "text", "text": texto_usuario},
        bloco_imagem(caminho_llm),
    ]

    return chamar_llm(
        etapa="encontra_objeto",
        modelo=config.modelo_visao,
        sistema=SISTEMA_ENCONTRA_OBJETO,
        conteudo=conteudo,
        schema=ResultadoLocalizacao,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )
