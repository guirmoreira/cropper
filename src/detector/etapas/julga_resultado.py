"""Etapa 8: julga se o recorte candidato corresponde ao objeto pedido."""

from __future__ import annotations

from pathlib import Path

from detector.config import Configuracao
from detector.imagem import prepara_para_llm
from detector.llm import bloco_imagem, chamar_llm
from detector.modelos import DescricaoMelhorada, Veredito
from detector.prompts import SISTEMA_JULGA_RESULTADO
from detector.telemetria import Telemetria


def julga_resultado(
    candidato: Path,
    descricao: DescricaoMelhorada,
    config: Configuracao,
    telemetria: Telemetria,
    caminho_contexto: Path | None = None,
) -> Veredito:
    conteudo: list[dict] = [
        {"type": "text", "text": f"OBJETO PEDIDO:\n{descricao.descricao_melhorada}"},
        bloco_imagem(prepara_para_llm(candidato, config.limiar_dimensao)),
    ]

    if caminho_contexto is not None:
        conteudo.append(
            {"type": "text", "text": "Contexto: tile de origem, com o recorte candidato acima."}
        )
        conteudo.append(bloco_imagem(prepara_para_llm(caminho_contexto, config.limiar_dimensao)))

    return chamar_llm(
        etapa="julga_resultado",
        modelo=config.modelo_juiz,
        sistema=SISTEMA_JULGA_RESULTADO,
        conteudo=conteudo,
        schema=Veredito,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )
