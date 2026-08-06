"""Etapa 5: ordena os tiles pela probabilidade de conter o objeto-alvo."""

from __future__ import annotations

import json

from detector.config import Configuracao
from detector.llm import chamar_llm
from detector.modelos import DescricaoMelhorada, DescricaoTile, ItemRanking, Ranking
from detector.prompts import SISTEMA_ESCOLHE_IMAGEM
from detector.telemetria import Telemetria

PONTUACAO_MINIMA = 0.05


def escolher_imagem(
    descricao: DescricaoMelhorada,
    descricoes: list[DescricaoTile],
    config: Configuracao,
    telemetria: Telemetria,
) -> Ranking:
    payload = {
        "descricao_melhorada": descricao.descricao_melhorada,
        "criterios_exclusao": descricao.criterios_exclusao,
        "tiles": [d.model_dump() for d in descricoes],
    }
    conteudo = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]

    ranking = chamar_llm(
        etapa="escolhe_imagem",
        modelo=config.modelo_texto,
        sistema=SISTEMA_ESCOLHE_IMAGEM,
        conteudo=conteudo,
        schema=Ranking,
        telemetria=telemetria,
        temperatura=config.temperatura,
        seed=config.seed,
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    itens_ordenados = sorted(ranking.itens, key=lambda item: item.pontuacao, reverse=True)
    itens_validos = [item for item in itens_ordenados if item.pontuacao >= PONTUACAO_MINIMA]
    itens_truncados = itens_validos[: config.max_candidatos]

    if not itens_truncados:
        itens_truncados = [
            ItemRanking(
                indice_tile=d.indice_tile,
                pontuacao=0.0,
                justificativa="fallback: ranking vazio, tentando todos os tiles",
            )
            for d in descricoes
        ]

    return Ranking(itens=itens_truncados)
