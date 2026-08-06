"""Unico ponto de chamada ao litellm.

Garante telemetria uniforme, retries, parsing de JSON e tratamento de erro
para toda chamada a LLM no sistema -- nenhuma chamada deve escapar daqui
(ver teste que faz grep por `litellm.completion` fora deste modulo).
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from detector.excecoes import ErroParsing, ErroProvedor
from detector.prompts import INSTRUCAO_JSON
from detector.telemetria import Telemetria

logger = logging.getLogger(__name__)

ERROS_RETENTAVEIS = (
    litellm.Timeout,
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
)
ERROS_PARAM_NAO_SUPORTADO = (litellm.UnsupportedParamsError, litellm.BadRequestError)


def bloco_imagem(caminho: Path) -> dict[str, Any]:
    dados = base64.b64encode(caminho.read_bytes()).decode()
    mime = "image/png" if caminho.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{dados}"}}


def extrair_json(texto: str) -> str:
    """Remove cercas markdown e devolve o primeiro objeto JSON balanceado."""
    if not texto:
        raise ErroParsing("Resposta vazia da LLM; nenhum JSON para extrair.")

    sem_cercas = texto.replace("```json", "```").replace("```", "")
    inicio = sem_cercas.find("{")
    if inicio == -1:
        raise ErroParsing(f"Nenhum objeto JSON encontrado na resposta: {texto[:200]!r}")

    profundidade = 0
    dentro_string = False
    escapando = False
    for i in range(inicio, len(sem_cercas)):
        c = sem_cercas[i]
        if escapando:
            escapando = False
            continue
        if c == "\\":
            escapando = True
            continue
        if c == '"':
            dentro_string = not dentro_string
            continue
        if dentro_string:
            continue
        if c == "{":
            profundidade += 1
        elif c == "}":
            profundidade -= 1
            if profundidade == 0:
                return sem_cercas[inicio : i + 1]

    raise ErroParsing(f"JSON desbalanceado na resposta: {texto[:200]!r}")


def _extrai_texto(resp: Any) -> str:
    mensagem = resp.choices[0].message
    if mensagem.content:
        return str(mensagem.content)
    tool_calls = getattr(mensagem, "tool_calls", None)
    if tool_calls:
        return str(tool_calls[0].function.arguments)
    return ""


def _suporta_response_format(modelo: str) -> bool:
    try:
        return bool(litellm.supports_response_schema(model=modelo))
    except Exception:
        logger.debug("Nao foi possivel verificar suporte a response_format para %s", modelo)
        return True


def _executa_com_retry(max_retries: int, **kwargs: Any) -> Any:
    @retry(
        stop=stop_after_attempt(max(1, max_retries)),
        wait=wait_exponential(multiplier=1, max=20),
        retry=retry_if_exception_type(ERROS_RETENTAVEIS),
        reraise=True,
    )
    def _chamar() -> Any:
        return litellm.completion(**kwargs)

    return _chamar()


def _requisitar(
    *,
    etapa: str,
    modelo: str,
    mensagens: list[dict[str, Any]],
    temperatura: float,
    max_tokens: int,
    seed: int | None,
    timeout_s: int,
    max_retries: int,
    response_format: type[BaseModel] | None,
    telemetria: Telemetria,
) -> str:
    kwargs: dict[str, Any] = {
        "model": modelo,
        "messages": mensagens,
        "temperature": temperatura,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
    }
    if seed is not None:
        kwargs["seed"] = seed
    if response_format is not None:
        kwargs["response_format"] = response_format

    logger.debug("Chamando LLM (etapa=%s, modelo=%s): %s", etapa, modelo, str(mensagens)[:500])

    inicio = time.monotonic()
    try:
        resp = _executa_com_retry(max_retries, **kwargs)
    except ERROS_RETENTAVEIS as exc:
        raise ErroProvedor(
            f"Falha ao chamar '{modelo}' na etapa '{etapa}' apos {max_retries} tentativa(s): {exc}"
        ) from exc
    duracao = time.monotonic() - inicio

    usage = getattr(resp, "usage", None)
    tokens_entrada = int(getattr(usage, "prompt_tokens", 0) or 0)
    tokens_saida = int(getattr(usage, "completion_tokens", 0) or 0)
    tokens_cache = int(getattr(usage, "cache_read_input_tokens", 0) or 0)

    try:
        custo_usd = float(litellm.completion_cost(completion_response=resp))
    except Exception:
        logger.debug("Falha ao calcular custo via completion_cost; usando 0.0", exc_info=True)
        custo_usd = 0.0

    telemetria.registra(
        etapa=etapa,
        tokens_entrada=tokens_entrada,
        tokens_saida=tokens_saida,
        tokens_cache=tokens_cache,
        custo_usd=custo_usd,
        duracao_s=duracao,
    )

    texto = _extrai_texto(resp)
    logger.debug("Resposta bruta (etapa=%s): %s", etapa, texto[:2000])
    return texto


def chamar_llm[T: BaseModel](
    *,
    etapa: str,
    modelo: str,
    sistema: str,
    conteudo: list[dict[str, Any]],
    schema: type[T],
    telemetria: Telemetria,
    temperatura: float = 0.0,
    max_tokens: int = 2048,
    seed: int | None = None,
    timeout_s: int = 120,
    max_retries: int = 3,
) -> T:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    sistema_fallback = f"{sistema}\n\n{INSTRUCAO_JSON.format(schema_json=schema_json)}"

    def _monta_mensagens(usar_fallback: bool) -> list[dict[str, Any]]:
        texto_sistema = sistema_fallback if usar_fallback else sistema
        return [
            {"role": "system", "content": texto_sistema},
            {"role": "user", "content": conteudo},
        ]

    usar_fallback = not _suporta_response_format(modelo)

    def _chama(usar_fallback: bool) -> str:
        return _requisitar(
            etapa=etapa,
            modelo=modelo,
            mensagens=_monta_mensagens(usar_fallback),
            temperatura=temperatura,
            max_tokens=max_tokens,
            seed=seed,
            timeout_s=timeout_s,
            max_retries=max_retries,
            response_format=None if usar_fallback else schema,
            telemetria=telemetria,
        )

    try:
        texto = _chama(usar_fallback)
    except ERROS_PARAM_NAO_SUPORTADO:
        if usar_fallback:
            raise
        logger.warning(
            "Modelo '%s' nao aceitou response_format estruturado; usando fallback JSON.", modelo
        )
        usar_fallback = True
        texto = _chama(usar_fallback)

    try:
        return schema.model_validate_json(extrair_json(texto))
    except (ValidationError, ErroParsing) as erro_original:
        logger.info("Resposta invalida na etapa '%s'; solicitando correcao.", etapa)
        mensagens_correcao = _monta_mensagens(usar_fallback) + [
            {"role": "assistant", "content": texto},
            {
                "role": "user",
                "content": (
                    "Sua resposta anterior nao satisfez o schema esperado. "
                    f"Erro de validacao:\n{erro_original}\n\n"
                    "Corrija e responda novamente, exclusivamente com o JSON valido."
                ),
            },
        ]
        texto_corrigido = _requisitar(
            etapa=etapa,
            modelo=modelo,
            mensagens=mensagens_correcao,
            temperatura=temperatura,
            max_tokens=max_tokens,
            seed=seed,
            timeout_s=timeout_s,
            max_retries=max_retries,
            response_format=None if usar_fallback else schema,
            telemetria=telemetria,
        )
        try:
            return schema.model_validate_json(extrair_json(texto_corrigido))
        except (ValidationError, ErroParsing) as erro_final:
            raise ErroParsing(
                f"Resposta invalida da etapa '{etapa}' apos retry de correcao: {erro_final}"
            ) from erro_final
