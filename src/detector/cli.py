"""Interface de linha de comando."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from detector import pipeline
from detector.config import Configuracao
from detector.excecoes import (
    DetectorErro,
    ErroConfiguracao,
    ErroEntrada,
    ErroParsing,
    ErroProvedor,
)
from detector.modelos import Metricas, ResultadoDeteccao

load_dotenv()

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()
console_erro = Console(stderr=True)


def _formata_decimal_br(valor: float, casas: int = 2) -> str:
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _formata_inteiro_br(valor: int) -> str:
    return f"{valor:,}".replace(",", ".")


def _configura_logging(verbose: bool) -> None:
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(message)s",
        handlers=[RichHandler(console=console_erro, show_time=False, show_path=False)],
        force=True,
    )


def _verifica_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ErroConfiguracao("ANTHROPIC_API_KEY nao esta definida no ambiente.")


def _monta_configuracao(
    *,
    dir_saida: Path | None,
    modelo: str | None,
    max_tentativas: int | None,
    max_candidatos: int | None,
    tamanho_tile: int | None,
    sobreposicao: float | None,
    manter_temporarios: bool,
    parar_no_primeiro_candidato: bool,
    verbose: bool,
) -> Configuracao:
    overrides: dict[str, Any] = {}
    if dir_saida is not None:
        overrides["dir_saida"] = dir_saida
    if modelo is not None:
        overrides["modelo_visao"] = modelo
    if max_tentativas is not None:
        overrides["max_tentativas_refino"] = max_tentativas
    if max_candidatos is not None:
        overrides["max_candidatos"] = max_candidatos
    if tamanho_tile is not None:
        overrides["tamanho_tile"] = tamanho_tile
    if sobreposicao is not None:
        overrides["sobreposicao"] = sobreposicao
    if manter_temporarios:
        overrides["manter_temporarios"] = True
    if parar_no_primeiro_candidato:
        overrides["parar_no_primeiro_candidato"] = True
    if verbose:
        overrides["log_level"] = "DEBUG"

    try:
        return Configuracao(**overrides)
    except ValidationError as erro:
        raise ErroConfiguracao(str(erro)) from erro


def _tabela_metricas(metricas: Metricas) -> Table:
    tabela = Table(show_header=False, box=None, padding=(0, 2))
    tabela.add_column(justify="left", style="bold")
    tabela.add_column(justify="right")

    tabela.add_row("Chamadas a LLM", _formata_inteiro_br(metricas.chamadas_llm))
    tabela.add_row("Tokens de entrada", _formata_inteiro_br(metricas.tokens_entrada))
    tabela.add_row("Tokens de saida", _formata_inteiro_br(metricas.tokens_saida))
    if metricas.tokens_cache_leitura:
        tabela.add_row("Tokens de cache (leitura)", _formata_inteiro_br(metricas.tokens_cache_leitura))

    custo = f"US$ {_formata_decimal_br(metricas.custo_usd, 4)}"
    if metricas.custo_brl is not None:
        custo += f"  (R$ {_formata_decimal_br(metricas.custo_brl, 2)})"
    tabela.add_row("Custo total", custo)
    tabela.add_row("Tempo de processamento", f"{_formata_decimal_br(metricas.tempo_total_s, 1)} s")

    return tabela


def _imprime_resultado_humano(resultado: ResultadoDeteccao, config: Configuracao) -> None:
    if resultado.sucesso:
        console.print("[bold green]OK[/bold green] Objeto detectado")
        console.print(f"  Arquivo:      {resultado.caminho_imagem}")
        if resultado.caixa_no_original is not None:
            caixa = resultado.caixa_no_original
            console.print(
                f"  Regiao:       ({caixa.x0}, {caixa.y0}) -> ({caixa.x1}, {caixa.y1})   "
                f"[{caixa.largura} x {caixa.altura} px]"
            )
        console.print(f"  Confianca:    {_formata_decimal_br(resultado.confianca_final, 2)}")
    else:
        console.print("[bold red]FALHA[/bold red] Objeto nao detectado")
        console.print(f"  Motivo:       {resultado.mensagem}")
        if resultado.caminho_imagem is not None:
            console.print(f"  Melhor esforco (nao aprovado): {resultado.caminho_imagem}")

    console.print(
        f"  Tentativas:   {resultado.tentativas_refino} de "
        f"{config.max_candidatos * config.max_tentativas_refino}   |   "
        f"Tiles avaliados: {resultado.tiles_avaliados}"
    )
    console.print()
    console.print(Panel(_tabela_metricas(resultado.metricas), title="Metricas", expand=False))


def _emite_erro(erro: DetectorErro, json_saida: bool) -> None:
    if json_saida and erro.metricas is not None:
        console.print_json(data={"sucesso": False, "erro": str(erro), "metricas": erro.metricas.model_dump()})
    else:
        console_erro.print(f"[bold red]Erro:[/bold red] {erro}")
        if erro.metricas is not None:
            console_erro.print(Panel(_tabela_metricas(erro.metricas), title="Metricas", expand=False))


@app.command()
def detectar(
    imagem: Path = typer.Option(..., "--imagem", "-i", help="Caminho da imagem de entrada"),
    descricao: str = typer.Option(
        ..., "--descricao", "-d", help="Descricao do objeto em linguagem natural"
    ),
    dir_saida: Path | None = typer.Option(
        None, "--dir-saida", "-o", help="Diretorio do resultado permanente"
    ),
    modelo: str | None = typer.Option(None, "--modelo", help="Sobrescreve o modelo de visao"),
    max_tentativas: int | None = typer.Option(
        None, "--max-tentativas", help="Refinos por candidato"
    ),
    max_candidatos: int | None = typer.Option(
        None, "--max-candidatos", help="Tiles tentados do ranking"
    ),
    tamanho_tile: int | None = typer.Option(None, "--tamanho-tile", help="Lado do tile em px"),
    sobreposicao: float | None = typer.Option(
        None, "--sobreposicao", help="Fracao de sobreposicao"
    ),
    manter_temporarios: bool = typer.Option(
        False, "--manter-temporarios", help="Nao apaga o diretorio temporario"
    ),
    parar_no_primeiro_candidato: bool = typer.Option(
        False, "--parar-no-primeiro-candidato", help="Nao tenta candidatos alem do primeiro"
    ),
    json_saida: bool = typer.Option(
        False, "--json", help="Emite ResultadoDeteccao em JSON no stdout"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log em nivel DEBUG"),
) -> None:
    _configura_logging(verbose)

    try:
        config = _monta_configuracao(
            dir_saida=dir_saida,
            modelo=modelo,
            max_tentativas=max_tentativas,
            max_candidatos=max_candidatos,
            tamanho_tile=tamanho_tile,
            sobreposicao=sobreposicao,
            manter_temporarios=manter_temporarios,
            parar_no_primeiro_candidato=parar_no_primeiro_candidato,
            verbose=verbose,
        )
        _verifica_api_key()
    except ErroConfiguracao as erro:
        _emite_erro(erro, json_saida)
        raise typer.Exit(code=3) from None

    try:
        resultado = pipeline.detecta(imagem, descricao, config)
    except ErroEntrada as erro:
        _emite_erro(erro, json_saida)
        raise typer.Exit(code=2) from None
    except ErroConfiguracao as erro:
        _emite_erro(erro, json_saida)
        raise typer.Exit(code=3) from None
    except (ErroProvedor, ErroParsing) as erro:
        _emite_erro(erro, json_saida)
        raise typer.Exit(code=4) from None

    if json_saida:
        console.print_json(resultado.model_dump_json())
    else:
        _imprime_resultado_humano(resultado, config)

    raise typer.Exit(code=0 if resultado.sucesso else 1)


if __name__ == "__main__":
    app()
