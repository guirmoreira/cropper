"""Modelos de dados (Pydantic) usados em todo o pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class Tamanho(BaseModel):
    """Dimensoes em pixels de uma imagem."""

    largura: int
    altura: int


class Retangulo(BaseModel):
    """Coordenadas em PIXELS ABSOLUTOS da imagem de referencia."""

    x0: int
    y0: int
    x1: int
    y1: int

    @model_validator(mode="after")
    def valida(self) -> Retangulo:
        if self.x1 <= self.x0:
            raise ValueError(f"x1 deve ser maior que x0: x0={self.x0}, x1={self.x1}")
        if self.y1 <= self.y0:
            raise ValueError(f"y1 deve ser maior que y0: y0={self.y0}, y1={self.y1}")
        return self

    @property
    def largura(self) -> int:
        return self.x1 - self.x0

    @property
    def altura(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.largura * self.altura

    def expandir(self, margem_px: int, limite: Tamanho) -> Retangulo:
        """Expande em todas as direcoes, clampando aos limites da imagem."""
        return Retangulo(
            x0=max(0, self.x0 - margem_px),
            y0=max(0, self.y0 - margem_px),
            x1=min(limite.largura, self.x1 + margem_px),
            y1=min(limite.altura, self.y1 + margem_px),
        )

    def deslocar(self, dx: int, dy: int) -> Retangulo:
        return Retangulo(x0=self.x0 + dx, y0=self.y0 + dy, x1=self.x1 + dx, y1=self.y1 + dy)


class RetanguloNormalizado(BaseModel):
    """Saida da LLM. Espaco 0..1000, relativo a imagem enviada."""

    x0: int = Field(ge=0, le=1000)
    y0: int = Field(ge=0, le=1000)
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)

    def para_pixels(self, largura: int, altura: int) -> Retangulo:
        return Retangulo(
            x0=round(self.x0 / 1000 * largura),
            y0=round(self.y0 / 1000 * altura),
            x1=round(self.x1 / 1000 * largura),
            y1=round(self.y1 / 1000 * altura),
        )


class Tile(BaseModel):
    indice: int
    caminho: Path
    origem: tuple[int, int]  # (x, y) do canto superior esquerdo no ORIGINAL
    tamanho: tuple[int, int]  # (largura, altura) do tile em px
    linha: int
    coluna: int


class DescricaoTile(BaseModel):
    indice_tile: int
    resumo: str  # 1-2 frases sobre o conteudo geral
    elementos: list[str] = Field(default_factory=list)  # elementos visuais identificados
    texto_visivel: list[str] = Field(default_factory=list)  # textos legiveis no tile


class DescricaoMelhorada(BaseModel):
    descricao_melhorada: str
    criterios_exclusao: list[str] = Field(default_factory=list)
    termos_chave: list[str] = Field(default_factory=list)


class ItemRanking(BaseModel):
    indice_tile: int
    pontuacao: float = Field(ge=0.0, le=1.0)
    justificativa: str


class Ranking(BaseModel):
    itens: list[ItemRanking]  # ordenado por pontuacao desc


class ResultadoLocalizacao(BaseModel):
    encontrado: bool
    caixa: RetanguloNormalizado | None = None
    confianca: float = Field(ge=0.0, le=1.0, default=0.0)
    observacao: str = ""


class Veredito(BaseModel):
    aprovado: bool
    confianca: float = Field(ge=0.0, le=1.0)
    feedback: str  # sempre preenchido; se reprovado, acionavel


class MetricasEtapa(BaseModel):
    chamadas: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    custo_usd: float = 0.0
    tempo_s: float = 0.0


class Metricas(BaseModel):
    chamadas_llm: int = 0
    tokens_entrada: int = 0
    tokens_saida: int = 0
    tokens_cache_leitura: int = 0
    custo_usd: float = 0.0
    custo_brl: float | None = None
    tempo_total_s: float = 0.0
    por_etapa: dict[str, MetricasEtapa] = Field(default_factory=dict)


class ResultadoDeteccao(BaseModel):
    sucesso: bool
    caminho_imagem: Path | None = None
    caixa_no_original: Retangulo | None = None
    descricao_original: str
    descricao_melhorada: str
    tiles_avaliados: int
    tentativas_refino: int
    confianca_final: float = 0.0
    mensagem: str = ""  # motivo da falha, se sucesso=False
    metricas: Metricas
    run_id: str
