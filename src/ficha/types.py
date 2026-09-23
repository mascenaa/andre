"""Tipos compartilhados entre as camadas. Este módulo é o contrato do projeto.

Regras:
- Tudo aqui é imutável (``frozen=True``) exceto :class:`ExtractionRecord`, que é o registro
  bruto persistido e preenchido em etapas.
- Nenhuma camada importa de outra camada "lateral"; todas importam daqui.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ficha.schema import FichaExtraida

# --------------------------------------------------------------------------- documentos


@dataclass(frozen=True, slots=True)
class Page:
    """Uma página de um artigo, já com o texto limpo (ver ``ingest.clean``)."""

    number: int
    """Número da página, 1-based, como aparece no visualizador de PDF."""
    text: str

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError(f"Página deve ser >= 1, recebido {self.number}")


@dataclass(frozen=True, slots=True)
class Document:
    """Um artigo inteiro, página a página."""

    arquivo: str
    """Nome do arquivo PDF (ex.: ``artigo_07.pdf``). É a chave de tudo."""
    pages: tuple[Page, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Texto completo, páginas separadas por dupla quebra de linha."""
        return "\n\n".join(p.text for p in self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def page(self, number: int) -> Page:
        """Página pelo número 1-based."""
        for p in self.pages:
            if p.number == number:
                return p
        raise KeyError(f"{self.arquivo}: página {number} não existe (1..{self.n_pages})")

    def sha256(self) -> str:
        """Hash do conteúdo, para rastrear que texto exato foi usado em cada execução."""
        return hashlib.sha256(self.full_text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- seleção de contexto


@dataclass(frozen=True, slots=True)
class Chunk:
    """Um pedaço de texto de uma página, com sua origem rastreável."""

    arquivo: str
    page: int
    text: str
    start: int
    """Offset inicial dentro de ``Page.text``."""
    end: int
    """Offset final (exclusivo) dentro de ``Page.text``."""
    score: float | None = None
    """Pontuação da estratégia (similaridade, posição...). ``None`` quando não se aplica."""
    label: str | None = None
    """Rótulo livre da estratégia (ex.: a consulta semântica ou a palavra-chave que casou)."""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Offsets inválidos: start={self.start} end={self.end}")


@dataclass(frozen=True, slots=True)
class Context:
    """O que efetivamente vai para o modelo sobre um artigo, e de onde veio."""

    arquivo: str
    strategy: str
    """Nome da estratégia de seleção (ex.: ``first_pages``, ``keyword``, ``semantic``)."""
    chunks: tuple[Chunk, ...]
    params: dict[str, Any] = field(default_factory=dict)
    """Parâmetros da estratégia (n páginas, tamanho do chunk, sobreposição, k...)."""

    PAGE_MARK: str = field(default="[p. {page}]", init=False, repr=False)

    def render(self) -> str:
        """Texto enviado ao modelo. Cada chunk é precedido de um marcador de página.

        O marcador é o que permite ao modelo devolver ``evidencia.pagina`` correto e ao
        auditor verificar fidelidade contra **exatamente** este texto.
        """
        parts: list[str] = []
        for c in self.chunks:
            parts.append(f"{self.PAGE_MARK.format(page=c.page)}\n{c.text.strip()}")
        return "\n\n".join(parts)

    @property
    def n_chars(self) -> int:
        return sum(len(c.text) for c in self.chunks)

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({c.page for c in self.chunks}))


@runtime_checkable
class ContextSelector(Protocol):
    """Estratégia da Seção 4.1: decide que pedaço de cada artigo é enviado."""

    @property
    def name(self) -> str: ...

    def select(self, doc: Document) -> Context: ...


# --------------------------------------------------------------------------- modelo de linguagem


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Parâmetros de geração. ``temperature`` e ``seed`` são declarados e rastreados."""

    temperature: float = 0.0
    max_new_tokens: int = 1024
    seed: int | None = 42
    top_p: float = 1.0


@dataclass(frozen=True, slots=True)
class Usage:
    """Tokens consumidos em uma chamada — base do cálculo de custo (Seção 4.5)."""

    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True, slots=True)
class Completion:
    """Resposta bruta do modelo. ``text`` é guardado sem nenhum pós-processamento."""

    text: str
    usage: Usage
    model: str
    latency_s: float
    params: GenerationParams
    raw: dict[str, Any] = field(default_factory=dict)
    """Payload original do provedor, quando houver (para depuração)."""


@runtime_checkable
class LLMClient(Protocol):
    """Contrato mínimo de um cliente de modelo. Implementações em ``ficha.llm``."""

    @property
    def model_name(self) -> str: ...

    def generate(self, system: str | None, user: str, params: GenerationParams) -> Completion: ...

    def count_tokens(self, text: str) -> int:
        """Conta tokens com o tokenizador do próprio modelo (ou aproximação declarada)."""
        ...


# --------------------------------------------------------------------------- prompts


@dataclass(frozen=True, slots=True)
class PromptFeatures:
    """Técnicas da Seção 4.2, cada uma ligável/desligável de forma identificável."""

    role: bool = True
    delimiters: bool = True
    few_shot: bool = True
    abstention: bool = True
    chain_of_thought: bool = False

    def describe(self) -> str:
        on = [k for k, v in asdict(self).items() if v]
        off = [k for k, v in asdict(self).items() if not v]
        return f"on={on} off={off}"


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Prompt pronto para envio, com a variante e as técnicas registradas."""

    variant: str
    """Identificador da versão (ex.: ``v_full``, ``v_sem_fewshot``)."""
    features: PromptFeatures
    system: str | None
    user: str

    def sha256(self) -> str:
        return hashlib.sha256(f"{self.system}\n{self.user}".encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- extração


class ParseStatus(StrEnum):
    """Resultado do parse da saída do modelo."""

    OK = "ok"
    """JSON válido e conforme o schema de primeira."""
    REPAIRED = "repaired"
    """Precisou de reparo (cercas de código, vírgula sobrando, sentinela de null...)."""
    FAILED = "failed"
    """Não foi possível obter uma ficha válida."""


@dataclass(slots=True)
class ExtractionRecord:
    """Registro bruto de UMA chamada ao modelo para UM artigo. É o que se persiste.

    Sem isso não há auditoria de estabilidade (Seção 7, Reprodutibilidade). O ``context_text``
    é guardado na íntegra porque a verificação de fidelidade (4.4a) compara o trecho devolvido
    com **o texto que foi enviado**, não com o artigo inteiro.
    """

    run_id: str
    arquivo: str
    strategy: str
    prompt_variant: str
    prompt_sha: str
    model: str
    temperature: float
    seed: int | None
    repetition: int
    """1 para a primeira execução, 2 para a repetição de estabilidade, etc."""
    context_text: str
    context_pages: tuple[int, ...]
    raw_output: str
    usage: Usage
    latency_s: float
    json_valid_first_try: bool
    parse_status: ParseStatus
    ficha: FichaExtraida | None
    error: str | None = None
    repairs: list[str] = field(default_factory=list)
    """Reparos aplicados pelo parser (vazio quando ``parse_status == OK``)."""
    doc_sha256: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        d["parse_status"] = self.parse_status.value
        d["ficha"] = self.ficha.model_dump() if self.ficha else None
        d["context_pages"] = list(self.context_pages)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractionRecord:
        d = dict(d)
        d["usage"] = Usage(**d["usage"])
        d["parse_status"] = ParseStatus(d["parse_status"])
        d["ficha"] = FichaExtraida.model_validate(d["ficha"]) if d.get("ficha") else None
        d["context_pages"] = tuple(d.get("context_pages", ()))
        d.setdefault("repairs", [])
        return cls(**d)


def group_by_arquivo(records: Iterable[ExtractionRecord]) -> dict[str, list[ExtractionRecord]]:
    """Agrupa registros por artigo, preservando a ordem de chegada."""
    out: dict[str, list[ExtractionRecord]] = {}
    for r in records:
        out.setdefault(r.arquivo, []).append(r)
    return out


def sorted_arquivos(records: Sequence[ExtractionRecord]) -> list[str]:
    """Nomes de arquivo únicos, ordenados."""
    return sorted({r.arquivo for r in records})
