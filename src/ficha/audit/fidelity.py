"""Verificação de fidelidade (Seção 4.4a): o ``evidencia.trecho`` existe no texto enviado.

Premissas declaradas (ver ADR 0005):

- A comparação é contra ``ExtractionRecord.context_text`` — o texto **exato** que o modelo
  recebeu — e não contra o PDF inteiro. Um trecho que existe no artigo mas não no contexto
  enviado não pode ter sido lido pelo modelo: ele foi "lembrado" ou inventado.
- Os dois lados passam por :func:`ficha.audit.normalize.normalize_for_match`.
- Primeiro tenta-se substring exata (``method="exact"``, ``score=1.0``). Se falhar, usa-se
  ``rapidfuzz.fuzz.partial_ratio`` (``method="fuzzy"``): a melhor similaridade entre o trecho
  e **qualquer janela do mesmo tamanho** do contexto. É a métrica certa porque o trecho deve
  ser uma subsequência contígua do contexto; ela tolera pequenas diferenças de limpeza
  (uma palavra hifenizada, uma vírgula) mas cai rápido com paráfrase ou invenção.
- ``found = score >= threshold`` com limiar padrão 0.90.
- A página é verificada separadamente: ``page_found`` é a página do bloco ``[p. N]`` onde o
  trecho foi localizado; ``page_ok`` exige que seja a página declarada pelo modelo.

Também há :func:`check_limitacao_support`, uma heurística **declarada** para o erro mais grave
do enunciado (limitação inventada): o campo ``limitacao`` veio preenchido, mas o texto enviado
não contém nenhum vocabulário de limitação?
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd
from rapidfuzz import fuzz

from ficha.audit.normalize import normalize_for_match, split_context_blocks
from ficha.types import ExtractionRecord

DEFAULT_FIDELITY_THRESHOLD = 0.90
"""Limiar padrão de ``partial_ratio`` (0-1) para considerar o trecho encontrado."""

FidelityMethod = Literal["exact", "fuzzy", "none"]


@dataclass(frozen=True, slots=True)
class FidelityResult:
    """Resultado da verificação de fidelidade de UMA ficha."""

    arquivo: str
    run_id: str
    found: bool
    """``True`` se o trecho foi localizado no contexto (exato ou ``score >= threshold``)."""
    score: float
    """Similaridade 0-1 (1.0 para exato; ``partial_ratio/100`` para fuzzy; 0.0 sem ficha)."""
    method: FidelityMethod
    page_claimed: int | None
    """``evidencia.pagina`` devolvida pelo modelo."""
    page_found: int | None
    """Página do bloco ``[p. N]`` onde o trecho foi localizado (``None`` se não encontrado)."""
    page_ok: bool
    """``page_found == page_claimed`` (e ambos definidos)."""
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Located:
    score: float
    page: int | None


class _NormalizedContext:
    """Contexto normalizado e concatenado, com o mapa offset → página de cada bloco."""

    def __init__(self, context_text: str) -> None:
        self.blocks: list[tuple[int, str]] = [
            (page, normalize_for_match(text)) for page, text in split_context_blocks(context_text)
        ]
        spans: list[tuple[int, int, int]] = []
        pieces: list[str] = []
        pos = 0
        for page, text in self.blocks:
            spans.append((pos, pos + len(text), page))
            pieces.append(text)
            pos += len(text) + 1  # separador " "
        self.text = " ".join(pieces)
        self._spans = spans

    def page_at(self, offset: int) -> int | None:
        """Página do bloco que contém ``offset`` (``None`` se o bloco não tem marcador)."""
        for start, end, page in self._spans:
            if start <= offset <= end:
                return page or None
        return None

    def page_texts(self, page: int) -> list[str]:
        return [t for p, t in self.blocks if p == page]


def _best_fuzzy(needle: str, ctx: _NormalizedContext) -> _Located:
    """Melhor ``partial_ratio`` do trecho contra o contexto inteiro, e a página onde caiu."""
    if not ctx.text:
        return _Located(0.0, None)
    if len(needle) <= len(ctx.text):
        al = fuzz.partial_ratio_alignment(needle, ctx.text)
        if al is None:  # só ocorre com score_cutoff, que não usamos
            return _Located(0.0, None)
        return _Located(al.score / 100.0, ctx.page_at(al.dest_start))
    # Trecho maior que o contexto inteiro: partial_ratio inverte os papéis; não há página.
    return _Located(fuzz.partial_ratio(needle, ctx.text) / 100.0, None)


def _page_score(needle: str, ctx: _NormalizedContext, page: int | None) -> float:
    """Melhor similaridade do trecho dentro dos blocos de uma página específica."""
    if page is None:
        return 0.0
    best = 0.0
    for text in ctx.page_texts(page):
        if needle in text:
            return 1.0
        best = max(best, fuzz.partial_ratio(needle, text) / 100.0)
    return best


def check_fidelity(
    record: ExtractionRecord, threshold: float = DEFAULT_FIDELITY_THRESHOLD
) -> FidelityResult:
    """Verifica se ``record.ficha.evidencia.trecho`` está em ``record.context_text``.

    Algoritmo (em ordem):

    1. Sem ficha (parse ``FAILED``) → ``found=False, method="none", score=0``.
    2. Trecho normalizado é substring do contexto normalizado → ``exact``, ``score=1.0``.
    3. Senão ``partial_ratio`` contra o contexto inteiro → ``fuzzy``;
       ``found = score >= threshold``.
    4. Página: se o trecho também está (≥ limiar) na página declarada, ``page_found`` é ela —
       um trecho que se repete em duas páginas não é penalizado. Caso contrário, é a página do
       bloco onde está a melhor ocorrência. Trecho não encontrado → ``page_found=None``.
    """
    if record.ficha is None:
        return FidelityResult(
            arquivo=record.arquivo,
            run_id=record.run_id,
            found=False,
            score=0.0,
            method="none",
            page_claimed=None,
            page_found=None,
            page_ok=False,
            threshold=threshold,
        )

    claimed = record.ficha.evidencia.pagina
    needle = normalize_for_match(record.ficha.evidencia.trecho)
    ctx = _NormalizedContext(record.context_text)

    method: FidelityMethod
    idx = ctx.text.find(needle) if needle else -1
    if idx >= 0:
        method, score, best_page = "exact", 1.0, ctx.page_at(idx)
    else:
        loc = _best_fuzzy(needle, ctx)
        method, score, best_page = "fuzzy", loc.score, loc.page

    found = score >= threshold
    page_found: int | None = None
    if found:
        page_found = claimed if _page_score(needle, ctx, claimed) >= threshold else best_page

    return FidelityResult(
        arquivo=record.arquivo,
        run_id=record.run_id,
        found=found,
        score=round(score, 4),
        method=method,
        page_claimed=claimed,
        page_found=page_found,
        page_ok=found and page_found is not None and page_found == claimed,
        threshold=threshold,
    )


def _rate(num: int, den: int) -> float:
    """Taxa ``num/den``; 0.0 quando não há denominador (declarado: sem dados, sem crédito)."""
    return num / den if den else 0.0


@dataclass(frozen=True, slots=True)
class FidelitySummary:
    """Agregado de fidelidade sobre uma execução (as 19 fichas, tipicamente).

    As taxas usam **todos** os registros como denominador: uma extração que falhou conta
    como ficha sem evidência verificável (não é descartada do total).
    """

    n: int
    n_found: int
    rate_found: float
    n_page_ok: int
    rate_page_ok: float
    by_method: dict[str, int]
    results: list[FidelityResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_found": self.n_found,
            "rate_found": self.rate_found,
            "n_page_ok": self.n_page_ok,
            "rate_page_ok": self.rate_page_ok,
            "by_method": dict(self.by_method),
        }

    def to_frame(self) -> pd.DataFrame:
        """Uma linha por ficha: arquivo, run_id, found, score, method, páginas, page_ok."""
        cols = [
            "arquivo",
            "run_id",
            "found",
            "score",
            "method",
            "page_claimed",
            "page_found",
            "page_ok",
            "threshold",
        ]
        return pd.DataFrame([r.to_dict() for r in self.results], columns=cols)

    def failures(self) -> list[FidelityResult]:
        """Fichas cujo trecho não foi encontrado (candidatas a trecho inventado)."""
        return [r for r in self.results if not r.found]


def fidelity_summary(
    records: Sequence[ExtractionRecord], threshold: float = DEFAULT_FIDELITY_THRESHOLD
) -> FidelitySummary:
    """Aplica :func:`check_fidelity` a cada registro e agrega."""
    results = [check_fidelity(r, threshold) for r in records]
    n = len(results)
    n_found = sum(r.found for r in results)
    n_page_ok = sum(r.page_ok for r in results)
    by_method = Counter(r.method for r in results)
    return FidelitySummary(
        n=n,
        n_found=n_found,
        rate_found=_rate(n_found, n),
        n_page_ok=n_page_ok,
        rate_page_ok=_rate(n_page_ok, n),
        by_method={m: by_method.get(m, 0) for m in ("exact", "fuzzy", "none")},
        results=results,
    )


# --------------------------------------------------------------------------- limitação

LIMITACAO_VOCAB_RE = re.compile(
    r"\blimitations?\b|\blimited\b|\bshortcomings?\b|\bdrawbacks?\b|\bcaveats?\b"
    r"|\bthreats? to (?:the )?validity\b|\bfuture (?:work|research)\b|\bwe acknowledge\b"
    r"|\bweakness(?:es)?\b",
    re.IGNORECASE,
)
"""Vocabulário que indica que os autores declaram alguma limitação (artigos em inglês)."""

LimitacaoVerdict = Literal[
    "null_ok", "null_suspeito", "preenchida_com_suporte", "preenchida_sem_suporte", "sem_ficha"
]


@dataclass(frozen=True, slots=True)
class LimitacaoCheck:
    """Heurística declarada sobre ``limitacao`` versus o texto enviado.

    - ``preenchida_sem_suporte``: o modelo preencheu ``limitacao``, mas o contexto não contém
      NENHUM termo de :data:`LIMITACAO_VOCAB_RE` → forte suspeita de limitação inventada.
    - ``null_suspeito``: ``limitacao`` é null, mas o contexto tem vocabulário de limitação →
      possível omissão (erro menos grave que inventar).
    - ``null_ok`` / ``preenchida_com_suporte``: coerente com o vocabulário do contexto.

    É um indício, não prova: vocabulário presente não garante que a limitação devolvida seja a
    dos autores. Por isso só **rebaixa** a confiança, nunca a eleva.
    """

    arquivo: str
    limitacao_is_null: bool | None
    vocab_in_context: bool
    keywords: tuple[str, ...]
    verdict: LimitacaoVerdict

    @property
    def supported(self) -> bool | None:
        """``False`` só no caso grave (preenchida sem suporte); ``None`` sem ficha."""
        if self.verdict == "sem_ficha":
            return None
        return self.verdict != "preenchida_sem_suporte"


def check_limitacao_support(record: ExtractionRecord) -> LimitacaoCheck:
    """Classifica ``limitacao`` contra o vocabulário de limitação do texto enviado."""
    kws = tuple(
        sorted({m.group(0).lower() for m in LIMITACAO_VOCAB_RE.finditer(record.context_text)})
    )
    vocab = bool(kws)
    if record.ficha is None:
        return LimitacaoCheck(record.arquivo, None, vocab, kws, "sem_ficha")
    is_null = record.ficha.limitacao is None
    verdict: LimitacaoVerdict
    if is_null:
        verdict = "null_suspeito" if vocab else "null_ok"
    else:
        verdict = "preenchida_com_suporte" if vocab else "preenchida_sem_suporte"
    return LimitacaoCheck(record.arquivo, is_null, vocab, kws, verdict)
