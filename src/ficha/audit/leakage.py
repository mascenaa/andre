"""Vazamento de exemplos few-shot (verificação 4.4e): o modelo copiou o exemplo do prompt.

A fidelidade (4.4a) não pega este erro quando o campo copiado não é o trecho — e a
heurística de limitação também não, se o contexto tiver vocabulário de limitação. Caso real
(execução híbrida, 2026-09-23): em 06_Barnes_2016 (explosões solares) o modelo devolveu como
``limitacao`` a limitação do exemplo sobre fraude em cartão ("os rótulos vêm de chargebacks...").
É a "invenção convincente" que a rubrica premia detectar.

Método (declarado no ADR 0005):

- Para cada campo textual da ficha (``problema, dados, metodo, metrica, limitacao,
  evidencia.trecho``), compara-se com o campo **homólogo** de cada exemplo, após
  :func:`normalize_for_match`.
- Similaridade = máximo entre ``fuzz.ratio/100`` e ``fuzz.partial_ratio/100``; o método que deu
  o máximo é registrado. ``partial_ratio`` só é usado quando a string mais curta tem pelo menos
  :data:`MIN_PARTIAL_LEN` caracteres — com strings curtas (ex.: "AUPRC") ele daria 1.0 por
  mera coincidência de sigla.
- ``evidencia.trecho`` também é comparado com o **texto de entrada** do exemplo
  (``partial_ratio``): copiar qualquer frase do exemplo, não só o trecho dele, é vazamento.
- ``None`` nunca vaza (abster-se não é copiar).
- Vazou se similaridade ≥ ``threshold`` (0.80). Na execução real, campos legítimos de artigos
  diferentes ficaram em ≈0.4–0.5 contra os exemplos; a cópia de Barnes deu 0.93.

Segundo método, para **contaminação parcial** (que a similaridade do campo inteiro não pega):
termos de domínio característicos dos exemplos (:data:`EXAMPLE_MARKER_TERMS` — fraude,
chargeback, supermercados, WMAPE, suavização exponencial...) que aparecem na ficha mas cujo
equivalente em inglês **não** aparece no texto enviado. Caso real: 09_AsensioRamos (física
solar) com dados "magnetogramas ... de supermercados brasileiros" — similaridade só 0.77, mas
"supermercados" sem "supermarket" no contexto é inequívoco. Os exemplos foram escritos em
domínios diferentes dos artigos justamente para que esses termos sejam marcadores.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd
from rapidfuzz import fuzz

from ficha.audit.normalize import normalize_for_match, split_context_blocks
from ficha.prompts import FEW_SHOT_EXAMPLES, FewShotExample
from ficha.schema import FichaExtraida
from ficha.types import ExtractionRecord

DEFAULT_LEAKAGE_THRESHOLD = 0.80
"""Similaridade mínima com um campo do exemplo para considerar cópia."""

MIN_PARTIAL_LEN = 40
"""Tamanho mínimo (caracteres normalizados) da string mais curta para usar ``partial_ratio``."""

TEXT_FIELDS: tuple[str, ...] = (
    "problema",
    "dados",
    "metodo",
    "metrica",
    "limitacao",
    "evidencia.trecho",
)
"""Campos textuais comparados com os exemplos."""

LeakMethod = Literal["ratio", "partial_ratio", "partial_ratio_contexto", "termo_do_exemplo"]


@dataclass(frozen=True, slots=True)
class MarkerTerm:
    """Termo característico de um exemplo few-shot (em pt, como a ficha) e seu par em inglês.

    Vaza se ``pattern_pt`` casa com um campo da ficha e ``pattern_en`` NÃO casa com o texto
    enviado (se o artigo fala de fraude, "fraude" na ficha é legítimo).
    """

    pattern_pt: str
    pattern_en: str
    example_name: str


EXAMPLE_MARKER_TERMS: tuple[MarkerTerm, ...] = (
    MarkerTerm(r"\bfraude", r"\bfraud", "com_limitacao"),
    MarkerTerm(r"\bchargebacks?\b", r"\bchargebacks?\b", "com_limitacao"),
    MarkerTerm(r"cart[aã]o de cr[eé]dito", r"credit.card", "com_limitacao"),
    MarkerTerm(r"\bemissor", r"\bissuer", "com_limitacao"),
    MarkerTerm(r"\btransa[cç](?:[aã]o|[oõ]es)\b", r"\btransactions?\b", "com_limitacao"),
    MarkerTerm(r"\bsupermercados?\b", r"\bsupermarkets?\b", "sem_limitacao"),
    MarkerTerm(r"\bvarej", r"\bretail", "sem_limitacao"),
    MarkerTerm(r"\bvendas\b", r"\bsales\b", "sem_limitacao"),
    MarkerTerm(r"\bestoques?\b", r"\binventor(?:y|ies)\b", "sem_limitacao"),
    MarkerTerm(r"\bpromo[cç](?:[aã]o|[oõ]es)\b", r"\bpromotions?\b", "sem_limitacao"),
    MarkerTerm(r"\bferiados?\b", r"\bholidays?\b", "sem_limitacao"),
    MarkerTerm(r"\bwmape\b", r"\bwmape\b", "sem_limitacao"),
    MarkerTerm(r"suaviza[cç][aã]o exponencial", r"exponential smoothing", "sem_limitacao"),
)
"""Marcadores de domínio dos exemplos de ``ficha.prompts.examples`` (fraude em cartão; varejo).
Se os exemplos mudarem, esta lista precisa acompanhar (há teste que confere que cada termo
existe no seu exemplo)."""


def _text_value(ficha: FichaExtraida, name: str) -> str | None:
    if name == "evidencia.trecho":
        return ficha.evidencia.trecho
    value: str | None = getattr(ficha, name)
    return value


@dataclass(frozen=True, slots=True)
class LeakedField:
    """Um campo da ficha que reproduz o exemplo few-shot."""

    field: str
    example_index: int
    """Índice do exemplo em ``examples`` (0 = primeiro exemplo do prompt)."""
    example_name: str
    similarity: float
    method: LeakMethod
    example_value: str
    """Texto do exemplo com que o campo casou (campo homólogo ou, p/ trecho, a entrada)."""
    value: str
    """Texto devolvido pelo modelo."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LeakageResult:
    """Resultado da verificação de vazamento de UMA ficha."""

    arquivo: str
    run_id: str
    leaked: list[LeakedField] = field(default_factory=list)
    threshold: float = DEFAULT_LEAKAGE_THRESHOLD
    max_similarity: dict[str, float] = field(default_factory=dict)
    """Maior similaridade de cada campo com qualquer exemplo (vazou ou não) — para calibrar."""

    leaked_terms: list[LeakedField] = field(default_factory=list)
    """Contaminação parcial: termos de domínio do exemplo ausentes do texto enviado
    (``method="termo_do_exemplo"``, ``similarity`` = 1.0 por convenção, ``example_value`` =
    o termo casado)."""

    @property
    def any_leak(self) -> bool:
        return bool(self.leaked or self.leaked_terms)

    @property
    def leaked_fields(self) -> list[str]:
        """Campos com qualquer vazamento, sem repetição, na ordem de :data:`TEXT_FIELDS`."""
        hit = {lf.field for lf in (*self.leaked, *self.leaked_terms)}
        return [f for f in TEXT_FIELDS if f in hit]

    @property
    def all_leaks(self) -> list[LeakedField]:
        """Vazamentos por similaridade seguidos dos por termo."""
        return [*self.leaked, *self.leaked_terms]


def _similarity(value: str, example: str) -> tuple[float, LeakMethod]:
    """Máximo entre ``ratio`` e (se aplicável) ``partial_ratio``, com o método vencedor."""
    r = fuzz.ratio(value, example) / 100.0
    if min(len(value), len(example)) >= MIN_PARTIAL_LEN:
        p = fuzz.partial_ratio(value, example) / 100.0
        if p > r:
            return p, "partial_ratio"
    return r, "ratio"


def _term_leaks(
    record: ExtractionRecord,
    ficha: FichaExtraida,
    examples: Sequence[FewShotExample],
    markers: Sequence[MarkerTerm],
) -> list[LeakedField]:
    """Termos de domínio do exemplo na ficha, sem equivalente no texto enviado."""
    names = [ex.name for ex in examples]
    out: list[LeakedField] = []
    for name in TEXT_FIELDS:
        raw = _text_value(ficha, name)
        if raw is None:
            continue
        for mk in markers:
            m = re.search(mk.pattern_pt, raw, re.IGNORECASE)
            if m is None or re.search(mk.pattern_en, record.context_text, re.IGNORECASE):
                continue
            out.append(
                LeakedField(
                    field=name,
                    example_index=names.index(mk.example_name) if mk.example_name in names else -1,
                    example_name=mk.example_name,
                    similarity=1.0,
                    method="termo_do_exemplo",
                    example_value=m.group(0),
                    value=raw,
                )
            )
    return out


def check_fewshot_leakage(
    record: ExtractionRecord,
    examples: Sequence[FewShotExample] | None = None,
    threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    marker_terms: Sequence[MarkerTerm] | None = None,
) -> LeakageResult:
    """Compara cada campo textual da ficha com os exemplos few-shot (ver docstring do módulo).

    Para cada campo registra-se no máximo um :class:`LeakedField` por similaridade: o do
    exemplo mais parecido. ``marker_terms`` padrão: :data:`EXAMPLE_MARKER_TERMS` quando
    ``examples`` é o padrão; nenhum quando exemplos customizados são passados (os marcadores
    são específicos dos exemplos do projeto).
    """
    exs = FEW_SHOT_EXAMPLES if examples is None else examples
    if marker_terms is None:
        marker_terms = EXAMPLE_MARKER_TERMS if examples is None else ()
    if record.ficha is None:
        return LeakageResult(record.arquivo, record.run_id, [], threshold, {})

    leaked: list[LeakedField] = []
    best_by_field: dict[str, float] = {}
    for name in TEXT_FIELDS:
        raw = _text_value(record.ficha, name)
        if raw is None:
            continue
        value = normalize_for_match(raw)
        best: LeakedField | None = None
        for i, ex in enumerate(exs):
            candidates: list[tuple[str, float, LeakMethod]] = []
            ex_raw = _text_value(ex.output, name)
            if ex_raw is not None:
                sim, method = _similarity(value, normalize_for_match(ex_raw))
                candidates.append((ex_raw, sim, method))
            if name == "evidencia.trecho" and len(value) >= MIN_PARTIAL_LEN:
                ctx = " ".join(t for _, t in split_context_blocks(ex.context_text))
                sim = fuzz.partial_ratio(value, normalize_for_match(ctx)) / 100.0
                candidates.append((ctx, sim, "partial_ratio_contexto"))
            for ex_value, sim, method in candidates:
                if best is None or sim > best.similarity:
                    best = LeakedField(
                        field=name,
                        example_index=i,
                        example_name=ex.name,
                        similarity=round(sim, 4),
                        method=method,
                        example_value=ex_value,
                        value=raw,
                    )
        if best is not None:
            best_by_field[name] = best.similarity
            if best.similarity >= threshold:
                leaked.append(best)
    terms = _term_leaks(record, record.ficha, exs, marker_terms)
    return LeakageResult(record.arquivo, record.run_id, leaked, threshold, best_by_field, terms)


@dataclass(frozen=True, slots=True)
class LeakageSummary:
    """Agregado de vazamento sobre uma execução. Denominador: todos os registros."""

    n: int
    n_with_leak: int
    rate: float
    by_field: dict[str, int]
    """Nº de fichas com vazamento em cada campo (todas as chaves de :data:`TEXT_FIELDS`)."""
    results: list[LeakageResult]

    def leaks(self) -> list[LeakageResult]:
        """Somente as fichas com algum campo vazado."""
        return [r for r in self.results if r.any_leak]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_with_leak": self.n_with_leak,
            "rate": self.rate,
            "by_field": dict(self.by_field),
            "leaks": [
                {"arquivo": r.arquivo, "run_id": r.run_id, "fields": r.leaked_fields}
                for r in self.leaks()
            ],
        }

    def to_frame(self) -> pd.DataFrame:
        """Uma linha por vazamento (similaridade ou termo); vazio se nada vazou."""
        cols = [
            "arquivo",
            "run_id",
            "field",
            "example_index",
            "example_name",
            "similarity",
            "method",
            "value",
            "example_value",
        ]
        rows = [
            {"arquivo": r.arquivo, "run_id": r.run_id, **lf.to_dict()}
            for r in self.results
            for lf in r.all_leaks
        ]
        return pd.DataFrame(rows, columns=cols)


def leakage_summary(
    records: Sequence[ExtractionRecord],
    examples: Sequence[FewShotExample] | None = None,
    threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    marker_terms: Sequence[MarkerTerm] | None = None,
) -> LeakageSummary:
    """Aplica :func:`check_fewshot_leakage` a cada registro e agrega."""
    results = [check_fewshot_leakage(r, examples, threshold, marker_terms) for r in records]
    n = len(results)
    n_leak = sum(r.any_leak for r in results)
    counts = Counter(f for r in results for f in r.leaked_fields)
    return LeakageSummary(
        n=n,
        n_with_leak=n_leak,
        rate=n_leak / n if n else 0.0,
        by_field={f: counts.get(f, 0) for f in TEXT_FIELDS},
        results=results,
    )
