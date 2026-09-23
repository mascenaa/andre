"""Estatísticas de UMA execução: taxas de parse, abstenção e fidelidade.

Base comum de 4.2 (comparação de prompts), 4.4c (estratégias) e 4.4d (temperatura).

Denominadores (declarados):

- ``rate_json_valid_first_try``, ``rate_parse_*`` e ``rate_fidelity``: **todos** os registros
  da execução — uma saída que falhou conta contra a variante.
- ``rate_limitacao_null``: fichas válidas (sem ficha não há campo a observar).
- ``rate_null_when_expected``: fichas válidas de artigos em que a limitação **não** é
  declarada, segundo o gabarito (se fornecido) ou, na falta dele, a heurística de vocabulário
  de :func:`ficha.audit.fidelity.check_limitacao_support`. ``None`` se não há nenhum caso.
- ``rate_filled_when_unexpected``: o erro grave — fichas desses mesmos artigos com
  ``limitacao`` preenchida (``1 - rate_null_when_expected``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ficha.audit.fidelity import (
    DEFAULT_FIDELITY_THRESHOLD,
    FidelitySummary,
    check_limitacao_support,
    fidelity_summary,
)
from ficha.types import ExtractionRecord, ParseStatus

NullGroundTruth = Literal["gabarito", "heuristica_vocabulario"]


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True, slots=True)
class RunStats:
    """Números de uma execução (uma variante de prompt, uma estratégia ou uma temperatura)."""

    variant: str
    n: int
    rate_json_valid_first_try: float
    rate_parse_ok: float
    rate_parse_repaired: float
    rate_parse_failed: float
    rate_limitacao_null: float
    rate_null_when_expected: float | None
    n_null_expected: int
    null_ground_truth: NullGroundTruth
    rate_fidelity: float
    rate_page_ok: float
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_stats(
    records: Sequence[ExtractionRecord],
    variant: str,
    *,
    limitacao_gabarito: Mapping[str, bool] | None = None,
    threshold: float = DEFAULT_FIDELITY_THRESHOLD,
    fidelity: FidelitySummary | None = None,
) -> RunStats:
    """Calcula :class:`RunStats`.

    ``limitacao_gabarito``: ``{arquivo: True se os autores declaram limitação}``, conferido à
    mão. Quando ausente, usa-se a heurística de vocabulário (e isso fica registrado em
    ``null_ground_truth``).
    """
    n = len(records)
    fid = fidelity if fidelity is not None else fidelity_summary(records, threshold)
    parsed = [r for r in records if r.ficha is not None]
    n_null = sum(r.ficha is not None and r.ficha.limitacao is None for r in records)

    expected_null: list[ExtractionRecord] = []
    for r in parsed:
        if limitacao_gabarito is not None:
            if r.arquivo in limitacao_gabarito and not limitacao_gabarito[r.arquivo]:
                expected_null.append(r)
        elif not check_limitacao_support(r).vocab_in_context:
            expected_null.append(r)
    n_exp = len(expected_null)
    n_exp_null = sum(r.ficha is not None and r.ficha.limitacao is None for r in expected_null)

    def count(status: ParseStatus) -> int:
        return sum(r.parse_status == status for r in records)

    return RunStats(
        variant=variant,
        n=n,
        rate_json_valid_first_try=_rate(sum(r.json_valid_first_try for r in records), n),
        rate_parse_ok=_rate(count(ParseStatus.OK), n),
        rate_parse_repaired=_rate(count(ParseStatus.REPAIRED), n),
        rate_parse_failed=_rate(count(ParseStatus.FAILED), n),
        rate_limitacao_null=_rate(n_null, len(parsed)),
        rate_null_when_expected=n_exp_null / n_exp if n_exp else None,
        n_null_expected=n_exp,
        null_ground_truth="gabarito"
        if limitacao_gabarito is not None
        else "heuristica_vocabulario",
        rate_fidelity=fid.rate_found,
        rate_page_ok=fid.rate_page_ok,
        input_tokens=sum(r.usage.input_tokens for r in records),
        output_tokens=sum(r.usage.output_tokens for r in records),
    )


def restrict_to(records: Sequence[ExtractionRecord], arquivos: set[str]) -> list[ExtractionRecord]:
    """Somente os registros dos ``arquivos`` dados (para comparar sobre o mesmo subconjunto)."""
    return [r for r in records if r.arquivo in arquivos]


def label_of(records: Sequence[ExtractionRecord], attr: str, default: str) -> str:
    """Rótulo de uma execução a partir de um atributo comum (``prompt_variant``, ``strategy``)."""
    values = sorted({str(getattr(r, attr)) for r in records})
    return values[0] if len(values) == 1 else default
