"""Medição obrigatória da Seção 4.2: duas versões do prompt sobre os mesmos artigos.

Para cada variante: taxa de JSON válido de primeira, taxas de parse (ok / reparado / falhou),
taxa de ``limitacao`` null, taxa de null **quando devido**, e taxa de fidelidade. Entre elas:
um :class:`~ficha.audit.diff.DiffReport` ("as fichas discordam, e em quê?").

A escolha da versão final **não é por intuição**: :meth:`PromptComparison.winner` aplica um
critério lexicográfico declarado (:data:`DEFAULT_WINNER_CRITERIA`):

1. maior taxa de JSON válido de primeira;
2. empate → maior taxa de fidelidade;
3. empate → maior taxa de null quando devido (abstenção correta).

Empate em todos → fica a variante A (por convenção, a de referência), e a explicação diz
explicitamente que "a técnica não mudou nada" nestes números.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ficha.audit.diff import DiffReport, FieldDiff, diff_runs
from ficha.audit.fidelity import DEFAULT_FIDELITY_THRESHOLD, FidelitySummary, fidelity_summary
from ficha.audit.runstats import RunStats, label_of, run_stats
from ficha.types import ExtractionRecord

DEFAULT_WINNER_CRITERIA: tuple[str, ...] = (
    "rate_json_valid_first_try",
    "rate_fidelity",
    "rate_null_when_expected",
)
"""Critérios de desempate, em ordem. Nomes de atributos de :class:`RunStats` (maior é melhor)."""

_EPS = 1e-9

_METRIC_ROWS: tuple[str, ...] = (
    "n",
    "rate_json_valid_first_try",
    "rate_parse_ok",
    "rate_parse_repaired",
    "rate_parse_failed",
    "rate_limitacao_null",
    "rate_null_when_expected",
    "n_null_expected",
    "rate_fidelity",
    "rate_page_ok",
    "input_tokens",
    "output_tokens",
)


@dataclass(frozen=True, slots=True)
class PromptVerdict:
    """Variante vencedora, critério que decidiu e explicação legível para o relatório."""

    winner: str
    decided_by: str | None
    """Critério decisivo; ``None`` = empate em todos (fica a variante A)."""
    explanation: str
    values: dict[str, tuple[float | None, float | None]]
    """``{critério: (valor A, valor B)}`` dos critérios avaliados."""


@dataclass(frozen=True, slots=True)
class PromptComparison:
    """Comparação entre duas variantes de prompt."""

    a: RunStats
    b: RunStats
    fidelity_a: FidelitySummary
    fidelity_b: FidelitySummary
    diff: DiffReport

    @property
    def variants(self) -> tuple[str, str]:
        return self.a.variant, self.b.variant

    def disagreements(self, min_similarity: float = 0.8) -> list[FieldDiff]:
        """Campos em que as fichas das duas variantes discordam de forma substantiva."""
        return self.diff.disagreements(min_similarity)

    def to_frame(self) -> pd.DataFrame:
        """Tabela lado a lado: uma linha por métrica; colunas = variantes + ``delta (b-a)``."""
        da, db = self.a.to_dict(), self.b.to_dict()
        deltas: list[float | None] = []
        for k in _METRIC_ROWS:
            va, vb = da[k], db[k]
            deltas.append(None if va is None or vb is None else float(vb) - float(va))
        df = pd.DataFrame(
            {
                self.a.variant: [da[k] for k in _METRIC_ROWS],
                self.b.variant: [db[k] for k in _METRIC_ROWS],
                "delta (b-a)": deltas,
            },
            index=list(_METRIC_ROWS),
        )
        df.loc["rate_identical_entre_variantes"] = [
            self.diff.rate_identical,
            self.diff.rate_identical,
            None,
        ]
        return df

    def decide(self, criteria: Sequence[str] = DEFAULT_WINNER_CRITERIA) -> PromptVerdict:
        """Aplica os critérios em ordem; o primeiro que diferencia decide.

        Um critério com valor ``None`` em algum lado (ex.: nenhum artigo sem limitação para
        medir abstenção) é registrado e pulado — não se decide com base em ausência de dados.
        """
        values: dict[str, tuple[float | None, float | None]] = {}
        for crit in criteria:
            va: float | None = getattr(self.a, crit)
            vb: float | None = getattr(self.b, crit)
            values[crit] = (va, vb)
            if va is None or vb is None or abs(va - vb) <= _EPS:
                continue
            win = self.a.variant if va > vb else self.b.variant
            return PromptVerdict(
                winner=win,
                decided_by=crit,
                explanation=(
                    f"{win} vence por {crit}: {va:.3f} ({self.a.variant}) vs "
                    f"{vb:.3f} ({self.b.variant}); critérios anteriores empataram."
                ),
                values=values,
            )
        return PromptVerdict(
            winner=self.a.variant,
            decided_by=None,
            explanation=(
                "Empate em todos os critérios: a técnica que difere entre as variantes não "
                f"mudou nada nestes números; mantém-se {self.a.variant} (referência)."
            ),
            values=values,
        )

    def winner(self, criteria: Sequence[str] = DEFAULT_WINNER_CRITERIA) -> str:
        """Nome da variante recomendada pela regra declarada (ver :meth:`decide`)."""
        return self.decide(criteria).winner

    def to_dict(self) -> dict[str, Any]:
        v = self.decide()
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "diff": self.diff.to_dict(),
            "n_disagreements": len(self.disagreements()),
            "winner": v.winner,
            "decided_by": v.decided_by,
            "explanation": v.explanation,
            "criteria": list(DEFAULT_WINNER_CRITERIA),
        }


def compare_prompt_variants(
    run_a: Sequence[ExtractionRecord],
    run_b: Sequence[ExtractionRecord],
    *,
    limitacao_gabarito: Mapping[str, bool] | None = None,
    threshold: float = DEFAULT_FIDELITY_THRESHOLD,
) -> PromptComparison:
    """Compara duas variantes de prompt rodadas sobre os mesmos artigos.

    ``limitacao_gabarito`` (``{arquivo: autores declaram limitação?}``) torna a taxa de null
    quando devido exata; sem ele, usa-se a heurística de vocabulário (declarada em
    :mod:`ficha.audit.runstats`).
    """
    va = label_of(run_a, "prompt_variant", "A")
    vb = label_of(run_b, "prompt_variant", "B")
    fa = fidelity_summary(run_a, threshold)
    fb = fidelity_summary(run_b, threshold)
    return PromptComparison(
        a=run_stats(run_a, va, limitacao_gabarito=limitacao_gabarito, fidelity=fa),
        b=run_stats(run_b, vb, limitacao_gabarito=limitacao_gabarito, fidelity=fb),
        fidelity_a=fa,
        fidelity_b=fb,
        diff=diff_runs(run_a, run_b, label_a=va, label_b=vb),
    )
