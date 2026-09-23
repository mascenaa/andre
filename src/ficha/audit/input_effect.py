"""Efeito da entrada (Seção 4.4c): mesmos artigos, duas estratégias de seleção (Seção 4.1).

Responde "onde as fichas discordam?" (``DiffReport`` + :meth:`InputEffectReport.disagreements`)
e "qual estratégia você defenderia, e com base em quê?" (:meth:`InputEffectReport.recommend`),
com critério declarado:

1. maior taxa de fidelidade (trecho encontrado no texto enviado);
2. empate → maior taxa de página correta;
3. empate → menos erros de parse (maior taxa de fichas válidas);
4. empate → menos tokens de entrada (mais barata).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ficha.audit.diff import DiffReport, FieldDiff, diff_runs
from ficha.audit.fidelity import DEFAULT_FIDELITY_THRESHOLD, FidelitySummary, fidelity_summary
from ficha.audit.runstats import RunStats, label_of, run_stats
from ficha.audit.stability import config_warnings
from ficha.types import ExtractionRecord

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class StrategyVerdict:
    """Estratégia recomendada e o critério que decidiu."""

    strategy: str
    decided_by: str | None
    """Critério que desempatou; ``None`` = empate total (fica a estratégia A)."""
    explanation: str


@dataclass(frozen=True, slots=True)
class InputEffectReport:
    """Resultado de 4.4c."""

    strategy_a: str
    strategy_b: str
    diff: DiffReport
    changed_fields_by_arquivo: dict[str, int]
    fidelity_a: FidelitySummary
    fidelity_b: FidelitySummary
    stats_a: RunStats
    stats_b: RunStats
    warnings: list[str] = field(default_factory=list)

    def disagreements(self, min_similarity: float = 0.8) -> list[FieldDiff]:
        """Campos em que as estratégias discordam de forma substantiva (similaridade < limite)."""
        return self.diff.disagreements(min_similarity)

    def fidelity_comparison(self) -> dict[str, float]:
        """Taxas de fidelidade lado a lado e a diferença (B - A)."""
        return {
            f"rate_found[{self.strategy_a}]": self.fidelity_a.rate_found,
            f"rate_found[{self.strategy_b}]": self.fidelity_b.rate_found,
            "delta_rate_found(b-a)": self.fidelity_b.rate_found - self.fidelity_a.rate_found,
            f"rate_page_ok[{self.strategy_a}]": self.fidelity_a.rate_page_ok,
            f"rate_page_ok[{self.strategy_b}]": self.fidelity_b.rate_page_ok,
            "delta_rate_page_ok(b-a)": self.fidelity_b.rate_page_ok - self.fidelity_a.rate_page_ok,
        }

    def recommend(self) -> StrategyVerdict:
        """Aplica o critério declarado no docstring do módulo."""
        a, b = self.stats_a, self.stats_b
        criteria: list[tuple[str, float, float]] = [
            ("maior taxa de fidelidade", a.rate_fidelity, b.rate_fidelity),
            ("maior taxa de página correta", a.rate_page_ok, b.rate_page_ok),
            ("menos falhas de parse", -a.rate_parse_failed, -b.rate_parse_failed),
            ("menos tokens de entrada", -float(a.input_tokens), -float(b.input_tokens)),
        ]
        for name, va, vb in criteria:
            if abs(va - vb) > _EPS:
                win = self.strategy_a if va > vb else self.strategy_b
                return StrategyVerdict(
                    win, name, f"{win} vence por {name} ({abs(va):.3g} vs {abs(vb):.3g})."
                )
        return StrategyVerdict(
            self.strategy_a, None, "Empate em todos os critérios; mantém-se a estratégia A."
        )

    def to_frame(self) -> pd.DataFrame:
        """Tabela lado a lado das duas estratégias (uma linha por métrica)."""
        da, db = self.stats_a.to_dict(), self.stats_b.to_dict()
        keys = [k for k in da if k not in ("variant", "null_ground_truth")]
        return pd.DataFrame(
            {self.strategy_a: [da[k] for k in keys], self.strategy_b: [db[k] for k in keys]},
            index=keys,
        )

    def to_dict(self) -> dict[str, Any]:
        v = self.recommend()
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "diff": self.diff.to_dict(),
            "fidelity": self.fidelity_comparison(),
            "stats_a": self.stats_a.to_dict(),
            "stats_b": self.stats_b.to_dict(),
            "n_disagreements": len(self.disagreements()),
            "recommended": v.strategy,
            "recommended_by": v.decided_by,
            "recommendation": v.explanation,
            "warnings": list(self.warnings),
        }


def input_effect_report(
    run_strategy_a: Sequence[ExtractionRecord],
    run_strategy_b: Sequence[ExtractionRecord],
    *,
    threshold: float = DEFAULT_FIDELITY_THRESHOLD,
    limitacao_gabarito: Mapping[str, bool] | None = None,
) -> InputEffectReport:
    """Compara duas estratégias de seleção sobre os mesmos artigos."""
    sa = label_of(run_strategy_a, "strategy", "A")
    sb = label_of(run_strategy_b, "strategy", "B")
    fa = fidelity_summary(run_strategy_a, threshold)
    fb = fidelity_summary(run_strategy_b, threshold)
    d = diff_runs(run_strategy_a, run_strategy_b, label_a=sa, label_b=sb)
    return InputEffectReport(
        strategy_a=sa,
        strategy_b=sb,
        diff=d,
        changed_fields_by_arquivo=d.changed_fields_by_arquivo,
        fidelity_a=fa,
        fidelity_b=fb,
        stats_a=run_stats(run_strategy_a, sa, limitacao_gabarito=limitacao_gabarito, fidelity=fa),
        stats_b=run_stats(run_strategy_b, sb, limitacao_gabarito=limitacao_gabarito, fidelity=fb),
        warnings=config_warnings(
            run_strategy_a, run_strategy_b, ("prompt_variant", "model", "temperature")
        ),
    )
