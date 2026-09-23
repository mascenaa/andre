"""Efeito da temperatura (Seção 4.4d): um subconjunto repetido com temperatura diferente.

A comparação é feita **sobre o mesmo subconjunto**: a execução em ``t0`` é restrita aos
artigos presentes na execução alternativa antes de calcular as taxas.

Critério declarado para ``escolha_t0_se_sustenta``: a temperatura escolhida (``t0``) tem
taxa de fidelidade **≥** e taxa de JSON válido de primeira **≥** as da temperatura
alternativa. Se a alternativa for estritamente melhor em qualquer uma das duas, a escolha
da Seção 4.2 não se sustenta com estes dados.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ficha.audit.diff import DiffReport, diff_runs
from ficha.audit.fidelity import DEFAULT_FIDELITY_THRESHOLD, fidelity_summary
from ficha.audit.runstats import RunStats, restrict_to, run_stats
from ficha.types import ExtractionRecord

CRITERIO_TEMPERATURA = (
    "A temperatura escolhida se sustenta se, no mesmo subconjunto de artigos, tiver taxa de "
    "fidelidade (trecho encontrado no texto enviado) maior ou igual e taxa de JSON válido de "
    "primeira maior ou igual às da temperatura alternativa."
)


def _temperature_of(records: Sequence[ExtractionRecord]) -> float | None:
    temps = {r.temperature for r in records}
    return temps.pop() if len(temps) == 1 else None


@dataclass(frozen=True, slots=True)
class TemperatureReport:
    """Resultado de 4.4d."""

    temperature_t0: float | None
    temperature_alt: float | None
    arquivos: list[str]
    """Subconjunto comparado."""
    stats_t0: RunStats
    stats_alt: RunStats
    diff: DiffReport
    criterio: str = CRITERIO_TEMPERATURA

    @property
    def rate_json_valid_first_try_t0(self) -> float:
        return self.stats_t0.rate_json_valid_first_try

    @property
    def rate_json_valid_first_try_alt(self) -> float:
        return self.stats_alt.rate_json_valid_first_try

    @property
    def rate_parse_ok_t0(self) -> float:
        return self.stats_t0.rate_parse_ok

    @property
    def rate_parse_ok_alt(self) -> float:
        return self.stats_alt.rate_parse_ok

    @property
    def rate_fidelity_t0(self) -> float:
        return self.stats_t0.rate_fidelity

    @property
    def rate_fidelity_alt(self) -> float:
        return self.stats_alt.rate_fidelity

    @property
    def escolha_t0_se_sustenta(self) -> bool:
        """Aplica :data:`CRITERIO_TEMPERATURA`."""
        return (
            self.rate_fidelity_t0 >= self.rate_fidelity_alt
            and self.rate_json_valid_first_try_t0 >= self.rate_json_valid_first_try_alt
        )

    def to_frame(self) -> pd.DataFrame:
        """Tabela lado a lado (uma linha por métrica), mais a taxa de fichas idênticas."""
        a, b = self.stats_t0.to_dict(), self.stats_alt.to_dict()
        keys = [k for k in a if k not in ("variant", "null_ground_truth")]
        col_a = f"t={self.temperature_t0}"
        col_b = f"t={self.temperature_alt}"
        df = pd.DataFrame({col_a: [a[k] for k in keys], col_b: [b[k] for k in keys]}, index=keys)
        df.loc["rate_identical_entre_temperaturas"] = [self.diff.rate_identical] * 2
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_t0": self.temperature_t0,
            "temperature_alt": self.temperature_alt,
            "arquivos": list(self.arquivos),
            "stats_t0": self.stats_t0.to_dict(),
            "stats_alt": self.stats_alt.to_dict(),
            "diff": self.diff.to_dict(),
            "escolha_t0_se_sustenta": self.escolha_t0_se_sustenta,
            "criterio": self.criterio,
        }


def temperature_report(
    run_t0: Sequence[ExtractionRecord],
    run_t_alt: Sequence[ExtractionRecord],
    *,
    threshold: float = DEFAULT_FIDELITY_THRESHOLD,
) -> TemperatureReport:
    """Compara a temperatura escolhida com a alternativa, no subconjunto de ``run_t_alt``."""
    subset = {r.arquivo for r in run_t_alt}
    t0 = restrict_to(run_t0, subset)
    tt = _temperature_of(t0)
    ta = _temperature_of(run_t_alt)
    la, lb = f"t={tt}", f"t={ta}"
    return TemperatureReport(
        temperature_t0=tt,
        temperature_alt=ta,
        arquivos=sorted(subset),
        stats_t0=run_stats(t0, la, fidelity=fidelity_summary(t0, threshold)),
        stats_alt=run_stats(run_t_alt, lb, fidelity=fidelity_summary(run_t_alt, threshold)),
        diff=diff_runs(t0, run_t_alt, label_a=la, label_b=lb),
    )
