"""Estabilidade (Seção 4.4b): a mesma extração, rodada duas vezes sem mudar nada.

Quantificação: taxa de fichas idênticas, taxa de mudança por campo, similaridade média por
campo e nº de campos que mudaram por artigo (este último alimenta a regra de confiança).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ficha.audit.diff import DiffReport, diff_runs
from ficha.types import ExtractionRecord

_CONFIG_ATTRS = ("strategy", "prompt_variant", "prompt_sha", "model", "temperature", "seed")


@dataclass(frozen=True, slots=True)
class StabilityReport:
    """Resultado de 4.4b. ``diff`` tem todos os detalhes; o resto são atalhos tipados."""

    diff: DiffReport
    changed_fields_by_arquivo: dict[str, int]
    """``{arquivo: nº de campos que mudaram entre as repetições}`` (só pares comparáveis)."""
    warnings: list[str] = field(default_factory=list)
    """Avisos de configuração: se as repetições NÃO foram rodadas com a mesma configuração, a
    comparação deixa de medir estabilidade."""

    @property
    def n_pairs(self) -> int:
        return self.diff.n_pairs

    @property
    def rate_identical(self) -> float:
        return self.diff.rate_identical

    @property
    def field_change_rate(self) -> dict[str, float]:
        return self.diff.field_change_rate

    def most_unstable_fields(self) -> list[tuple[str, float]]:
        return self.diff.most_unstable_fields()

    def to_frame(self) -> pd.DataFrame:
        return self.diff.to_frame()

    def field_frame(self) -> pd.DataFrame:
        return self.diff.field_frame()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.diff.to_dict(),
            "changed_fields_by_arquivo": dict(self.changed_fields_by_arquivo),
            "warnings": list(self.warnings),
        }


def config_warnings(
    run_a: Sequence[ExtractionRecord], run_b: Sequence[ExtractionRecord], attrs: Sequence[str]
) -> list[str]:
    """Lista atributos de configuração que diferem entre as duas execuções."""
    out: list[str] = []
    for attr in attrs:
        va = {getattr(r, attr) for r in run_a}
        vb = {getattr(r, attr) for r in run_b}
        if va != vb:
            out.append(
                f"{attr} difere entre as execuções: {sorted(map(str, va))} vs "
                f"{sorted(map(str, vb))}"
            )
    return out


def stability_report(
    rep1: Sequence[ExtractionRecord], rep2: Sequence[ExtractionRecord]
) -> StabilityReport:
    """Compara a repetição 1 com a repetição 2 da MESMA configuração."""
    d = diff_runs(rep1, rep2)
    return StabilityReport(
        diff=d,
        changed_fields_by_arquivo=d.changed_fields_by_arquivo,
        warnings=config_warnings(rep1, rep2, _CONFIG_ATTRS),
    )
