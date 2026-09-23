from __future__ import annotations

from pathlib import Path

import pandas as pd

from ficha.report.figures import (
    plot_confidence_distribution,
    plot_field_change_rates,
    plot_prompt_comparison,
)
from ficha.schema import Confianca, Evidencia, Ficha

PNG = b"\x89PNG"


def test_prompt_comparison_index_or_column(tmp_path: Path) -> None:
    by_index = pd.DataFrame(
        {"v_full": [0.9, 0.8], "v_sem_fewshot": [0.7, 0.5]},
        index=["json_valido", "null_correto"],
    )
    p1 = plot_prompt_comparison(by_index, tmp_path / "a.png")
    by_col = by_index.reset_index().rename(columns={"index": "metrica"})
    p2 = plot_prompt_comparison(by_col, tmp_path / "b.png")
    for p in (p1, p2):
        assert p.read_bytes().startswith(PNG)


def test_field_change_rates(tmp_path: Path) -> None:
    frame = pd.DataFrame({"campo": ["metodo", "limitacao"], "taxa": [0.2, 0.1]})
    assert plot_field_change_rates(frame, tmp_path / "c.png").read_bytes().startswith(PNG)


def test_confidence_distribution_default_path() -> None:
    ev = Evidencia(trecho="x" * 30, pagina=1)
    fichas = [
        Ficha(
            arquivo=f"a{i}.pdf",
            problema="p",
            dados="d",
            metodo="m",
            metrica="m",
            limitacao=None,
            evidencia=ev,
            confianca=c,
        )
        for i, c in enumerate([Confianca.ALTA, Confianca.BAIXA, None])
    ]
    path = plot_confidence_distribution(fichas)
    assert path.exists() and path.read_bytes().startswith(PNG)
