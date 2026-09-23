from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ficha.cli import run_rehearsal
from ficha.report.assemble import (
    Narrative,
    build_report_content,
    features_label,
    pct,
    prompt_table,
    temp,
)
from ficha.report.overview import corpus_frame, selection_frame, strategy_summary
from ficha.report.pdf import build_report_pdf, count_pages
from ficha.select import FirstPagesSelector, KeywordSectionSelector
from ficha.types import Document


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    return run_rehearsal(tmp_path_factory.mktemp("reh"), n_docs=4)


def test_helpers() -> None:
    assert pct(0.955) == "96%"
    assert pct(None) == "—"
    assert pct(0.5, 1) == "50,0%"
    assert temp(0.7) == "0,7"
    assert "few-shot" in features_label("v_full")
    assert "few-shot" not in features_label("v_sem_fewshot")


def test_prompt_table_uses_percentages(rehearsal) -> None:  # type: ignore[no-untyped-def]
    table = prompt_table(rehearsal.summary.prompt_comparison)
    assert list(table.columns) == ["métrica", "v_full", "v_sem_fewshot"]
    row = table.set_index("métrica").loc["JSON válido de primeira"]
    assert row["v_full"].endswith("%")


def test_build_content_overrides_and_pdf(rehearsal, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    nar = Narrative(conclusoes={"fidelidade": "Conclusão revisada pelo grupo."})
    content = build_report_content(
        integrantes=["Ana Souza"],
        modelo="fake",
        summary=rehearsal.summary,
        cost=rehearsal.cost,
        narrative=nar,
    )
    assert content.resultados_auditoria.fidelidade.conclusao == "Conclusão revisada pelo grupo."
    assert "Distribuição resultante" in content.regra_confianca
    assert content.custo.razao_ingenuo_real == pytest.approx(rehearsal.cost.ratio)
    out = build_report_pdf(content, tmp_path / "r.pdf")
    assert count_pages(out) <= 3


def test_requires_all_four_checks(rehearsal) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    incomplete = replace(rehearsal.summary, temperature=None)
    with pytest.raises(ValueError, match=r"4\.4d"):
        build_report_content(
            integrantes=["A B"], modelo="m", summary=incomplete, cost=rehearsal.cost
        )


def test_overview_frames(docs: list[Document]) -> None:
    corpus = corpus_frame(docs, len)
    assert corpus.iloc[-1]["arquivo"] == "TOTAL"
    assert corpus.iloc[-1]["caracteres"] == sum(d.n_chars for d in docs)
    sel = selection_frame(docs, [FirstPagesSelector(n_pages=1), KeywordSectionSelector()], len)
    assert set(sel["estrategia"]) == {"first_pages", "keyword"}
    summary = strategy_summary(sel)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.index) == ["first_pages", "keyword"]


def test_fidelity_block_separates_invented_from_failed() -> None:
    from types import SimpleNamespace

    from ficha.audit.fidelity import FidelityResult
    from ficha.report.assemble import fidelity_block

    def res(arquivo: str, method: str) -> FidelityResult:
        found = method != "none" and arquivo != "b.pdf"
        return FidelityResult(arquivo, "r", found, 1.0 if found else 0.4, method, 1, 1, found, 0.9)  # type: ignore[arg-type]

    results = [res("a.pdf", "exact"), res("b.pdf", "fuzzy"), res("c.pdf", "none")]
    fid = SimpleNamespace(
        n=3,
        n_found=1,
        rate_found=1 / 3,
        n_page_ok=1,
        by_method={"exact": 1, "fuzzy": 1, "none": 1},
        failures=lambda: [r for r in results if not r.found],
    )
    block = fidelity_block(SimpleNamespace(fidelity=fid))  # type: ignore[arg-type]
    assert "inventados: b.pdf" in block.conclusao
    assert "sem ficha válida" in block.conclusao and "c.pdf" in block.conclusao
    assert "c.pdf" not in block.conclusao.split("inventados:")[1].split(".")[0]
