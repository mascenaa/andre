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


def _fid(arquivo: str, found: bool, method: str, claimed: int = 1, page: int | None = 1):
    from ficha.audit.fidelity import FidelityResult

    return FidelityResult(
        arquivo, "r", found, 1.0 if found else 0.47, method, claimed, page, page == claimed, 0.9
    )  # type: ignore[arg-type]


def _fake_summary(results, trechos):  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    from ficha.types import ParseStatus

    fid = SimpleNamespace(
        n=len(results),
        n_found=sum(r.found for r in results),
        rate_found=sum(r.found for r in results) / len(results),
        n_page_ok=sum(r.page_ok for r in results),
        by_method={"exact": 0, "fuzzy": 0, "none": 0},
        results=results,
        failures=lambda: [r for r in results if not r.found],
    )
    fichas = [
        SimpleNamespace(
            arquivo=a,
            failed=False,
            parse_status=ParseStatus.OK,
            ficha=SimpleNamespace(evidencia=SimpleNamespace(trecho=t)),
        )
        for a, t in trechos.items()
    ]
    return SimpleNamespace(fidelity=fid, fichas=fichas)


LEAK = (
    "Forecast accuracy is measured with the weighted mean absolute percentage error (WMAPE) "
    "over a 13-week horizon."
)


def test_fidelity_block_separates_invented_wrong_page_and_failed() -> None:
    from ficha.report.assemble import fidelity_block

    results = [
        _fid("08_Leka_2019_Comparison_III.pdf", False, "fuzzy", 5, None),
        _fid("11_Jarolim_2023_PINN.pdf", True, "exact", 15, 16),
        _fid("c_sem.pdf", False, "none", 1, None),
        _fid("ok_2020_x.pdf", True, "exact"),
    ]
    summary = _fake_summary(results, {"08_Leka_2019_Comparison_III.pdf": LEAK})
    text = fidelity_block(summary).conclusao  # type: ignore[arg-type]
    assert "08_Leka_2019 (cópia do exemplo few-shot do prompt)" in text
    assert "11_Jarolim_2023: declarada 15, encontrada 16" in text
    assert "sem ficha válida" in text and "c_sem" in text


def test_few_shot_leak_and_highlight() -> None:
    from ficha.report.assemble import few_shot_leak, invention_highlight, short_name

    assert few_shot_leak(LEAK) is not None
    assert few_shot_leak("We use the MIMIC-IV database, covering 2008 to 2019.") is None
    assert short_name("D04_Jiao_2020_Flare_Intensity.pdf") == "D04_Jiao_2020"
    assert short_name("artigo_01.pdf") == "artigo_01"
    summary = _fake_summary(
        [_fid("08_Leka_2019_X_Y.pdf", False, "fuzzy", 5, None)], {"08_Leka_2019_X_Y.pdf": LEAK}
    )
    text = invention_highlight(summary)  # type: ignore[arg-type]
    assert "EXEMPLO few-shot" in text and "13-week" in text and "BAIXA" in text
    assert invention_highlight(_fake_summary([_fid("a.pdf", True, "exact")], {})) == ""  # type: ignore[arg-type]


def test_calibration_and_distributions(rehearsal) -> None:  # type: ignore[no-untyped-def]
    from ficha.report.assemble import calibration_sentence, mean_text_similarity

    assert mean_text_similarity({"problema": 0.5, "dados": 0.3, "limitacao": 1.0}) == 0.4
    text = calibration_sentence(
        rehearsal.summary, {"v1 estrita": {"alta": 0, "media": 2, "baixa": 17}}
    )
    assert "v1 estrita: alta 0/média 2/baixa 17" in text
    assert "entre estratégias" in text and "entre repetições" in text


def test_report_has_highlight_and_coverage(rehearsal, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import pymupdf

    nar = Narrative(cobertura_limitacao="Só 9 de 62 frases chegaram.", nota_limitacao="Nota X.")
    content = build_report_content(
        integrantes=["Ana Souza"],
        modelo="fake",
        summary=rehearsal.summary,
        cost=rehearsal.cost,
        narrative=nar,
        rule_distributions={"v1": {"alta": 1, "media": 1, "baixa": 2}},
    )
    assert content.estrategia_evidencia.startswith("Só 9")
    assert content.resultados_auditoria.entrada.conclusao.endswith("Nota X.")
    out = build_report_pdf(content, tmp_path / "r.pdf")
    assert count_pages(out) <= 3
    with pymupdf.open(out) as pdf:
        text = " ".join(p.get_text() for p in pdf)
    assert "Evidência medida" in " ".join(text.split())
