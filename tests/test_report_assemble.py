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


def _fake_summary(results, trechos, leaks=None):  # type: ignore[no-untyped-def]
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
    leakage = SimpleNamespace(
        leaks=lambda: [
            SimpleNamespace(arquivo=a, leaked_fields=f) for a, f in (leaks or {}).items()
        ]
    )
    return SimpleNamespace(fidelity=fid, fichas=fichas, leakage=leakage)


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
    summary = _fake_summary(
        results,
        {"08_Leka_2019_Comparison_III.pdf": LEAK},
        leaks={"08_Leka_2019_Comparison_III.pdf": ["metodo", "evidencia.trecho"]},
    )
    text = fidelity_block(summary).conclusao  # type: ignore[arg-type]
    assert "08_Leka_2019 (cópia do exemplo few-shot do prompt)" in text
    assert "11_Jarolim_2023: declarada 15, encontrada 16" in text
    assert "sem ficha válida" in text and "c_sem" in text


def test_invention_cases_detect_leak_and_invented_trecho(rehearsal) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from ficha.extract import RunStore
    from ficha.prompts import FEW_SHOT_EXAMPLES
    from ficha.report.assemble import (
        invention_cases,
        invention_highlight,
        leakage_block,
        short_name,
    )

    assert short_name("D04_Jiao_2020_Flare_Intensity.pdf") == "D04_Jiao_2020"
    assert short_name("artigo_01.pdf") == "artigo_01"

    store = RunStore(rehearsal.table_csv.parent.parent / "runs")
    primary = store.load(rehearsal.run_ids["primary"])
    ok = next(r for r in primary if r.ficha is not None)
    copied = FEW_SHOT_EXAMPLES[0].output.limitacao
    assert copied
    leaked = replace(ok, ficha=ok.ficha.model_copy(update={"limitacao": copied}))  # type: ignore[union-attr]
    runs = {"principal": [leaked if r is ok else r for r in primary]}

    cases = invention_cases(runs)
    leak_case = next(c for c in cases if c.arquivo == ok.arquivo)
    assert leak_case.leaked_fields == ["limitacao"]
    assert leak_case.trecho_found
    text = invention_highlight(cases)
    assert "Só a comparação dos campos com os exemplos do prompt detecta" in text
    block = leakage_block(runs)
    assert block.titulo.startswith("e)")
    assert "1/" in str(block.numeros["principal"]) and "limitacao" in str(
        block.numeros["principal"]
    )
    assert invention_highlight([]) == ""


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


def test_budget_sentence_and_before_after(docs: list[Document]) -> None:
    from ficha.report.assemble import before_after_sentence, budget_sentence

    curve = pd.DataFrame(
        {
            "estrategia": ["hybrid 12k", "hybrid 20k"],
            "artigos_cobertos": [9, 15],
            "artigos_com_frases": [15, 15],
            "recall_frases": [0.302, 0.444],
            "chars_medios": [100, 160],
            "tokens_por_artigo": [2861, 4475],
        }
    )
    recall = pd.DataFrame({"estrategia": ["semantic"], "chars_medios": [50]})
    text = budget_sentence(curve, recall, docs)
    assert "12k: 30%, 9/15 artigos, ~2.861 tok" in text
    assert "+100% de caracteres sobre o semantic" in text
    assert "20k fica documentado como alternativa (15/15 artigos, +56% de tokens)" in text

    outcome = pd.DataFrame(
        [
            {
                "execucao": "semantic (antes)",
                "n": 19,
                "limitacao_null": 18,
                "preenchida_com_trecho_fiel": 1,
            },
            {
                "execucao": "hybrid (principal)",
                "n": 19,
                "limitacao_null": 18,
                "preenchida_com_trecho_fiel": 1,
            },
        ]
    )
    sentence = before_after_sentence(outcome, "semantic (antes)", "hybrid (principal)")
    assert "18/19 com semantic (antes) → 18/19 com hybrid (principal)" in sentence
    assert "não reduziu os nulls" in sentence
    assert before_after_sentence(outcome, "x", "y") == ""


def test_limitation_outcome(rehearsal) -> None:  # type: ignore[no-untyped-def]
    from ficha.cli import run_rehearsal  # noqa: F401 — garante o mesmo pacote
    from ficha.report.assemble import limitation_outcome

    recs = [f for f in rehearsal.summary.fichas]
    assert recs  # a execução existe
    from ficha.extract import RunStore

    store = RunStore(rehearsal.table_csv.parent.parent / "runs")
    primary = store.load(rehearsal.run_ids["primary"])
    frame = limitation_outcome({"p": primary})
    row = frame.iloc[0]
    assert row["n"] == len(primary)
    assert row["limitacao_preenchida"] + row["limitacao_null"] == row["n"]
    assert row["preenchida_com_trecho_fiel"] <= row["limitacao_preenchida"]
