from __future__ import annotations

import pytest

from conftest import make_document
from ficha.config import get_settings
from ficha.cost import (
    CostReport,
    PricePremise,
    compare_costs,
    cost_by_run,
    cost_of_records,
    cost_table,
    naive_cost,
)
from ficha.types import Usage
from test_audit_helpers import make_record

P = PricePremise(input_per_mtok=3.0, output_per_mtok=15.0, source="teste")


def test_numeros_conhecidos() -> None:
    r = CostReport.build("x", 1, 1_000_000, 0, P)
    assert r.usd_input == pytest.approx(3.0) and r.usd_total == pytest.approx(3.0)
    r2 = CostReport.build("x", 1, 0, 200_000, P)
    assert r2.usd_output == pytest.approx(3.0)


def test_premissa_visivel() -> None:
    d = cost_of_records([make_record()], P).to_dict()
    assert "US$ 3.00" in d["premissa"] and "teste" in d["premissa"]
    s = get_settings()
    pp = PricePremise.from_settings(s)
    assert pp.input_per_mtok == s.price_input_per_mtok and pp.source == s.price_source


def test_repeticoes_somam() -> None:
    recs = [make_record("a.pdf", usage=Usage(10_000, 500), run_id=f"rep{i}") for i in (1, 2)]
    recs += [make_record("b.pdf", usage=Usage(20_000, 500), run_id=f"rep{i}") for i in (1, 2)]
    r = cost_of_records(recs, P)
    assert r.n_calls == 4 and r.input_tokens == 60_000 and r.output_tokens == 2_000
    assert r.usd_total == pytest.approx(60_000 / 1e6 * 3 + 2_000 / 1e6 * 15)
    per_run = cost_by_run(recs, P)
    assert [c.label for c in per_run] == ["rep1", "rep2"]
    assert sum(c.usd_total for c in per_run) == pytest.approx(r.usd_total)


def test_ingenuo_e_razao() -> None:
    docs = [make_document("a.pdf"), make_document("b.pdf")]

    def count(t: str) -> int:
        return 100_000  # por artigo

    naive = naive_cost(docs, count, P, n_calls_per_doc=2, expected_output_tokens=300)
    assert naive.n_calls == 4 and naive.input_tokens == 400_000 and naive.output_tokens == 1_200
    real = cost_of_records(
        [make_record(n, usage=Usage(10_000, 300)) for n in ("a.pdf", "b.pdf") for _ in (1, 2)], P
    )
    cmp = compare_costs(real, naive)
    exp_naive = 0.4 * 3 + 0.0012 * 15
    exp_real = 0.04 * 3 + 0.0012 * 15
    assert cmp.ratio == pytest.approx(exp_naive / exp_real)
    assert cmp.token_ratio == pytest.approx(10.0)
    assert cmp.savings_usd == pytest.approx(exp_naive - exp_real)
    assert cmp.savings_pct == pytest.approx((exp_naive - exp_real) / exp_naive)
    assert "mais caro" in cmp.describe() and "teste" in cmp.to_dict()["premissa"]


def test_premissas_diferentes_nao_comparam() -> None:
    other = PricePremise(1.0, 1.0, "outra")
    with pytest.raises(ValueError):
        compare_costs(CostReport.build("a", 1, 1, 1, P), CostReport.build("b", 1, 1, 1, other))


def test_cost_table() -> None:
    df = cost_table([cost_of_records([make_record()], P), CostReport.build("n", 1, 5, 5, P)])
    assert list(df.columns) == [
        "label",
        "n_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usd_input",
        "usd_output",
        "usd_total",
        "premissa",
    ]
    assert len(df) == 2


def test_naive_cost_from_records_espelha_chamadas() -> None:
    from ficha.cost import naive_cost_from_records

    doc = make_document("a.pdf")

    def count(t: str) -> int:
        return len(t)  # 1 token por caractere, para contas exatas

    rec = make_record("a.pdf", usage=Usage(0, 300))
    ctx_len = len(rec.context_text)
    recs = [make_record("a.pdf", usage=Usage(ctx_len + 500, 300), run_id=f"rep{i}") for i in (1, 2)]
    naive = naive_cost_from_records(recs, [doc], count, P)
    assert naive.n_calls == 2
    assert naive.input_tokens == 2 * (500 + len(doc.full_text))
    assert naive.output_tokens == 600
    same = naive_cost_from_records(recs, {"a.pdf": doc}, count, P)
    assert same.input_tokens == naive.input_tokens
    with pytest.raises(KeyError, match=r"b\.pdf"):
        naive_cost_from_records([make_record("b.pdf")], [doc], count, P)
