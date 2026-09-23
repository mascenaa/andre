from __future__ import annotations

import pytest

from conftest import make_document
from ficha.audit.diff import FIELDS, compare_values, diff_runs
from ficha.audit.input_effect import input_effect_report
from ficha.audit.stability import stability_report
from ficha.audit.temperature import temperature_report
from ficha.llm.fake import FakeBehavior
from ficha.types import GenerationParams, ParseStatus
from test_audit_helpers import ficha_dict, make_record, record_from_fake

DOCS = [make_document(f"artigo_{i:02d}.pdf", with_limitations=i % 2 == 0) for i in range(1, 9)]


def fake_run(
    behavior: FakeBehavior | None = None,
    seed: int = 42,
    temperature: float = 0.0,
    run_id: str = "r",
) -> list:
    p = GenerationParams(temperature=temperature, seed=seed)
    return [record_from_fake(d, behavior, params=p, run_id=run_id) for d in DOCS]


def test_fields() -> None:
    assert FIELDS == (
        "problema",
        "dados",
        "metodo",
        "metrica",
        "limitacao",
        "evidencia.trecho",
        "evidencia.pagina",
    )


def test_compare_values_regras() -> None:
    assert compare_values(None, None) == (True, 1.0)
    assert compare_values(None, "x") == (False, 0.0)
    assert compare_values("Olá  Mundo", "olá mundo") == (True, 1.0)
    assert compare_values(3, 3) == (True, 1.0)
    assert compare_values(3, 4) == (False, 0.0)
    eq, sim = compare_values("abcd", "abce")
    assert not eq and sim == 0.75


def test_execucoes_identicas() -> None:
    d = diff_runs(fake_run(run_id="r1"), fake_run(run_id="r2"))
    assert d.n_pairs == 8 and d.n_identical == 8 and d.rate_identical == 1.0
    assert all(v == 0.0 for v in d.field_change_rate.values())
    assert all(v == 1.0 for v in d.field_mean_similarity.values())
    assert d.label_a == "r1" and d.label_b == "r2"


def test_instabilidade_aparece_em_problema_e_metodo() -> None:
    b = FakeBehavior(unstable_rate=1.0, temperature_noise=False)
    rep = stability_report(fake_run(b, seed=1, run_id="rep1"), fake_run(b, seed=2, run_id="rep2"))
    assert rep.rate_identical < 1.0
    top = [f for f, _ in rep.most_unstable_fields()[:3]]
    assert "problema" in top and "metodo" in top
    assert rep.field_change_rate["problema"] > 0.5
    assert rep.field_change_rate["dados"] == 0.0
    assert set(rep.changed_fields_by_arquivo) == {d.arquivo for d in DOCS}
    assert any("seed" in w for w in rep.warnings)
    assert {"field", "change_rate", "mean_similarity"} == set(rep.field_frame().columns)
    assert len(rep.to_frame()) == 8 * len(FIELDS)


def test_none_vs_none_e_none_vs_str() -> None:
    a = [
        make_record("x.pdf", ficha_dict(limitacao=None)),
        make_record("y.pdf", ficha_dict(limitacao=None)),
    ]
    b = [
        make_record("x.pdf", ficha_dict(limitacao=None)),
        make_record("y.pdf", ficha_dict(limitacao="Amostra pequena.")),
    ]
    d = diff_runs(a, b)
    assert d.changed_fields("x.pdf") == []
    assert d.changed_fields("y.pdf") == ["limitacao"]
    assert d.field_change_rate["limitacao"] == 0.5
    assert d.field_mean_similarity["limitacao"] == 0.5


def test_failed_conta_como_unpaired() -> None:
    a = [make_record("x.pdf"), make_record("y.pdf", failed=True), make_record("z.pdf")]
    b = [make_record("x.pdf"), make_record("y.pdf")]
    d = diff_runs(a, b)
    assert d.n_pairs == 1 and d.n_unpaired == 2
    assert d.unpaired == {"y.pdf": "extração falhou em A", "z.pdf": "ausente em B"}
    assert d.n_changed("y.pdf") is None and d.n_changed("x.pdf") == 0


def test_duplicata_falha_alto() -> None:
    with pytest.raises(ValueError, match="mais de um registro"):
        diff_runs([make_record("x.pdf"), make_record("x.pdf")], [make_record("x.pdf")])


def test_input_effect() -> None:
    a = [
        make_record("x.pdf", strategy="first_pages"),
        make_record("y.pdf", strategy="first_pages"),
    ]
    b = [
        make_record("x.pdf", ficha_dict(dados="Dados totalmente diferentes"), strategy="semantic"),
        make_record(
            "y.pdf",
            ficha_dict(
                evidencia={"trecho": "Um trecho que não existe no texto enviado, inventado."}
            ),
            strategy="semantic",
        ),
    ]
    rep = input_effect_report(a, b)
    assert (rep.strategy_a, rep.strategy_b) == ("first_pages", "semantic")
    dis = rep.disagreements(min_similarity=0.8)
    assert {(d.arquivo, d.field) for d in dis} == {
        ("x.pdf", "dados"),
        ("y.pdf", "evidencia.trecho"),
    }
    fc = rep.fidelity_comparison()
    assert fc["rate_found[first_pages]"] == 1.0 and fc["rate_found[semantic]"] == 0.5
    v = rep.recommend()
    assert v.strategy == "first_pages" and v.decided_by == "maior taxa de fidelidade"
    assert rep.changed_fields_by_arquivo == {"x.pdf": 1, "y.pdf": 1}
    assert rep.to_dict()["recommended"] == "first_pages"


def test_temperature_report_subconjunto() -> None:
    t0 = fake_run(run_id="t0")
    alt_b = FakeBehavior(invent_trecho_rate=1.0, broken_json_rate=1.0, temperature_noise=False)
    t_alt = fake_run(alt_b, temperature=0.7, run_id="t07")[:4]
    rep = temperature_report(t0, t_alt)
    assert rep.arquivos == [d.arquivo for d in DOCS[:4]]
    assert rep.stats_t0.n == 4 and rep.stats_alt.n == 4
    assert rep.temperature_t0 == 0.0 and rep.temperature_alt == 0.7
    assert rep.rate_json_valid_first_try_t0 == 1.0 and rep.rate_json_valid_first_try_alt == 0.0
    assert rep.rate_parse_ok_alt == 0.0 and rep.stats_alt.rate_parse_repaired == 1.0
    assert rep.rate_fidelity_t0 == 1.0 and rep.rate_fidelity_alt == 0.0
    assert rep.escolha_t0_se_sustenta
    assert rep.diff.field_change_rate["evidencia.trecho"] == 1.0
    assert "t=0.0" in rep.to_frame().columns

    inverso = temperature_report(t_alt, t0)
    assert not inverso.escolha_t0_se_sustenta


def test_status_failed_nao_quebra_temperatura() -> None:
    t0 = [make_record("x.pdf")]
    t1 = [make_record("x.pdf", failed=True, temperature=0.7)]
    rep = temperature_report(t0, t1)
    assert rep.diff.n_unpaired == 1 and rep.stats_alt.rate_parse_failed == 1.0
    assert rep.stats_alt.variant == "t=0.7"
    assert t1[0].parse_status == ParseStatus.FAILED
