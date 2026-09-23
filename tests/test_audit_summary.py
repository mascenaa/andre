from __future__ import annotations

from ficha.audit import (
    AuditSummary,
    build_final_fichas,
    compare_prompt_variants,
    fichas_nao_defensaveis,
    fidelity_summary,
    input_effect_report,
    stability_report,
    temperature_report,
)
from ficha.audit.confidence import ConfidenceRule
from test_audit_helpers import ficha_dict, make_record

NAMES = ["a.pdf", "b.pdf", "c.pdf"]


def build() -> AuditSummary:
    primary = [
        make_record("a.pdf"),
        make_record("b.pdf", ficha_dict(evidencia={"trecho": "Trecho totalmente inventado aqui."})),
        make_record("c.pdf", failed=True),
    ]
    rep2 = [make_record(n, run_id="rep2") for n in NAMES]
    alt = [make_record(n, strategy="semantic", run_id="alt") for n in NAMES]
    t_alt = [make_record("a.pdf", temperature=0.7, run_id="t07")]
    prompt_b = [make_record(n, variant="v_sem_fewshot", run_id="pb") for n in NAMES]
    rule = ConfidenceRule()
    return AuditSummary(
        fidelity=fidelity_summary(primary),
        fichas=build_final_fichas(primary, rep2, alt, rule),
        rule=rule,
        stability=stability_report(primary, rep2),
        input_effect=input_effect_report(primary, alt),
        temperature=temperature_report(primary, t_alt),
        prompt_comparison=compare_prompt_variants(primary, prompt_b),
    )


def test_tables_colunas() -> None:
    t = build().tables()
    assert set(t) == {
        "fidelidade",
        "estabilidade_campos",
        "entrada_campos",
        "entrada_estrategias",
        "entrada_divergencias",
        "temperatura",
        "temperatura_campos",
        "prompts",
        "prompts_campos",
        "confianca",
        "fichas",
        "nao_defensaveis",
    }
    assert {"arquivo", "found", "score", "method", "page_ok"} <= set(t["fidelidade"].columns)
    assert list(t["estabilidade_campos"].columns) == ["field", "change_rate", "mean_similarity"]
    assert list(t["confianca"].columns) == ["confianca", "n"]
    assert list(t["nao_defensaveis"].columns) == ["arquivo", "confianca", "motivos"]
    assert len(t["fichas"]) == 3
    assert {"arquivo", "confianca", "evidencia_trecho", "audit_motivos"} <= set(t["fichas"].columns)
    assert list(t["nao_defensaveis"]["arquivo"]) == ["b.pdf", "c.pdf"]


def test_to_dict_e_distribuicao() -> None:
    s = build()
    d = s.to_dict()
    assert d["distribuicao_confianca"] == {"alta": 1, "media": 0, "baixa": 2}
    assert d["fidelidade"]["n_found"] == 1
    assert "Regra de confiança" in d["regra_confianca"]
    assert [x["arquivo"] for x in d["nao_defensaveis"]] == ["b.pdf", "c.pdf"]
    assert len(fichas_nao_defensaveis(s.fichas)) == 2


def test_tabelas_opcionais_ausentes() -> None:
    primary = [make_record("a.pdf")]
    s = AuditSummary(
        fidelity=fidelity_summary(primary), fichas=build_final_fichas(primary, None, None)
    )
    assert set(s.tables()) == {"fidelidade", "confianca", "fichas", "nao_defensaveis"}
    assert s.to_dict()["estabilidade"] is None


def test_build_audit_summary_equivale_ao_manual() -> None:
    from ficha.audit import build_audit_summary

    primary = [make_record(n) for n in NAMES]
    rep2 = [make_record(n, run_id="rep2") for n in NAMES]
    s = build_audit_summary(
        primary,
        stability_rep=rep2,
        alt_input=rep2,
        t_alt=[make_record("a.pdf", temperature=0.7)],
        prompt_runs=(primary, rep2),
    )
    assert s.confidence_distribution == {"alta": 3, "media": 0, "baixa": 0}
    assert s.stability is not None and s.stability.rate_identical == 1.0
    assert s.temperature is not None and s.temperature.escolha_t0_se_sustenta
    assert s.prompt_comparison is not None
