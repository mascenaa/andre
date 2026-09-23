from __future__ import annotations

import pytest

from ficha.audit import build_audit_summary, compare_prompt_variants
from ficha.audit.confidence import RULE_V1, RULE_V2, ConfidenceRule, build_final_fichas
from ficha.audit.fidelity import check_fidelity
from ficha.audit.leakage import TEXT_FIELDS, check_fewshot_leakage, leakage_summary
from ficha.prompts import FEW_SHOT_EXAMPLES
from ficha.schema import Confianca
from ficha.types import ParseStatus
from test_audit_helpers import TRECHO_OK, ficha_dict, make_record

# Caso real (06_Barnes_2016, execução híbrida): limitação do exemplo de fraude em cartão.
BARNES_LIMITACAO = (
    "Os rótulos vêm de chargebacks, então fraudes nunca contestadas pelos clientes ficam "
    "fora do ground truth"
)
EX1_TRECHO = FEW_SHOT_EXAMPLES[0].output.evidencia.trecho
TRECHO_FIXTURE = TRECHO_OK


def test_limitacao_copiada_do_exemplo_vaza() -> None:
    rec = make_record("06_Barnes_2016.pdf", ficha_dict(limitacao=BARNES_LIMITACAO))
    r = check_fewshot_leakage(rec)
    assert r.any_leak and r.leaked_fields == ["limitacao"]
    assert {t.example_value.lower() for t in r.leaked_terms} == {"fraude", "chargebacks"}
    lf = r.leaked[0]
    assert lf.example_index == 0 and lf.example_name == "com_limitacao"
    assert lf.similarity >= 0.9 and lf.method in ("ratio", "partial_ratio")
    assert lf.value == BARNES_LIMITACAO and "chargebacks" in lf.example_value


def test_barnes_passa_na_fidelidade_mas_cai_para_baixa() -> None:
    rec = make_record("06_Barnes_2016.pdf", ficha_dict(limitacao=BARNES_LIMITACAO))
    assert check_fidelity(rec).found and check_fidelity(rec).page_ok
    for rule in (RULE_V1, RULE_V2):
        f = build_final_fichas([rec], [rec], [rec], rule)[0]
        assert f.confianca == Confianca.BAIXA
        assert len(f.motivos) == 2  # similaridade + um motivo com os termos do campo
        assert f.motivos[1] == (
            "campo limitacao contém 'fraude', 'chargebacks', termos do exemplo few-shot "
            "ausentes do texto enviado"
        )
        assert f.motivos[0].startswith(
            "campo limitacao copiado do exemplo few-shot (similaridade 0.9"
        )
        assert f.to_row()["audit_vazamento_fewshot"] == "limitacao"


def test_trecho_copiado_vaza() -> None:
    # 08_Leka: trecho = frase do exemplo few-shot.
    rec = make_record("08_Leka.pdf", ficha_dict(evidencia={"trecho": EX1_TRECHO, "pagina": 4}))
    r = check_fewshot_leakage(rec)
    assert "evidencia.trecho" in r.leaked_fields
    assert r.leaked[r.leaked_fields.index("evidencia.trecho")].similarity == 1.0


def test_trecho_copiado_de_outra_frase_da_entrada_do_exemplo() -> None:
    frase = "A two-layer LSTM is trained on sequences of the last 30 transactions per card."
    rec = make_record("x.pdf", ficha_dict(evidencia={"trecho": frase}))
    r = check_fewshot_leakage(rec)
    assert r.leaked_fields == ["evidencia.trecho"]
    assert r.leaked[0].method == "partial_ratio_contexto" and r.leaked[0].similarity == 1.0


def test_ficha_legitima_nao_vaza() -> None:
    r = check_fewshot_leakage(make_record())
    assert not r.any_leak
    assert set(r.max_similarity) == set(TEXT_FIELDS)
    assert max(r.max_similarity.values()) < 0.6


def test_none_nao_vaza_e_sem_ficha() -> None:
    # Exemplo 2 tem limitacao None; ficha com None não pode "casar" com ele.
    r = check_fewshot_leakage(make_record(ficha=ficha_dict(limitacao=None)))
    assert "limitacao" not in r.max_similarity and not r.any_leak
    assert not check_fewshot_leakage(make_record(failed=True)).any_leak


def test_sigla_curta_nao_usa_partial_ratio() -> None:
    # "AUPRC." está contido na métrica do exemplo 1, mas é coincidência de sigla.
    r = check_fewshot_leakage(make_record(ficha=ficha_dict(metrica="AUPRC.")))
    assert not r.any_leak


def test_limiar_configuravel_e_regra_desligavel() -> None:
    rec = make_record(ficha=ficha_dict(limitacao=BARNES_LIMITACAO))
    assert not check_fewshot_leakage(rec, threshold=0.99, marker_terms=()).any_leak
    only_sim = ConfidenceRule(check_leakage_terms=False)
    f = build_final_fichas([rec], [rec], [rec], only_sim)[0]
    assert f.confianca == Confianca.BAIXA and len(f.motivos) == 1
    assert "fraude" not in only_sim.describe()
    off = ConfidenceRule(check_leakage=False)
    assert build_final_fichas([rec], [rec], [rec], off)[0].confianca == Confianca.ALTA
    assert "exemplo few-shot" in RULE_V2.describe()
    assert "exemplo few-shot" not in off.describe()


def test_examples_customizados() -> None:
    rec = make_record(ficha=ficha_dict(limitacao=BARNES_LIMITACAO))
    # Exemplos customizados: sem marcadores padrão, e o exemplo de fraude não está na lista.
    assert not check_fewshot_leakage(rec, examples=FEW_SHOT_EXAMPLES[1:]).any_leak


def test_assign_com_leakage() -> None:
    rec = make_record(ficha=ficha_dict(limitacao=BARNES_LIMITACAO))
    leak = check_fewshot_leakage(rec)
    lvl, why = RULE_V2.assign(check_fidelity(rec), 0, 0, ParseStatus.OK, True, leak)
    assert lvl == Confianca.BAIXA and "campo limitacao copiado" in why[0]


def test_leakage_summary() -> None:
    recs = [
        make_record("a.pdf"),
        make_record("b.pdf", ficha_dict(limitacao=BARNES_LIMITACAO)),
        make_record(
            "c.pdf",
            ficha_dict(limitacao=BARNES_LIMITACAO, evidencia={"trecho": EX1_TRECHO}),
        ),
        make_record("d.pdf", failed=True),
    ]
    s = leakage_summary(recs)
    assert (s.n, s.n_with_leak, s.rate) == (4, 2, 0.5)
    assert s.by_field["limitacao"] == 2 and s.by_field["evidencia.trecho"] == 1
    assert s.by_field["problema"] == 0 and set(s.by_field) == set(TEXT_FIELDS)
    df = s.to_frame()
    # b: 1 similaridade + 2 termos; c: 2 similaridades + 3 termos (trecho tem "chargebacks")
    assert len(df) == 8
    assert set(df["method"]) >= {"ratio", "termo_do_exemplo"}
    assert list(df.columns) == [
        "arquivo",
        "run_id",
        "field",
        "example_index",
        "example_name",
        "similarity",
        "method",
        "value",
        "example_value",
    ]
    assert [r.arquivo for r in s.leaks()] == ["b.pdf", "c.pdf"]
    assert s.to_dict()["n_with_leak"] == 2


def test_prompt_compare_expoe_taxa_de_vazamento() -> None:
    names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    few = [
        make_record(
            n, ficha_dict(limitacao=BARNES_LIMITACAO) if n == "a.pdf" else None, variant="v_full"
        )
        for n in names
    ]
    sem = [make_record(n, variant="v_sem_fewshot") for n in names]
    cmp = compare_prompt_variants(few, sem)
    assert cmp.a.rate_fewshot_leak == pytest.approx(0.25)
    assert cmp.b.rate_fewshot_leak == 0.0
    df = cmp.to_frame()
    assert df.loc["rate_fewshot_leak", "delta (b-a)"] == pytest.approx(-0.25)
    # O critério declarado de vitória não inclui vazamento.
    assert "rate_fewshot_leak" not in cmp.decide().values


def test_summary_inclui_bloco_de_vazamento() -> None:
    recs = [make_record("a.pdf"), make_record("b.pdf", ficha_dict(limitacao=BARNES_LIMITACAO))]
    s = build_audit_summary(recs, stability_rep=recs, alt_input=recs)
    assert s.leakage is not None and s.leakage.n_with_leak == 1
    d = s.to_dict()
    assert d["vazamento_exemplos"]["leaks"] == [
        {"arquivo": "b.pdf", "run_id": "run_a", "fields": ["limitacao"]}
    ]
    t = s.tables()
    assert set(t["vazamento"]["arquivo"]) == {"b.pdf"}
    assert d["distribuicao_confianca"] == {"alta": 1, "media": 0, "baixa": 1}


# --------------------------------------------------------------------------- termos de domínio


def test_contaminacao_parcial_por_termo() -> None:
    # 09_AsensioRamos (física solar): similaridade 0.77 < 0.80, mas "supermercados" sem
    # "supermarket" no texto enviado é inequívoco.
    dados = (
        "Dados de magnetogramas HMI SHARP de 123 observações de supermercados brasileiros, "
        "período e volume não informados"
    )
    rec = make_record("09_AsensioRamos.pdf", ficha_dict(dados=dados))
    r = check_fewshot_leakage(rec)
    assert r.leaked == [] and r.leaked_fields == ["dados"]
    t = r.leaked_terms[0]
    assert t.example_name == "sem_limitacao" and t.example_index == 1
    assert t.method == "termo_do_exemplo" and t.example_value == "supermercados"
    f = build_final_fichas([rec], [rec], [rec], RULE_V2)[0]
    assert f.confianca == Confianca.BAIXA
    assert f.motivos == [
        "campo dados contém 'supermercados', termo do exemplo few-shot ausente do texto enviado"
    ]


def test_termo_legitimo_quando_o_artigo_fala_dele() -> None:
    ctx = "[p. 1]\nWe forecast weekly sales for 50 supermarkets. " + TRECHO_FIXTURE
    rec = make_record(
        ficha=ficha_dict(dados="Vendas semanais de 50 supermercados."), context_text=ctx
    )
    assert not check_fewshot_leakage(rec).any_leak


def test_cada_marcador_existe_no_seu_exemplo() -> None:
    import re

    from ficha.audit.leakage import EXAMPLE_MARKER_TERMS

    by_name = {ex.name: ex for ex in FEW_SHOT_EXAMPLES}
    for mk in EXAMPLE_MARKER_TERMS:
        ex = by_name[mk.example_name]
        out = " ".join(str(v) for v in ex.output.model_dump().values())
        assert re.search(mk.pattern_pt, out, re.I) or re.search(
            mk.pattern_en, ex.context_text, re.I
        ), mk
