from __future__ import annotations

from conftest import make_document
from ficha.audit.fidelity import check_fidelity, check_limitacao_support, fidelity_summary
from ficha.audit.normalize import normalize_for_match, split_context_blocks, split_context_pages
from ficha.llm.fake import FakeBehavior
from test_audit_helpers import CONTEXT, TRECHO_OK, ficha_dict, make_record, record_from_fake


def test_normalize_regras_tipograficas() -> None:
    a = "“Deep learning”  —  state – of – the – art ﬁndings ( see Fig. 1 ) ."
    assert normalize_for_match(a) == '"deep learning"-state-of-the-art findings (see fig. 1).'
    assert normalize_for_match(normalize_for_match(a)) == normalize_for_match(a)
    assert normalize_for_match("co­operation​") == "cooperation"
    assert normalize_for_match("It’s\n\n  FINE") == "it's fine"


def test_split_context_pages() -> None:
    text = "[p. 1]\nabc\n\n[p. 3]\ndef\n\n[p. 1]\nghi"
    assert split_context_blocks(text) == [(1, "abc"), (3, "def"), (1, "ghi")]
    assert split_context_pages(text) == {1: "abc\n\nghi", 3: "def"}
    assert split_context_pages("sem marcador") == {}
    assert split_context_blocks("sem marcador") == [(0, "sem marcador")]
    assert set(split_context_pages(CONTEXT)) == {1, 2, 3}


def test_trecho_exato() -> None:
    r = check_fidelity(make_record())
    assert (r.found, r.method, r.score) == (True, "exact", 1.0)
    assert r.page_found == 1 and r.page_ok


def test_trecho_com_tipografia_diferente_e_exato_apos_normalizar() -> None:
    trecho = "EXISTING approaches rely on “hand‐crafted”   risk scores that\ngeneralize poorly"
    ctx = CONTEXT.replace("hand-crafted", '"hand-crafted"')
    r = check_fidelity(
        make_record(ficha=ficha_dict(evidencia={"trecho": trecho}), context_text=ctx)
    )
    assert (r.found, r.method, r.score) == (True, "exact", 1.0)


def test_pequena_diferenca_de_limpeza_passa_por_fuzzy() -> None:
    trecho = "The model reaches an AUROC of 0.81 and an AUPRC of 0.34 on the held out test set"
    r = check_fidelity(make_record(ficha=ficha_dict(evidencia={"trecho": trecho, "pagina": 3})))
    assert r.method == "fuzzy" and r.found and r.score >= 0.9
    assert r.page_found == 3 and r.page_ok


def test_trecho_parafraseado_nao_e_encontrado() -> None:
    trecho = (
        "We examine the task of predicting whether patients return to hospital within a "
        "month after discharge."
    )
    r = check_fidelity(make_record(ficha=ficha_dict(evidencia={"trecho": trecho})))
    assert r.method == "fuzzy" and not r.found and r.score < 0.9
    assert r.page_found is None and not r.page_ok


def test_trecho_inventado_pelo_fake_nao_e_encontrado() -> None:
    rec = record_from_fake(make_document(), FakeBehavior(invent_trecho_rate=1.0))
    r = check_fidelity(rec)
    assert not r.found and r.method == "fuzzy" and r.score < 0.6


def test_pagina_errada() -> None:
    r = check_fidelity(make_record(ficha=ficha_dict(evidencia={"trecho": TRECHO_OK, "pagina": 2})))
    assert r.found and r.page_claimed == 2 and r.page_found == 1 and not r.page_ok


def test_trecho_repetido_em_duas_paginas_aceita_a_declarada() -> None:
    ctx = f"[p. 1]\n{TRECHO_OK}\n\n[p. 5]\n{TRECHO_OK}"
    rec = make_record(ficha=ficha_dict(evidencia={"pagina": 5}), context_text=ctx)
    assert check_fidelity(rec).page_ok


def test_ficha_none() -> None:
    r = check_fidelity(make_record(failed=True))
    assert (r.found, r.method, r.score, r.page_ok) == (False, "none", 0.0, False)


def test_fidelity_summary_conta_falhas_no_denominador() -> None:
    recs = [
        make_record("a.pdf"),
        make_record("b.pdf", ficha=ficha_dict(evidencia={"pagina": 2})),
        make_record("c.pdf", failed=True),
        record_from_fake(make_document("d.pdf"), FakeBehavior(invent_trecho_rate=1.0)),
    ]
    s = fidelity_summary(recs)
    assert (s.n, s.n_found, s.n_page_ok) == (4, 2, 1)
    assert s.rate_found == 0.5 and s.rate_page_ok == 0.25
    assert s.by_method == {"exact": 2, "fuzzy": 1, "none": 1}
    df = s.to_frame()
    assert len(df) == 4 and {"arquivo", "found", "score", "method", "page_ok"} <= set(df.columns)
    assert [f.arquivo for f in s.failures()] == ["c.pdf", "d.pdf"]


def test_limitacao_heuristica() -> None:
    sem_vocab = "[p. 1]\n" + TRECHO_OK
    inventada = make_record(context_text=sem_vocab)
    assert check_limitacao_support(inventada).verdict == "preenchida_sem_suporte"
    assert check_limitacao_support(inventada).supported is False
    ok = make_record()  # CONTEXT tem "5 Limitations"
    assert check_limitacao_support(ok).verdict == "preenchida_com_suporte"
    nula = make_record(ficha=ficha_dict(limitacao=None), context_text=sem_vocab)
    assert check_limitacao_support(nula).verdict == "null_ok"
    assert check_limitacao_support(make_record(failed=True)).supported is None
