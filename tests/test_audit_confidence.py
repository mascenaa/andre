from __future__ import annotations

from conftest import make_document
from ficha.audit.confidence import FALHOU, RULE_V1, RULE_V2, ConfidenceRule, build_final_fichas
from ficha.audit.fidelity import FidelityResult, check_fidelity
from ficha.llm.fake import FakeBehavior
from ficha.schema import Confianca
from ficha.types import ParseStatus
from test_audit_helpers import CONTEXT, TRECHO_OK, ficha_dict, make_record, record_from_fake

RULE = ConfidenceRule(input_effect_mode="strict")  # v1: testes originais inalterados


def fid(found: bool = True, score: float = 1.0, page_ok: bool = True) -> FidelityResult:
    return FidelityResult(
        arquivo="a.pdf",
        run_id="r",
        found=found,
        score=score,
        method="exact" if score == 1.0 else "fuzzy",
        page_claimed=1,
        page_found=(1 if page_ok else 2) if found else None,
        page_ok=page_ok and found,
        threshold=0.9,
    )


def test_describe() -> None:
    txt = RULE.describe()
    assert "BAIXA" in txt and "MEDIA" in txt and "ALTA" in txt and "0.90" in txt
    assert "página" in ConfidenceRule(require_page_ok=True).describe()
    assert "página diferente" not in ConfidenceRule(require_page_ok=False).describe()


def test_alta() -> None:
    lvl, why = RULE.assign(fid(), 0, 0, ParseStatus.OK)
    assert lvl == Confianca.ALTA and len(why) == 1 and "trecho encontrado" in why[0]
    # REPAIRED não rebaixa por si só (a ficha passou pelo esquema).
    assert RULE.assign(fid(), 0, 0, ParseStatus.REPAIRED)[0] == Confianca.ALTA


def test_baixa_parse_failed() -> None:
    lvl, why = RULE.assign(fid(False, 0.0, False), None, None, ParseStatus.FAILED)
    assert lvl == Confianca.BAIXA and why == [
        "extração falhou: o modelo não devolveu ficha válida (parse FAILED)"
    ]


def test_baixa_trecho_nao_encontrado() -> None:
    lvl, why = RULE.assign(fid(False, 0.41, False), 0, 0, ParseStatus.OK)
    assert lvl == Confianca.BAIXA
    assert why == [
        "trecho não encontrado no texto enviado (score 0.41 < 0.90): possível trecho inventado"
    ]


def test_baixa_pagina_errada_e_opcao_de_ignorar() -> None:
    lvl, why = RULE.assign(fid(page_ok=False), 0, 0, ParseStatus.OK)
    assert lvl == Confianca.BAIXA and "página errada: declarada p. 1" in why[0]
    lvl2, _ = ConfidenceRule(require_page_ok=False).assign(fid(page_ok=False), 0, 0, ParseStatus.OK)
    assert lvl2 == Confianca.ALTA


def test_media_por_mudanca_de_campos() -> None:
    lvl, why = RULE.assign(fid(), 1, 0, ParseStatus.OK)
    assert lvl == Confianca.MEDIA and why == ["1 campo(s) mudou(aram) entre repetições"]
    lvl, why = RULE.assign(fid(), 0, 2, ParseStatus.OK)
    assert lvl == Confianca.MEDIA and why == ["2 campo(s) mudou(aram) entre estratégias de entrada"]


def test_baixa_por_instabilidade() -> None:
    lvl, why = RULE.assign(fid(), 3, 0, ParseStatus.OK)
    assert lvl == Confianca.BAIXA and why == ["3 campos mudaram entre repetições (instável)"]


def test_nao_verificado_limita_a_media() -> None:
    lvl, why = RULE.assign(fid(), None, 0, ParseStatus.OK)
    assert lvl == Confianca.MEDIA and why == ["comparação entre repetições não verificada"]
    lvl, why = RULE.assign(fid(), None, None, ParseStatus.OK)
    assert lvl == Confianca.MEDIA and len(why) == 2


def test_limitacao_sem_suporte_limita_a_media() -> None:
    lvl, why = RULE.assign(fid(), 0, 0, ParseStatus.OK, limitacao_supported=False)
    assert lvl == Confianca.MEDIA and "possível limitação inventada" in why[0]
    off = ConfidenceRule(check_limitacao=False)
    assert off.assign(fid(), 0, 0, ParseStatus.OK, limitacao_supported=False)[0] == Confianca.ALTA


def test_motivos_se_acumulam_e_o_menor_nivel_vence() -> None:
    lvl, why = RULE.assign(fid(False, 0.5, False), 1, None, ParseStatus.OK, False)
    assert lvl == Confianca.BAIXA and len(why) == 4


def test_limiar_da_regra_prevalece() -> None:
    f = fid(True, 0.92)
    assert ConfidenceRule(fidelity_threshold=0.95).assign(f, 0, 0, ParseStatus.OK)[0] == (
        Confianca.BAIXA
    )


def test_build_final_fichas() -> None:
    names = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
    primary = [
        make_record("a.pdf"),
        make_record("b.pdf"),
        make_record("c.pdf", failed=True),
        record_from_fake(make_document("d.pdf"), FakeBehavior(invent_trecho_rate=1.0)),
        make_record("e.pdf"),
    ]
    rep2 = [
        make_record("a.pdf", run_id="rep2"),
        make_record(
            "b.pdf", ficha_dict(metodo="Outro método completamente distinto."), run_id="rep2"
        ),
        make_record("c.pdf", run_id="rep2"),
        make_record("d.pdf", run_id="rep2"),
        # e.pdf ausente na repetição → não verificado
    ]
    alt = [make_record(n, strategy="semantic", run_id="alt") for n in names]
    fichas = build_final_fichas(primary, rep2, alt, RULE)
    assert [f.arquivo for f in fichas] == names
    by = {f.arquivo: f for f in fichas}
    assert by["a.pdf"].confianca == Confianca.ALTA
    assert by["b.pdf"].confianca == Confianca.MEDIA and by["b.pdf"].diffs_stability == ["metodo"]
    assert by["c.pdf"].confianca == Confianca.BAIXA and by["c.pdf"].ficha.problema == FALHOU
    assert by["c.pdf"].to_row()["evidencia_pagina"] is None
    assert by["d.pdf"].confianca == Confianca.BAIXA
    assert any("trecho não encontrado" in m for m in by["d.pdf"].motivos)
    assert by["e.pdf"].confianca == Confianca.MEDIA and by["e.pdf"].n_changed_stability is None
    row = by["b.pdf"].to_row()
    assert row["confianca"] == "media" and row["audit_campos_instaveis"] == "metodo"
    assert all(f.ficha.confianca is not None for f in fichas)


def test_build_sem_comparacoes_no_maximo_media() -> None:
    fichas = build_final_fichas([make_record("a.pdf")], None, None, RULE)
    assert fichas[0].confianca == Confianca.MEDIA
    assert fichas[0].n_changed_input is None


def test_fidelidade_usa_threshold_da_regra() -> None:
    trecho = TRECHO_OK.replace("poorly", "badly")
    rec = make_record(ficha=ficha_dict(evidencia={"trecho": trecho}), context_text=CONTEXT)
    score = check_fidelity(rec).score
    assert 0.9 <= score < 1.0
    f_strict = build_final_fichas([rec], [rec], [rec], ConfidenceRule(fidelity_threshold=0.99))
    assert f_strict[0].confianca == Confianca.BAIXA
    f_def = build_final_fichas([rec], [rec], [rec], RULE)
    assert f_def[0].confianca == Confianca.ALTA


# --------------------------------------------------------------------------- regra v2


def _paraphrase_alt(arquivo: str, **kw: object) -> object:
    """Mesma ficha reescrita pela outra estratégia: texto livre e evidência diferentes."""
    return make_record(
        arquivo,
        ficha_dict(
            problema="Estimar o risco de retorno ao hospital após a alta.",
            dados="Registros eletrônicos de um hospital terciário em Boston.",
            metodo="Árvores de gradiente sobre variáveis e texto clínico.",
            metrica="Área sob a curva ROC.",
            evidencia={"trecho": TRECHO_OK, "pagina": 1},
            **kw,
        ),
        strategy="semantic",
        run_id="alt",
    )


def test_padrao_e_v2_categorical_e_versoes() -> None:
    assert ConfidenceRule().input_effect_mode == "categorical"
    assert ConfidenceRule() == RULE_V2 and RULE_V1.input_effect_mode == "strict"
    assert (RULE_V1.version, RULE_V2.version) == ("v1", "v2")
    d2 = RULE_V2.describe()
    assert "v2" in d2 and "categorical" in d2 and "0.49" in d2 and "1.00" in d2
    assert "NÃO rebaixa" in d2
    assert "strict" in RULE_V1.describe()


def test_categorical_parafrase_entre_estrategias_nao_conta() -> None:
    from ficha.audit.diff import diff_runs

    primary = [make_record("a.pdf")]
    alt = [_paraphrase_alt("a.pdf")]
    d = diff_runs(primary, alt)
    assert d.n_changed("a.pdf") == 4
    sims = [x.similarity for x in d.per_arquivo["a.pdf"] if not x.equal]
    assert max(sims) < 0.6  # paráfrase: similaridade baixa
    assert d.n_categorical_changes("a.pdf") == 0

    v2 = build_final_fichas(primary, primary, alt, RULE_V2)[0]
    assert v2.confianca == Confianca.ALTA and v2.n_changed_input == 0 and v2.diffs_input == []
    v1 = build_final_fichas(primary, primary, alt, RULE_V1)[0]
    assert v1.confianca == Confianca.BAIXA and v1.n_changed_input == 4


def test_categorical_none_vs_str_conta_um() -> None:
    primary = [make_record("a.pdf")]  # limitacao preenchida
    alt = [_paraphrase_alt("a.pdf", limitacao=None)]
    f = build_final_fichas(primary, primary, alt, RULE_V2)[0]
    assert f.n_changed_input == 1 and f.diffs_input == ["limitacao"]
    assert f.confianca == Confianca.MEDIA
    assert f.motivos == [
        "discordância categórica entre estratégias de entrada: limitacao é null numa e "
        "preenchida na outra"
    ]
    # str vs str diferente não é categórico
    alt2 = [_paraphrase_alt("a.pdf", limitacao="Outra limitação qualquer.")]
    assert build_final_fichas(primary, primary, alt2, RULE_V2)[0].n_changed_input == 0


def test_categorical_repeticoes_continuam_exatas() -> None:
    primary = [make_record("a.pdf")]
    rep = [make_record("a.pdf", ficha_dict(metodo="Outro método."), run_id="rep2")]
    f = build_final_fichas(primary, rep, primary, RULE_V2)[0]
    assert f.n_changed_stability == 1 and f.confianca == Confianca.MEDIA


def test_categorical_pagina_errada_e_trecho_inventado_continuam_baixa() -> None:
    errada = make_record("a.pdf", ficha_dict(evidencia={"pagina": 3}))
    inventado = record_from_fake(make_document("b.pdf"), FakeBehavior(invent_trecho_rate=1.0))
    primary = [errada, inventado]
    alt = [_paraphrase_alt("a.pdf"), _paraphrase_alt("b.pdf")]
    by = {f.arquivo: f for f in build_final_fichas(primary, primary, alt, RULE_V2)}
    assert by["a.pdf"].confianca == Confianca.BAIXA
    assert any("página errada" in m for m in by["a.pdf"].motivos)
    assert by["b.pdf"].confianca == Confianca.BAIXA
    assert any("trecho não encontrado" in m for m in by["b.pdf"].motivos)


def test_null_suspeito_nao_rebaixa() -> None:
    # CONTEXT contém "Limitations" (vocabulário) e a ficha tem limitacao null → null_suspeito.
    rec = make_record("a.pdf", ficha_dict(limitacao=None))
    f = build_final_fichas([rec], [rec], [rec], RULE_V2)[0]
    assert f.limitacao is not None and f.limitacao.verdict == "null_suspeito"
    assert f.confianca == Confianca.ALTA


def test_preenchida_sem_suporte_continua_media_em_v2() -> None:
    ctx = "[p. 1]\n" + TRECHO_OK
    rec = make_record("a.pdf", context_text=ctx)
    f = build_final_fichas([rec], [rec], [rec], RULE_V2)[0]
    assert f.limitacao is not None and f.limitacao.verdict == "preenchida_sem_suporte"
    assert f.confianca == Confianca.MEDIA


def test_revisao_manual_so_rebaixa_e_e_declarada() -> None:
    """A revisão manual (leitura contra o artigo) faz parte da regra: rebaixa a BAIXA com o
    motivo registrado, nunca promove, e aparece no texto declarado."""
    names = ["a.pdf", "b.pdf", "c.pdf"]
    primary = [make_record(n) for n in names]
    rep2 = [make_record(n, run_id="rep2") for n in names]
    alt = [make_record(n, strategy="semantic", run_id="alt") for n in names]
    regra = RULE_V2.with_manual_review({"b": "metrica expande a sigla TSS com nome inventado"})
    assert regra.manual_reason("b.pdf") == "metrica expande a sigla TSS com nome inventado"
    assert regra.manual_reason("a.pdf") is None
    assert "revisão manual" in regra.describe() and "revisão manual" not in RULE_V2.describe()

    by = {f.arquivo: f for f in build_final_fichas(primary, rep2, alt, regra)}
    assert by["a.pdf"].confianca == Confianca.ALTA
    assert by["b.pdf"].confianca == Confianca.BAIXA
    assert by["b.pdf"].motivos == ["revisão manual: metrica expande a sigla TSS com nome inventado"]
    # Numa ficha já rebaixada, o motivo lido se soma aos automáticos.
    primary[2] = make_record("c.pdf", failed=True)
    regra2 = regra.with_manual_review({"c": "o trecho é o título do artigo"})
    c = {f.arquivo: f for f in build_final_fichas(primary, rep2, alt, regra2)}["c.pdf"]
    assert c.confianca == Confianca.BAIXA and c.motivos[-1].startswith("revisão manual")
