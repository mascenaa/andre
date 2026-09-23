from __future__ import annotations

from conftest import make_document
from ficha.audit.confidence import FALHOU, ConfidenceRule, build_final_fichas
from ficha.audit.fidelity import FidelityResult, check_fidelity
from ficha.llm.fake import FakeBehavior
from ficha.schema import Confianca
from ficha.types import ParseStatus
from test_audit_helpers import CONTEXT, TRECHO_OK, ficha_dict, make_record, record_from_fake

RULE = ConfidenceRule()


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
    fichas = build_final_fichas([make_record("a.pdf")], None, None)
    assert fichas[0].confianca == Confianca.MEDIA
    assert fichas[0].n_changed_input is None


def test_fidelidade_usa_threshold_da_regra() -> None:
    trecho = TRECHO_OK.replace("poorly", "badly")
    rec = make_record(ficha=ficha_dict(evidencia={"trecho": trecho}), context_text=CONTEXT)
    score = check_fidelity(rec).score
    assert 0.9 <= score < 1.0
    f_strict = build_final_fichas([rec], [rec], [rec], ConfidenceRule(fidelity_threshold=0.99))
    assert f_strict[0].confianca == Confianca.BAIXA
    f_def = build_final_fichas([rec], [rec], [rec])
    assert f_def[0].confianca == Confianca.ALTA
