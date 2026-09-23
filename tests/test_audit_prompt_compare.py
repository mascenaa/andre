from __future__ import annotations

import pytest

from ficha.audit.prompt_compare import compare_prompt_variants
from ficha.types import ParseStatus
from test_audit_helpers import ficha_dict, make_record

NAMES = [f"artigo_{i:02d}.pdf" for i in range(10)]
# Gabarito: artigos pares declaram limitação; ímpares não (null é o correto).
GABARITO = {n: i % 2 == 0 for i, n in enumerate(NAMES)}


def run_a() -> list:
    """10 registros: 8 OK (6 de primeira, 2 reparados), 2 FAILED."""
    out = []
    for i, n in enumerate(NAMES):
        if i >= 8:
            out.append(make_record(n, failed=True, variant="v_sem_fewshot", run_id="A"))
            continue
        status = ParseStatus.REPAIRED if i in (6, 7) else ParseStatus.OK
        # A inventa limitação em 2 dos 4 ímpares entre 0..7 (1, 3); acerta null em 5 e 7.
        lim = None if i in (5, 7) else "Limitação qualquer."
        out.append(
            make_record(
                n,
                ficha_dict(limitacao=lim),
                variant="v_sem_fewshot",
                run_id="A",
                parse_status=status,
            )
        )
    return out


def run_b() -> list:
    """10 registros OK; null exatamente nos ímpares; 1 trecho inventado."""
    out = []
    for i, n in enumerate(NAMES):
        f = ficha_dict(limitacao=None if i % 2 else "Limitação qualquer.")
        if i == 0:
            f = ficha_dict(
                limitacao="Limitação qualquer.",
                evidencia={"trecho": "Trecho que não aparece em lugar nenhum do texto."},
            )
        out.append(make_record(n, f, variant="v_full", run_id="B"))
    return out


def test_taxas_exatas_e_vencedor() -> None:
    cmp = compare_prompt_variants(run_a(), run_b(), limitacao_gabarito=GABARITO)
    a, b = cmp.a, cmp.b
    assert cmp.variants == ("v_sem_fewshot", "v_full")
    assert a.n == 10 and b.n == 10
    assert a.rate_json_valid_first_try == pytest.approx(0.6)
    assert a.rate_parse_ok == pytest.approx(0.6)
    assert a.rate_parse_repaired == pytest.approx(0.2)
    assert a.rate_parse_failed == pytest.approx(0.2)
    assert a.rate_limitacao_null == pytest.approx(2 / 8)
    # ímpares com ficha em A: 1, 3, 5, 7 → null em 5 e 7
    assert a.n_null_expected == 4 and a.rate_null_when_expected == pytest.approx(0.5)
    assert a.rate_fidelity == pytest.approx(0.8)
    assert b.rate_json_valid_first_try == 1.0 and b.rate_parse_ok == 1.0
    assert b.rate_limitacao_null == pytest.approx(0.5)
    assert b.rate_null_when_expected == 1.0
    assert b.rate_fidelity == pytest.approx(0.9)

    v = cmp.decide()
    assert v.winner == "v_full" == cmp.winner()
    assert v.decided_by == "rate_json_valid_first_try"

    # Disagreements: limitacao em 1 e 3 (A inventou), trecho em 0; falhas não pareadas.
    assert cmp.diff.n_pairs == 8 and cmp.diff.n_unpaired == 2
    dis = {(d.arquivo, d.field) for d in cmp.disagreements()}
    assert dis == {
        ("artigo_00.pdf", "evidencia.trecho"),
        ("artigo_01.pdf", "limitacao"),
        ("artigo_03.pdf", "limitacao"),
    }

    df = cmp.to_frame()
    assert list(df.columns) == ["v_sem_fewshot", "v_full", "delta (b-a)"]
    assert df.loc["rate_json_valid_first_try", "delta (b-a)"] == pytest.approx(0.4)
    assert "rate_identical_entre_variantes" in df.index


def test_desempate_por_fidelidade_e_por_null() -> None:
    a = [make_record(n, variant="va") for n in NAMES[:2]]
    b = [
        make_record(NAMES[0], variant="vb"),
        make_record(NAMES[1], ficha_dict(evidencia={"trecho": "x" * 40}), variant="vb"),
    ]
    assert compare_prompt_variants(a, b).decide().decided_by == "rate_fidelity"
    assert compare_prompt_variants(a, b).winner() == "va"

    gab = {NAMES[0]: False, NAMES[1]: False}
    a2 = [make_record(n, ficha_dict(limitacao="Inventada."), variant="va") for n in NAMES[:2]]
    b2 = [make_record(n, ficha_dict(limitacao=None), variant="vb") for n in NAMES[:2]]
    v = compare_prompt_variants(a2, b2, limitacao_gabarito=gab).decide()
    assert v.winner == "vb" and v.decided_by == "rate_null_when_expected"


def test_empate_total_fica_a_referencia() -> None:
    a = [make_record(n, variant="va") for n in NAMES[:3]]
    b = [make_record(n, variant="vb") for n in NAMES[:3]]
    v = compare_prompt_variants(a, b).decide()
    assert v.winner == "va" and v.decided_by is None and "não mudou nada" in v.explanation


def test_sem_gabarito_usa_heuristica() -> None:
    cmp = compare_prompt_variants(run_a(), run_b())
    assert cmp.a.null_ground_truth == "heuristica_vocabulario"
    # CONTEXT tem vocabulário de limitação → nenhum caso de null esperado.
    assert cmp.a.rate_null_when_expected is None and cmp.a.n_null_expected == 0
