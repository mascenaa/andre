from __future__ import annotations

import pytest

from ficha.prompts.builder import FEATURE_MARKERS
from ficha.prompts.variants import (
    MANDATORY_PAIR,
    VARIANTS,
    build_prompt,
    describe_diff,
    feature_diff,
)
from ficha.types import PromptFeatures


def test_variantes_esperadas():
    assert {"v_full", "v_sem_fewshot", "v_sem_delimitadores", "v_cot"} <= set(VARIANTS)


def test_v_full_tem_todas_as_tecnicas_exigidas_sem_cot():
    f = VARIANTS["v_full"]
    assert f.role and f.delimiters and f.few_shot and f.abstention
    assert not f.chain_of_thought


def test_par_obrigatorio_difere_em_uma_tecnica():
    a, b = MANDATORY_PAIR
    assert feature_diff(a, b) == {"few_shot": (True, False)}


@pytest.mark.parametrize("variant", [v for v in VARIANTS if v != "v_full"])
def test_cada_variante_difere_de_v_full_em_uma_tecnica(variant):
    assert len(feature_diff("v_full", variant)) == 1


def test_describe_diff():
    assert describe_diff("v_full", "v_sem_fewshot") == (
        "v_full × v_sem_fewshot: few_shot: on → off"
    )
    assert "mesmas técnicas" in describe_diff("v_full", "v_full")
    assert "chain_of_thought: off → on" in describe_diff("v_full", "v_cot")
    assert "A × B" in describe_diff(PromptFeatures(), PromptFeatures(role=False))


def test_build_prompt_por_nome(context):
    b = build_prompt("v_sem_fewshot")
    assert b.variant == "v_sem_fewshot"
    assert FEATURE_MARKERS["few_shot"] not in b.render(context).user


def test_variante_desconhecida():
    with pytest.raises(KeyError, match="v_full"):
        build_prompt("v_inexistente")
