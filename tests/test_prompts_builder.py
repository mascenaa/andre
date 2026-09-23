from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ficha.prompts.builder import (
    ALWAYS_ON_MARKERS,
    ARTICLE_CLOSE,
    ARTICLE_OPEN,
    FEATURE_MARKERS,
    MARK_ARTICLE,
    MARK_EXAMPLES,
    MARK_FORMAT,
    MARK_ROLE,
    PLAIN_ARTICLE_LABEL,
    PromptBuilder,
    prompt_schema,
)
from ficha.types import PromptFeatures

FULL = PromptFeatures(
    role=True, delimiters=True, few_shot=True, abstention=True, chain_of_thought=True
)


def _full_text(rendered):
    return f"{rendered.system or ''}\n{rendered.user}"


@pytest.mark.parametrize("feature", sorted(FEATURE_MARKERS))
def test_cada_tecnica_liga_e_desliga_seu_bloco(context, feature):
    marker = FEATURE_MARKERS[feature]
    on = PromptBuilder(FULL, "on").render(context)
    off = PromptBuilder(replace(FULL, **{feature: False}), "off").render(context)
    assert marker in _full_text(on)
    assert marker not in _full_text(off)


@pytest.mark.parametrize("marker", ALWAYS_ON_MARKERS)
def test_instrucao_e_formato_sempre_presentes(context, marker):
    minimal = PromptFeatures(
        role=False, delimiters=False, few_shot=False, abstention=False, chain_of_thought=False
    )
    assert marker in PromptBuilder(minimal, "min").render(context).user


def test_papel_vai_no_system(context):
    on = PromptBuilder(PromptFeatures(), "v").render(context)
    assert on.system is not None and on.system.startswith(MARK_ROLE)
    assert MARK_ROLE not in on.user
    off = PromptBuilder(PromptFeatures(role=False), "v").render(context)
    assert off.system is None


def test_delimitadores_envolvem_exatamente_o_context_render(context):
    user = PromptBuilder(PromptFeatures(), "v").render(context).user
    rendered = context.render()
    assert user.count(ARTICLE_OPEN) == 1
    assert user.count(ARTICLE_CLOSE) == 1
    assert f"{ARTICLE_OPEN}\n{rendered}\n{ARTICLE_CLOSE}" in user
    assert user.endswith(ARTICLE_CLOSE)  # o artigo é o último bloco


def test_sem_delimitadores_artigo_apos_linha_artigo(context):
    user = PromptBuilder(PromptFeatures(delimiters=False), "v").render(context).user
    assert ARTICLE_OPEN not in user and ARTICLE_CLOSE not in user
    assert MARK_ARTICLE not in user
    assert user.endswith(f"\n{PLAIN_ARTICLE_LABEL}\n{context.render()}")


def test_few_shot_contem_caso_null(context):
    user = PromptBuilder(PromptFeatures(), "v").render(context).user
    exemplos = user.split(MARK_EXAMPLES, 1)[1]
    assert '"limitacao": null' in exemplos
    assert '"limitacao": "' in exemplos  # e um caso com limitação declarada


def test_formato_embute_schema_json_e_pede_json_puro(context):
    user = PromptBuilder(PromptFeatures(), "v").render(context).user
    formato = user.split(MARK_FORMAT, 1)[1]
    assert json.dumps(prompt_schema(), ensure_ascii=False, indent=2) in formato
    assert "SOMENTE o JSON" in formato
    assert "COPIADO LITERALMENTE" in user
    assert "[p. N]" in user


def test_prompt_schema_exige_limitacao_e_nao_tem_titulos():
    schema = prompt_schema()
    assert "limitacao" in schema["required"]
    assert "title" not in json.dumps(schema)


def test_cot_muda_formato_para_envelope(context):
    cot = PromptBuilder(FULL, "cot").render(context).user
    assert '"raciocinio"' in cot and '"ficha"' in cot
    # os exemplos few-shot também passam a usar o envelope
    exemplos = cot.split(MARK_EXAMPLES, 1)[1]
    assert '"raciocinio":' in exemplos


def test_template_sha_estavel_e_distinto_por_variante():
    a = PromptBuilder(PromptFeatures(), "a")
    b = PromptBuilder(PromptFeatures(few_shot=False), "b")
    assert a.template_sha() == PromptBuilder(PromptFeatures(), "a").template_sha()
    assert a.template_sha() != b.template_sha()


def test_render_registra_variante_e_features(context):
    feats = PromptFeatures(few_shot=False)
    r = PromptBuilder(feats, "v_sem_fewshot").render(context)
    assert r.variant == "v_sem_fewshot"
    assert r.features == feats
    assert len(r.sha256()) == 12
