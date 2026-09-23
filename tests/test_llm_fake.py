"""Comportamento do FakeLLM que depende do formato do prompt (delimitadores × ``Artigo:``)."""

from __future__ import annotations

from ficha.llm.fake import FakeLLM
from ficha.prompts import build_prompt
from ficha.schema import FichaExtraida


def _ficha(fake_llm, context, params, variant):
    p = build_prompt(variant).render(context)
    return FichaExtraida.model_validate_json(fake_llm.generate(p.system, p.user, params).text)


def test_fake_acha_o_artigo_com_e_sem_delimitadores(fake_llm, context, params):
    rendered = context.render()
    for variant in ("v_full", "v_sem_delimitadores"):
        ficha = _ficha(fake_llm, context, params, variant)
        # O trecho vem do artigo, nunca dos exemplos few-shot (que vêm antes no prompt).
        assert ficha.evidencia.trecho in rendered


def test_fake_ignora_mencao_aos_delimitadores_fora_do_bloco(params):
    user = (
        "Use <<<ARTIGO>>> e <<<FIM_ARTIGO>>> como marcadores.\n"
        "<<<ARTIGO>>>\n[p. 2]\nThis sentence is long enough to be chosen as the evidence text."
        "\n<<<FIM_ARTIGO>>>"
    )
    comp = FakeLLM().generate(None, user, params)
    ficha = FichaExtraida.model_validate_json(comp.text)
    assert ficha.evidencia.pagina == 2


def test_truncate_rate_corta_o_json_e_marca_length(context, params):
    from ficha.extract.parser import parse_completion
    from ficha.llm.fake import FakeBehavior
    from ficha.types import ParseStatus

    p = build_prompt("v_full").render(context)
    cheio = FakeLLM().generate(p.system, p.user, params)
    cortado = FakeLLM(behavior=FakeBehavior(truncate_rate=1.0)).generate(p.system, p.user, params)
    assert cheio.raw["finish_reason"] == "stop"
    assert cortado.raw["finish_reason"] == "length"
    assert cheio.text.startswith(cortado.text)
    assert len(cortado.text) < len(cheio.text)
    assert cortado.usage.output_tokens < cheio.usage.output_tokens
    assert parse_completion(cortado.text).status is ParseStatus.FAILED


def test_truncate_rate_zero_nao_muda_saida_padrao(context, params):
    from ficha.llm.fake import FakeBehavior

    p = build_prompt("v_full").render(context)
    a = FakeLLM().generate(p.system, p.user, params).text
    b = FakeLLM(behavior=FakeBehavior(truncate_rate=0.0)).generate(p.system, p.user, params).text
    assert a == b


def test_truncate_rate_respeita_temperature_noise(context):
    from ficha.llm.fake import FakeBehavior
    from ficha.types import GenerationParams

    p = build_prompt("v_full").render(context)
    quente = GenerationParams(temperature=1.0, seed=42)
    sem_ruido = FakeLLM(behavior=FakeBehavior(temperature_noise=False))
    assert all(
        sem_ruido.generate(p.system, p.user, quente).raw["finish_reason"] == "stop"
        for _ in range(20)
    )
    com_ruido = FakeLLM(behavior=FakeBehavior(truncate_rate=0.5))
    cortes = [com_ruido.generate(p.system, p.user, quente).raw["finish_reason"] for _ in range(40)]
    assert cortes.count("length") > 0
