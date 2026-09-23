"""Variantes do prompt para a medição obrigatória da Seção 4.2.

Cada variante difere de ``v_full`` em **uma** técnica, para que a diferença medida (taxa de
JSON válido de primeira, taxa de ``null`` correto, discordância entre fichas) possa ser
atribuída a ela.

- ``v_full``              — todas as técnicas exigidas; sem cadeia de raciocínio.
- ``v_sem_fewshot``       — par obrigatório de ``v_full``: mede o efeito do few-shot.
- ``v_sem_delimitadores`` — mede o efeito dos delimitadores (artigo após ``Artigo:``).
- ``v_cot``               — ``v_full`` + cadeia de raciocínio no campo ``raciocinio``.
"""

from __future__ import annotations

from dataclasses import asdict

from ficha.prompts.builder import PromptBuilder
from ficha.types import PromptFeatures

VARIANTS: dict[str, PromptFeatures] = {
    "v_full": PromptFeatures(
        role=True, delimiters=True, few_shot=True, abstention=True, chain_of_thought=False
    ),
    "v_sem_fewshot": PromptFeatures(
        role=True, delimiters=True, few_shot=False, abstention=True, chain_of_thought=False
    ),
    "v_sem_delimitadores": PromptFeatures(
        role=True, delimiters=False, few_shot=True, abstention=True, chain_of_thought=False
    ),
    "v_cot": PromptFeatures(
        role=True, delimiters=True, few_shot=True, abstention=True, chain_of_thought=True
    ),
}

DEFAULT_VARIANT = "v_full"
MANDATORY_PAIR: tuple[str, str] = ("v_full", "v_sem_fewshot")
"""As duas versões que a Seção 4.2 exige comparar (diferem só no few-shot)."""


def _features(v: str | PromptFeatures) -> PromptFeatures:
    if isinstance(v, PromptFeatures):
        return v
    try:
        return VARIANTS[v]
    except KeyError:
        raise KeyError(f"Variante desconhecida: {v!r}. Opções: {sorted(VARIANTS)}") from None


def feature_diff(a: str | PromptFeatures, b: str | PromptFeatures) -> dict[str, tuple[bool, bool]]:
    """Técnicas que mudam de ``a`` para ``b``: ``{técnica: (valor_em_a, valor_em_b)}``."""
    da, db = asdict(_features(a)), asdict(_features(b))
    return {k: (da[k], db[k]) for k in da if da[k] != db[k]}


def describe_diff(a: str | PromptFeatures, b: str | PromptFeatures) -> str:
    """Descrição legível do que muda entre duas variantes (para o notebook e o relatório)."""
    name_a = a if isinstance(a, str) else "A"
    name_b = b if isinstance(b, str) else "B"
    diff = feature_diff(a, b)
    if not diff:
        return f"{name_a} × {name_b}: mesmas técnicas"
    parts = [
        f"{k}: {'on' if va else 'off'} → {'on' if vb else 'off'}" for k, (va, vb) in diff.items()
    ]
    return f"{name_a} × {name_b}: " + "; ".join(parts)


def build_prompt(variant: str = DEFAULT_VARIANT) -> PromptBuilder:
    """``PromptBuilder`` de uma variante nomeada de :data:`VARIANTS`."""
    return PromptBuilder(features=_features(variant), variant=variant)
