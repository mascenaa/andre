"""Engenharia de prompt (Seção 4.2): técnicas identificáveis, ligáveis e comparáveis.

- :mod:`ficha.prompts.builder`  — :class:`PromptBuilder`, um bloco com cabeçalho por técnica.
- :mod:`ficha.prompts.examples` — exemplos few-shot (um deles com ``limitacao: null``).
- :mod:`ficha.prompts.variants` — :data:`VARIANTS`, o par obrigatório e :func:`describe_diff`.
"""

from ficha.prompts.builder import (
    ARTICLE_CLOSE,
    ARTICLE_OPEN,
    FEATURE_MARKERS,
    PromptBuilder,
)
from ficha.prompts.examples import FEW_SHOT_EXAMPLES, FewShotExample
from ficha.prompts.variants import (
    DEFAULT_VARIANT,
    MANDATORY_PAIR,
    VARIANTS,
    build_prompt,
    describe_diff,
    feature_diff,
)

__all__ = [
    "ARTICLE_CLOSE",
    "ARTICLE_OPEN",
    "DEFAULT_VARIANT",
    "FEATURE_MARKERS",
    "FEW_SHOT_EXAMPLES",
    "MANDATORY_PAIR",
    "VARIANTS",
    "FewShotExample",
    "PromptBuilder",
    "build_prompt",
    "describe_diff",
    "feature_diff",
]
