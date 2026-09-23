"""Clientes de modelo de linguagem. Todos implementam :class:`ficha.types.LLMClient`.

- :mod:`ficha.llm.fake`      — determinístico, sem rede/GPU; usado em testes e no modo de ensaio.
- :mod:`ficha.llm.qwen_local` — Qwen2.5-Instruct via ``transformers`` (Colab T4).
- :mod:`ficha.llm.anthropic_client` / :mod:`ficha.llm.openai_compat` — por API.
- :mod:`ficha.llm.tokens`    — contagem de tokens (real via tokenizador, ou aproximação declarada).

Use :func:`ficha.llm.factory.build_client` para instanciar a partir de
:class:`ficha.config.Settings`. Nenhum SDK pesado (``torch``, ``transformers``, ``anthropic``,
``openai``) é importado aqui: os imports são preguiçosos, dentro de cada cliente.
"""

from ficha.llm.factory import ClientConfigError, build_client
from ficha.llm.fake import FakeBehavior, FakeLLM
from ficha.llm.tokens import HFTokenCounter, approx_token_count

__all__ = [
    "ClientConfigError",
    "FakeBehavior",
    "FakeLLM",
    "HFTokenCounter",
    "approx_token_count",
    "build_client",
]
