"""Clientes de modelo de linguagem. Todos implementam :class:`ficha.types.LLMClient`.

- :mod:`ficha.llm.fake`      — determinístico, sem rede/GPU; usado em testes e no modo de ensaio.
- :mod:`ficha.llm.qwen_local` — Qwen2.5-Instruct via ``transformers`` (Colab T4).
- :mod:`ficha.llm.anthropic_client` / :mod:`ficha.llm.openai_compat` — por API.

Use :func:`ficha.llm.factory.build_client` para instanciar a partir de
:class:`ficha.config.Settings`.
"""

from ficha.llm.fake import FakeLLM

__all__ = ["FakeLLM"]
