"""Contagem de tokens.

Duas formas, com papéis diferentes:

- :func:`approx_token_count` é uma **premissa declarada**: ~4 caracteres por token, regra de
  bolso para texto em inglês com tokenizadores BPE modernos. É usada quando o tokenizador real
  não está disponível (testes, modo de ensaio, clientes de API sem endpoint de contagem) e o
  relatório precisa dizer isso explicitamente junto do número.
- :class:`HFTokenCounter` usa o tokenizador **real** do modelo (``transformers.AutoTokenizer``).
  No Colab, com o Qwen2.5, é essa a contagem que alimenta o custo (Seção 4.5).

``transformers`` é importado de forma preguiçosa: o pacote funciona (e os testes passam) sem ele.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

DEFAULT_CHARS_PER_TOKEN = 4.0
"""Premissa declarada da aproximação: caracteres por token."""


def approx_token_count(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Aproxima o número de tokens de ``text`` como ``len(text) / chars_per_token``.

    É uma premissa, não uma medida: o relatório deve declará-la sempre que for usada.
    Texto vazio conta como 0 tokens; texto não vazio conta no mínimo 1.
    """
    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token deve ser > 0, recebido {chars_per_token}")
    if not text:
        return 0
    return max(1, round(len(text) / chars_per_token))


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str) -> Any:
    """Carrega (uma vez por processo) o tokenizador do Hugging Face."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            'Contagem com tokenizador real exige transformers: pip install "ficha[local]"'
        ) from exc
    return AutoTokenizer.from_pretrained(model_name)


class HFTokenCounter:
    """Conta tokens com o tokenizador real de um modelo do Hugging Face (ex.: Qwen2.5).

    O tokenizador só é baixado/carregado na primeira contagem e fica em cache no processo.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def tokenizer(self) -> Any:
        """Tokenizador carregado (preguiçosamente)."""
        return _load_tokenizer(self.model_name)

    def count(self, text: str) -> int:
        """Número exato de tokens de ``text`` (sem tokens especiais)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def __call__(self, text: str) -> int:
        return self.count(text)

    def __repr__(self) -> str:
        return f"HFTokenCounter(model_name={self.model_name!r})"
