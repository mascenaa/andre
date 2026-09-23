"""Nomeação dos arquivos de entrega (Seção 5: ``AtividadeI_<sobrenomes>``).

O enunciado exige que os três arquivos (notebook, tabela e relatório) sigam o padrão
``AtividadeI_<sobrenomes>``. Centralizar a regra aqui evita que cada entregável invente a
sua variação (acentos, espaços, ordem dos sobrenomes).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

DELIVERY_PREFIX = "AtividadeI"
"""Prefixo fixo exigido pelo enunciado."""

PLACEHOLDER_BASENAME = f"{DELIVERY_PREFIX}_SOBRENOMES"
"""Nome usado enquanto o grupo não preencheu os integrantes."""


def _ascii_token(text: str) -> str:
    """Remove acentos e caracteres fora de ``[A-Za-z0-9]`` (nomes de arquivo portáveis)."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]", "", ascii_only)


def surname_of(full_name: str) -> str:
    """Último sobrenome de um nome completo, sem acentos (``"João da Silva"`` → ``"Silva"``)."""
    parts = [p for p in full_name.strip().split() if p]
    if not parts:
        raise ValueError("Nome vazio não tem sobrenome.")
    return _ascii_token(parts[-1])


def delivery_basename(integrantes: Sequence[str]) -> str:
    """Nome-base dos arquivos de entrega a partir dos nomes completos dos integrantes.

    A ordem dos sobrenomes segue a ordem em que os integrantes foram declarados.
    Sem integrantes (ou só placeholders ``<<...>>``), devolve :data:`PLACEHOLDER_BASENAME`.

    >>> delivery_basename(["Ana Souza", "João da Conceição"])
    'AtividadeI_Souza_Conceicao'
    """
    reais = [n for n in integrantes if n.strip() and "<<" not in n]
    if not reais:
        return PLACEHOLDER_BASENAME
    sobrenomes = [s for s in (surname_of(n) for n in reais) if s]
    return "_".join([DELIVERY_PREFIX, *sobrenomes])
