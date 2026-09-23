"""Estratégias de seleção de contexto (Seção 4.1). Todas implementam ``ContextSelector``.

- :class:`FirstPagesSelector`      — as N primeiras páginas (``first_pages``).
- :class:`KeywordSectionSelector`  — seções localizadas por palavra-chave (``keyword``).
- :class:`SemanticSelector`        — chunks por similaridade de embeddings (``semantic``).
- :mod:`ficha.select.chunking`     — segmentação com tamanho/sobreposição declarados.
- :mod:`ficha.select.embedders`    — ``SentenceTransformerEmbedder`` e ``HashingEmbedder``.
- :func:`build_selector`           — nome → seletor configurado por ``Settings``.

A decisão e as alternativas estão em ``docs/adr/0002-selecao-de-contexto.md``.
"""

from ficha.select.chunking import chunk_document, chunk_page
from ficha.select.embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from ficha.select.first_pages import FirstPagesSelector
from ficha.select.keyword import KeywordSectionSelector
from ficha.select.registry import SELECTOR_NAMES, build_selector
from ficha.select.semantic import DEFAULT_QUERIES, SemanticSelector

__all__ = [
    "DEFAULT_QUERIES",
    "SELECTOR_NAMES",
    "Embedder",
    "FirstPagesSelector",
    "HashingEmbedder",
    "KeywordSectionSelector",
    "SemanticSelector",
    "SentenceTransformerEmbedder",
    "build_selector",
    "chunk_document",
    "chunk_page",
]
