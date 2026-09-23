"""Estratégias de seleção de contexto (Seção 4.1). Todas implementam ``ContextSelector``.

- :class:`FirstPagesSelector`      — as N primeiras páginas (``first_pages``).
- :class:`KeywordSectionSelector`  — seções localizadas por palavra-chave (``keyword``).
- :class:`SemanticSelector`        — chunks por similaridade de embeddings (``semantic``).
- :class:`HybridSelector`          — semântica + seções de fechamento (``hybrid``).
- :mod:`ficha.select.diagnostics`  — recall offline das frases de limitação por estratégia.
- :mod:`ficha.select.chunking`     — segmentação com tamanho/sobreposição declarados.
- :mod:`ficha.select.embedders`    — ``SentenceTransformerEmbedder`` e ``HashingEmbedder``.
- :func:`build_selector`           — nome → seletor configurado por ``Settings``.

A decisão e as alternativas estão em ``docs/adr/0002-selecao-de-contexto.md``.
"""

from ficha.select.chunking import chunk_document, chunk_page
from ficha.select.diagnostics import (
    LIMITATION_SENTENCE_RE,
    LimitationRecall,
    limitation_sentence_recall,
    limitation_sentences,
    recall_table,
)
from ficha.select.embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from ficha.select.first_pages import FirstPagesSelector
from ficha.select.hybrid import HybridSelector
from ficha.select.keyword import KeywordSectionSelector
from ficha.select.registry import SELECTOR_NAMES, build_selector
from ficha.select.semantic import DEFAULT_QUERIES, SemanticSelector

__all__ = [
    "DEFAULT_QUERIES",
    "LIMITATION_SENTENCE_RE",
    "SELECTOR_NAMES",
    "Embedder",
    "FirstPagesSelector",
    "HashingEmbedder",
    "HybridSelector",
    "KeywordSectionSelector",
    "LimitationRecall",
    "SemanticSelector",
    "SentenceTransformerEmbedder",
    "build_selector",
    "chunk_document",
    "chunk_page",
    "limitation_sentence_recall",
    "limitation_sentences",
    "recall_table",
]
