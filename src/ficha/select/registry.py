"""Registro das estratégias de seleção: nome → seletor configurado a partir de ``Settings``.

Os nomes são os mesmos que vão para ``Context.strategy``, ``ExtractionRecord.strategy`` e o
``run_id`` — é por eles que a auditoria 4.4c compara estratégias.
"""

from __future__ import annotations

from ficha.config import Settings
from ficha.select.embedders import Embedder, SentenceTransformerEmbedder
from ficha.select.first_pages import FirstPagesSelector
from ficha.select.keyword import KeywordSectionSelector
from ficha.select.semantic import SemanticSelector
from ficha.types import ContextSelector

SELECTOR_NAMES: tuple[str, ...] = ("first_pages", "keyword", "semantic")
"""Estratégias disponíveis (Seção 4.1), na ordem do enunciado."""


def build_selector(
    name: str, settings: Settings, embedder: Embedder | None = None
) -> ContextSelector:
    """Constrói a estratégia ``name`` com os parâmetros de ``settings``.

    Para ``semantic``, se ``embedder`` for ``None`` usa :class:`SentenceTransformerEmbedder`
    com ``settings.embedding_model`` (carregado só na primeira chamada). Nos testes e no modo
    de ensaio, passe um :class:`ficha.select.embedders.HashingEmbedder`.

    Raises:
        ValueError: nome desconhecido.

    """
    if name == "first_pages":
        return FirstPagesSelector(n_pages=settings.first_pages_n)
    if name == "keyword":
        return KeywordSectionSelector()
    if name == "semantic":
        return SemanticSelector(
            embedder=embedder or SentenceTransformerEmbedder(settings.embedding_model),
            top_k=settings.semantic_top_k,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
            # orçamento opcional; campo proposto para ``Settings`` (ausente → sem limite)
            max_chars=getattr(settings, "semantic_max_chars", None),
        )
    raise ValueError(f"Estratégia desconhecida: {name!r}. Opções: {', '.join(SELECTOR_NAMES)}")
