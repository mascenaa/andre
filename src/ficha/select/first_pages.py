"""Estratégia 1 da Seção 4.1: as primeiras páginas do artigo.

Premissa: resumo e introdução bastam para ``problema``, ``dados`` e ``metodo``. Limite
conhecido: métrica detalhada, resultados e, principalmente, a seção de limitações costumam
estar no fim do artigo — esta estratégia tende a devolver ``limitacao = null`` mesmo quando os
autores declaram alguma. A auditoria 4.4c mede exatamente isso.
"""

from __future__ import annotations

from ficha.select.chunking import find_cut
from ficha.types import Chunk, Context, Document


class FirstPagesSelector:
    """As ``n_pages`` primeiras páginas, inteiras, opcionalmente limitadas a ``max_chars``.

    Páginas vazias não geram chunk. Com ``max_chars``, a última página é cortada numa fronteira
    natural (parágrafo, sentença ou espaço) para não passar do limite.
    """

    def __init__(self, n_pages: int = 3, max_chars: int | None = None) -> None:
        if n_pages < 1:
            raise ValueError(f"n_pages deve ser >= 1, recebido {n_pages}")
        if max_chars is not None and max_chars < 1:
            raise ValueError(f"max_chars deve ser >= 1, recebido {max_chars}")
        self.n_pages = n_pages
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "first_pages"

    def select(self, doc: Document) -> Context:
        """Seleciona as primeiras páginas de ``doc``."""
        chunks: list[Chunk] = []
        budget = self.max_chars
        truncated = False
        for page in doc.pages[: self.n_pages]:
            text = page.text
            if not text.strip():
                continue
            end = len(text)
            if budget is not None and end > budget:
                end = find_cut(text, budget // 2, budget) if budget > 1 else budget
                truncated = True
            chunks.append(
                Chunk(
                    arquivo=doc.arquivo,
                    page=page.number,
                    text=text[:end],
                    start=0,
                    end=end,
                    label=f"página {page.number}",
                )
            )
            if budget is not None:
                budget -= end
                if budget <= 0 or truncated:
                    break
        return Context(
            arquivo=doc.arquivo,
            strategy=self.name,
            chunks=tuple(chunks),
            params={"n_pages": self.n_pages, "max_chars": self.max_chars, "truncated": truncated},
        )
