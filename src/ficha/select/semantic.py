"""Estratégia 3 da Seção 4.1: trechos localizados por similaridade semântica.

Para cada artigo: segmenta em chunks (``chunking``), vetoriza chunks e consultas com o mesmo
``Embedder``, e para **cada consulta** pega os ``top_k`` chunks mais próximos (cosseno). A união
(sem duplicatas) é o contexto.

Decisões:

- **Consultas em português**, uma ou duas por campo da ficha, incluindo literalmente as duas
  do enunciado ("como o desempenho foi medido", "o que este trabalho não consegue fazer").
  Por isso o embedder padrão é multilíngue.
- **Ordem do artigo**: os chunks escolhidos são reordenados por ``(página, início)``. O modelo
  lê um texto que flui como o original (e não uma colagem por relevância), e os marcadores
  ``[p. N]`` aparecem em ordem crescente.
- **Chunks sobrepostos são unidos** (``merge_overlaps``): dois chunks vizinhos escolhidos
  compartilham ``overlap`` caracteres; uni-los evita mandar o mesmo texto duas vezes (custo,
  Seção 4.5) sem perder o offset exato.
- ``Chunk.score`` = maior similaridade entre as consultas que o escolheram;
  ``Chunk.label`` = a consulta dessa maior similaridade.
- ``max_chars`` (opcional) corta os chunks de menor score até caber no orçamento.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ficha.select.chunking import chunk_document
from ficha.select.embedders import Embedder
from ficha.types import Chunk, Context, Document

DEFAULT_QUERIES: tuple[str, ...] = (
    # problema
    "qual problema este trabalho tenta resolver e qual é o objetivo do estudo",
    # dados
    "que dados foram usados: fonte, período e volume do conjunto de dados",
    # método
    "qual é o método ou a abordagem principal proposta pelos autores",
    # métrica (a primeira é literal do enunciado)
    "como o desempenho foi medido",
    "métricas de avaliação e resultados numéricos obtidos",
    # limitação declarada pelos autores (a primeira é literal do enunciado)
    "o que este trabalho não consegue fazer",
    "limitações declaradas pelos autores e ameaças à validade do estudo",
    # evidência: afirmações centrais que sustentam a ficha
    "principais resultados e contribuições do artigo",
)
"""Consultas padrão, em português, cobrindo os seis campos da ficha."""

CACHE_SIZE = 64
"""Quantos artigos manter com embeddings de chunks em memória (≈ 19 artigos reais)."""


class SemanticSelector:
    """Top-k chunks por consulta, união sem duplicatas, reordenados por posição no artigo."""

    def __init__(
        self,
        embedder: Embedder,
        queries: Sequence[str] = DEFAULT_QUERIES,
        top_k: int = 6,
        chunk_size: int = 1200,
        overlap: int = 200,
        max_chars: int | None = None,
        *,
        merge_overlaps: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k deve ser >= 1, recebido {top_k}")
        if not queries:
            raise ValueError("É preciso ao menos uma consulta")
        self.embedder = embedder
        self.queries = tuple(queries)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chars = max_chars
        self.merge_overlaps = merge_overlaps
        self._query_vecs: np.ndarray | None = None
        self._chunk_cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return "semantic"

    def _queries_matrix(self) -> np.ndarray:
        if self._query_vecs is None:
            self._query_vecs = self.embedder.embed(list(self.queries))
        return self._query_vecs

    def describe(self) -> dict[str, object]:
        """Parâmetros da estratégia (sem contagens), para registro em ``Context.params``."""
        return {
            "embedder": self.embedder.name,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "max_chars": self.max_chars,
            "merge_overlaps": self.merge_overlaps,
            "queries": list(self.queries),
        }

    def _params(self, n_total: int, n_selected: int) -> dict[str, object]:
        return {**self.describe(), "n_chunks_total": n_total, "n_chunks_selected": n_selected}

    def _embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Vetores dos chunks, com cache por conteúdo (o ``hybrid`` e a 4.4c reusam)."""
        key = hashlib.sha256(
            "\x00".join(f"{c.page}:{c.start}:{c.text}" for c in chunks).encode()
        ).hexdigest()
        if key not in self._chunk_cache:
            if len(self._chunk_cache) >= CACHE_SIZE:
                self._chunk_cache.pop(next(iter(self._chunk_cache)))
            self._chunk_cache[key] = self.embedder.embed([c.text for c in chunks])
        return self._chunk_cache[key]

    def rank(self, doc: Document) -> list[Chunk]:
        """Chunks escolhidos (antes de orçamento e união), com ``score`` e ``label`` preenchidos."""
        return self._rank(chunk_document(doc, self.chunk_size, self.overlap))

    def _rank(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        sims = self._queries_matrix() @ self._embed_chunks(chunks).T
        k = min(self.top_k, len(chunks))
        best: dict[int, tuple[float, int]] = {}  # chunk -> (melhor score, consulta)
        for qi in range(sims.shape[0]):
            # desempate estável: maior similaridade, depois menor índice (ordem do artigo)
            order = sorted(range(len(chunks)), key=lambda ci: (-float(sims[qi, ci]), ci))[:k]
            for ci in order:
                score = float(sims[qi, ci])
                if ci not in best or score > best[ci][0]:
                    best[ci] = (score, qi)
        return [
            Chunk(
                arquivo=c.arquivo,
                page=c.page,
                text=c.text,
                start=c.start,
                end=c.end,
                score=round(best[ci][0], 6),
                label=self.queries[best[ci][1]],
            )
            for ci, c in enumerate(chunks)
            if ci in best
        ]

    def _apply_budget(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.max_chars is None:
            return chunks
        kept: list[Chunk] = []
        total = 0
        for c in sorted(chunks, key=lambda c: (-(c.score or 0.0), c.page, c.start)):
            if total + len(c.text) <= self.max_chars:
                kept.append(c)
                total += len(c.text)
        return kept

    @staticmethod
    def _merge(doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Une chunks sobrepostos da mesma página (offsets exatos, melhor score/rótulo)."""
        merged: list[Chunk] = []
        for c in chunks:
            prev = merged[-1] if merged else None
            if prev is not None and prev.page == c.page and c.start <= prev.end:
                end = max(prev.end, c.end)
                top = prev if (prev.score or 0.0) >= (c.score or 0.0) else c
                merged[-1] = Chunk(
                    arquivo=prev.arquivo,
                    page=prev.page,
                    text=doc.page(prev.page).text[prev.start : end],
                    start=prev.start,
                    end=end,
                    score=top.score,
                    label=top.label,
                )
            else:
                merged.append(c)
        return merged

    def select(self, doc: Document) -> Context:
        """Seleciona os trechos de ``doc`` mais próximos das consultas."""
        all_chunks = chunk_document(doc, self.chunk_size, self.overlap)
        n_total = len(all_chunks)
        chosen = self._apply_budget(self._rank(all_chunks))
        chosen.sort(key=lambda c: (c.page, c.start))
        n_selected = len(chosen)
        if self.merge_overlaps:
            chosen = self._merge(doc, chosen)
        return Context(
            arquivo=doc.arquivo,
            strategy=self.name,
            chunks=tuple(chosen),
            params=self._params(n_total, n_selected),
        )
