"""Estratégia híbrida: semântica + seções de fechamento por palavra-chave (revisão da 4.1).

Motivo (execução real, ver ADR 0002, "Revisão após execução real"): com ``semantic`` sozinho,
só ~14% das frases de limitação declarada chegavam ao contexto, porque elas ficam nas seções
finais (Discussion, Limitations, Conclusions) e as consultas semânticas as perdiam para trechos
de introdução e resultados. O modelo não pode declarar o que não recebe.

O híbrido une cinco fontes, nesta **prioridade de orçamento**:

a) janelas após cabeçalhos de **limitação** (``Limitations``, ``Caveats``, ``Threats to
   Validity``, ``Current limitations``) — é o campo que mais falhou;
b) até ``cue_windows`` janelas curtas em torno de frases com **pistas lexicais** de limitação
   (:data:`LIMITATION_CUE_RE`) — muitos artigos admitem limitações no meio do texto, sem
   cabeçalho próprio (medido: só 11 de 63 frases candidatas estão em seções de fechamento);
c) chunks semânticos, por score decrescente — trazem problema, dados, método e métrica;
d) janelas após **Discussion / Conclusion(s) / Summary and ... / Outlook / Future Work**;
e) opcionalmente, as ``tail_pages`` últimas páginas antes das referências, em chunks.

O abstract e a introdução não entram por palavra-chave: a semântica já os traz. O orçamento
conta só caracteres **novos**: um trecho que se sobrepõe ao que já entrou só paga pela parte
inédita. No fim, trechos sobrepostos ou vizinhos da mesma página são unidos (offsets exatos) e
ordenados por (página, início). ``params["chars_por_fonte"]`` registra quanto veio de cada fonte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ficha.select.chunking import chunk_page
from ficha.select.keyword import REFERENCES_RE, KeywordSectionSelector
from ficha.select.semantic import SemanticSelector
from ficha.types import Chunk, Context, Document, Page

LIMITATION_SECTIONS: tuple[str, ...] = ("limitations",)
CLOSING_SECTIONS_NO_LIMITATIONS: tuple[str, ...] = ("discussion", "conclusion", "outlook")
SOURCES: tuple[str, ...] = ("limitations", "cues", "semantic", "closing", "tail")

LIMITATION_CUE_RE = re.compile(
    r"\blimitations?\b|\bcaveats?\b|\bdrawbacks?\b|\bshortcomings?\b|beyond the scope"
    r"|future (?:work|studies|research)|further (?:work|research|investigation)"
    r"|(?:do|does|did|could|can) not (?:account|consider|include|capture|address"
    r"|generali[sz]e|resolve|distinguish)"
    r"|may not (?:generali[sz]e|hold|be representative)|restricted to"
    r"|small (?:sample|number of)|we (?:did|do) not|not (?:taken|been) into account"
    r"|lack of|uncertaint(?:y|ies) (?:in|of|remain)",
    re.IGNORECASE,
)
"""Pistas lexicais de limitação usadas para **selecionar** (fonte ``cues``).

É deliberadamente um conjunto diferente de ``diagnostics.LIMITATION_SENTENCE_RE`` (que
**mede**): as duas listas só compartilham ``limitation``/``caveat``/``drawback``/``shortcoming``/
``beyond the scope``. Ver ADR 0002 sobre o risco de circularidade na medição."""
STRONG_CUE_RE = re.compile(
    r"\blimitations?\b|\bcaveats?\b|\bdrawbacks?\b|\bshortcomings?\b|beyond the scope",
    re.IGNORECASE,
)
"""Pistas fortes: frases com elas têm prioridade sobre as demais pistas."""
_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+[\"')\]]*|$)")
CUE_WINDOW_CHARS = 600
"""Tamanho máximo de uma janela de pista (a frase com a pista e as vizinhas)."""


@dataclass(frozen=True, slots=True)
class _Piece:
    page: int
    start: int
    end: int
    source: str
    label: str | None
    score: float | None


class _Coverage:
    """Intervalos já cobertos por página, para contar só caracteres novos."""

    def __init__(self) -> None:
        self._spans: dict[int, list[tuple[int, int]]] = {}

    def new_chars(self, page: int, start: int, end: int) -> int:
        covered = 0
        for s, e in self._spans.get(page, []):
            covered += max(0, min(e, end) - max(s, start))
        return (end - start) - covered  # spans guardados são disjuntos

    def add(self, page: int, start: int, end: int) -> None:
        spans = [*self._spans.get(page, []), (start, end)]
        spans.sort()
        merged: list[tuple[int, int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self._spans[page] = merged


def tail_pages_before_references(doc: Document, n: int) -> list[Page]:
    """As ``n`` últimas páginas com texto **antes** da seção de referências.

    A página em que ``References`` começa entra só até o cabeçalho. Sem ``References``,
    usa as ``n`` últimas páginas do documento.
    """
    if n <= 0:
        return []
    body: list[Page] = []
    for page in doc.pages:
        m = REFERENCES_RE.search(page.text)
        if m:
            if page.text[: m.start()].strip():
                body.append(Page(page.number, page.text[: m.start()]))
            break
        body.append(page)
    return [p for p in body if p.text.strip()][-n:]


class HybridSelector:
    """Semântica + janelas de seções de fechamento, com orçamento por prioridade."""

    def __init__(
        self,
        semantic: SemanticSelector,
        keyword: KeywordSectionSelector,
        max_chars: int | None = 12000,
        tail_pages: int = 0,
        cue_windows: int = 10,
    ) -> None:
        if cue_windows < 0:
            raise ValueError(f"cue_windows deve ser >= 0, recebido {cue_windows}")
        if tail_pages < 0:
            raise ValueError(f"tail_pages deve ser >= 0, recebido {tail_pages}")
        self.semantic = semantic
        self.keyword = keyword
        self.max_chars = max_chars
        self.tail_pages = tail_pages
        self.cue_windows = cue_windows

    @property
    def name(self) -> str:
        return "hybrid"

    # ------------------------------------------------------------------ candidatos
    def _candidates(self, doc: Document) -> list[_Piece]:
        """Todos os candidatos, já na ordem de prioridade (a → e)."""

        def from_chunks(chunks: list[Chunk], source: str) -> list[_Piece]:
            return [_Piece(c.page, c.start, c.end, source, c.label, c.score) for c in chunks]

        lim = from_chunks(self.keyword.section_windows(doc, LIMITATION_SECTIONS), "limitations")
        sem = sorted(
            from_chunks(self.semantic.rank(doc), "semantic"),
            key=lambda p: (-(p.score or 0.0), p.page, p.start),
        )
        closing = from_chunks(
            self.keyword.section_windows(doc, CLOSING_SECTIONS_NO_LIMITATIONS), "closing"
        )
        tail: list[_Piece] = []
        for page in tail_pages_before_references(doc, self.tail_pages):
            chunks = chunk_page(page, doc.arquivo, self.semantic.chunk_size, self.semantic.overlap)
            tail.extend(
                _Piece(c.page, c.start, c.end, "tail", f"página final {c.page}", None)
                for c in chunks
            )
        cues = self._cue_pieces(doc)
        return lim + cues + sem + closing + tail

    def _cue_pieces(self, doc: Document) -> list[_Piece]:
        """Janelas em torno de frases com pistas de limitação (até ``cue_windows``).

        Só o corpo do artigo (antes de ``References``). Frases com pista forte vêm primeiro;
        dentro de cada nível, na ordem do artigo. A janela é a frase com a pista mais as
        vizinhas, até :data:`CUE_WINDOW_CHARS`.
        """
        if self.cue_windows <= 0:
            return []
        found: list[tuple[int, int, int, int, str]] = []  # (nível, página, início, fim, pista)
        for page in tail_pages_before_references(doc, doc.n_pages):
            text = page.text
            sentences = [(m.start(), m.end()) for m in _SENT_RE.finditer(text) if m.group().strip()]
            for i, (s, e) in enumerate(sentences):
                cue = LIMITATION_CUE_RE.search(text, s, e)
                if not cue:
                    continue
                lo, hi = s, e
                if i > 0 and hi - sentences[i - 1][0] <= CUE_WINDOW_CHARS:
                    lo = sentences[i - 1][0]
                if i + 1 < len(sentences) and sentences[i + 1][1] - lo <= CUE_WINDOW_CHARS:
                    hi = sentences[i + 1][1]
                lo = min(max(lo, 0), len(text))
                while lo < hi and text[lo].isspace():
                    lo += 1
                level = 0 if STRONG_CUE_RE.search(text, s, e) else 1
                found.append((level, page.number, lo, hi, cue.group(0)))
        found.sort(key=lambda f: (f[0], f[1], f[2]))
        return [
            _Piece(page, lo, hi, "cues", f"pista: {cue.lower()}", None)
            for _, page, lo, hi, cue in found[: self.cue_windows]
        ]

    # ------------------------------------------------------------------ seleção
    def select(self, doc: Document) -> Context:
        """Seleciona por prioridade dentro de ``max_chars`` e une sobreposições."""
        coverage = _Coverage()
        chosen: list[_Piece] = []
        chars_by_source = dict.fromkeys(SOURCES, 0)
        skipped_by_source = dict.fromkeys(SOURCES, 0)
        total = 0
        for piece in self._candidates(doc):
            new = coverage.new_chars(piece.page, piece.start, piece.end)
            if new <= 0:
                continue
            if self.max_chars is not None and total + new > self.max_chars:
                skipped_by_source[piece.source] += 1
                continue
            coverage.add(piece.page, piece.start, piece.end)
            chosen.append(piece)
            chars_by_source[piece.source] += new
            total += new

        chunks = self._merge(doc, chosen)
        params: dict[str, object] = {
            "max_chars": self.max_chars,
            "tail_pages": self.tail_pages,
            "cue_windows": self.cue_windows,
            "priority": list(SOURCES),
            "limitation_sections": list(LIMITATION_SECTIONS),
            "closing_sections": list(CLOSING_SECTIONS_NO_LIMITATIONS),
            "keyword_window_chars": self.keyword.window_chars,
            "semantic": {
                k: v
                for k, v in self.semantic.describe().items()
                if k not in ("n_chunks_total", "n_chunks_selected")
            },
            "chars_por_fonte": chars_by_source,
            "candidatos_fora_do_orcamento": skipped_by_source,
            "n_pieces": len(chosen),
        }
        return Context(doc.arquivo, self.name, tuple(chunks), params)

    @staticmethod
    def _merge(doc: Document, pieces: list[_Piece]) -> list[Chunk]:
        """Une peças sobrepostas/encostadas da mesma página; rótulos e fontes concatenados."""
        merged: list[tuple[int, int, int, list[str], float | None]] = []
        for p in sorted(pieces, key=lambda x: (x.page, x.start, x.end)):
            tag = f"{p.source}: {p.label}" if p.label else p.source
            if merged and merged[-1][0] == p.page and p.start <= merged[-1][2]:
                page, s, e, tags, score = merged[-1]
                best = max((x for x in (score, p.score) if x is not None), default=None)
                if tag not in tags:
                    tags.append(tag)
                merged[-1] = (page, s, max(e, p.end), tags, best)
            else:
                merged.append((p.page, p.start, p.end, [tag], p.score))
        return [
            Chunk(
                arquivo=doc.arquivo,
                page=page,
                text=doc.page(page).text[s:e],
                start=s,
                end=e,
                score=score,
                label=" | ".join(tags),
            )
            for page, s, e, tags, score in merged
        ]
