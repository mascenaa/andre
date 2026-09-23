"""Segmentação de páginas em chunks com tamanho e sobreposição declarados (Seção 4.1).

Invariantes (testadas):

- ``page.text[c.start:c.end] == c.text`` para todo chunk — o offset aponta para o texto real;
- os chunks cobrem todo o texto não-branco da página;
- chunks consecutivos se sobrepõem em **pelo menos** ``overlap`` caracteres (um pouco mais
  quando o início é recuado até o começo da palavra);
- um chunk nunca atravessa páginas: a página de cada trecho é exata, o que sustenta o
  ``evidencia.pagina`` da ficha.

O corte prefere, em ordem: fim de parágrafo, fim de sentença, espaço entre palavras, e só em
último caso o meio de uma palavra. A busca pelo corte fica na segunda metade da janela, para
nenhum chunk ficar menor que ``size / 2`` (exceto o último da página).
"""

from __future__ import annotations

import re

from ficha.types import Chunk, Document, Page

_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=\s)")
_WHITESPACE_RE = re.compile(r"\s")


def find_cut(text: str, lo: int, hi: int) -> int:
    """Melhor posição de corte em ``text[lo:hi]`` (exclusivo), preferindo fronteiras naturais.

    Devolve um offset em ``(lo, hi]``. Se ``hi`` já é o fim do texto, devolve ``hi``.
    """
    if hi >= len(text):
        return len(text)
    window = text[lo:hi]
    for pattern, use_end in ((_PARAGRAPH_BREAK_RE, False), (_SENTENCE_END_RE, True)):
        matches = list(pattern.finditer(window))
        if matches:
            m = matches[-1]
            cut = lo + (m.end() if use_end else m.start())
            if cut > lo:
                return cut
    spaces = [m.start() for m in _WHITESPACE_RE.finditer(window)]
    if spaces and lo + spaces[-1] > lo:
        return lo + spaces[-1]
    return hi


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Avança ``start`` e recua ``end`` sobre espaços, sem inverter o intervalo."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


_SENTENCE_START_RE = re.compile(r"(?:[.!?][\"')\]]*\s+|\n\s*\n)(?=\S)")


def next_start(text: str, start: int, end: int, overlap: int) -> int:
    """Início do próximo chunk: pelo menos ``overlap`` caracteres antes de ``end``.

    Prefere o começo de uma sentença entre ``overlap`` e ``1,5 x overlap`` antes do corte
    (o chunk seguinte começa legível); senão, recua até o começo da palavra. Sempre avança
    em relação a ``start``.
    """
    target = end - overlap
    if overlap > 0:
        lo = max(start + 1, end - overlap - overlap // 2)
        starts = [m.end() for m in _SENTENCE_START_RE.finditer(text, lo, target + 1)]
        starts = [s for s in starts if lo <= s <= target]
        if starts:
            return starts[-1]
    nxt = target
    while nxt > start + 1 and not text[nxt - 1].isspace():
        nxt -= 1
    if nxt <= start:
        nxt = target if target > start else end
    return nxt


def chunk_page(page: Page, arquivo: str, size: int, overlap: int) -> list[Chunk]:
    """Divide uma página em chunks de até ``size`` caracteres com ``overlap`` de sobreposição.

    Raises:
        ValueError: ``size <= 0`` ou ``overlap`` fora de ``[0, size / 2)``.

    """
    if size <= 0:
        raise ValueError(f"size deve ser > 0, recebido {size}")
    if not 0 <= overlap < size / 2:
        raise ValueError(f"overlap deve estar em [0, size/2), recebido {overlap} (size={size})")
    text = page.text
    n = len(text)
    chunks: list[Chunk] = []
    start, _ = _trim(text, 0, n)
    while start < n:
        hard_end = min(start + size, n)
        end = find_cut(text, start + size // 2, hard_end) if hard_end < n else n
        s, e = _trim(text, start, end)
        if e > s:
            chunks.append(Chunk(arquivo=arquivo, page=page.number, text=text[s:e], start=s, end=e))
        if end >= n:
            break
        start, _ = _trim(text, next_start(text, start, end, overlap), n)
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Chunks de todas as páginas, na ordem do artigo (página, offset)."""
    out: list[Chunk] = []
    for page in doc.pages:
        out.extend(chunk_page(page, doc.arquivo, size, overlap))
    return out
