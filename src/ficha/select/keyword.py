"""Estratégia 2 da Seção 4.1: seções localizadas por palavra-chave.

Procura cabeçalhos de seção (``3 Methods``, ``3. Methodology``, ``III. APPROACH``, ``Materials
and Methods``, ``Limitations``, ``Threats to Validity``...) e envia uma janela de texto depois de
cada um. A janela para no próximo cabeçalho (que terá a sua própria janela, se for de interesse)
e pode continuar na página seguinte, porque seções atravessam páginas.

Regras do casamento (para não confundir cabeçalho com frase comum):

- só no **início de linha/parágrafo** (o texto limpo tem um parágrafo por linha);
- com numeração (arábica, ``3.1``, romana ou letra), basta a palavra-chave depois do número;
- sem numeração, a palavra-chave precisa vir seguida de pontuação (``Abstract.``) ou fim de
  linha (``Limitations``) — ``Data come from...`` não é cabeçalho;
- a palavra-chave precisa começar com maiúscula;
- nada depois de ``References``/``Bibliography`` conta (entradas de referência como
  ``12. Data mining for...`` gerariam falsos positivos).

O orçamento ``max_chars`` é distribuído por prioridade: abstract/início primeiro (problema),
depois limitações (a decisão ``null`` depende de a seção estar ou não no contexto), método,
dados, resultados/avaliação e, por fim, discussão e conclusão.

Fallback declarado: se nenhuma seção for encontrada, comporta-se como ``first_pages(2)`` e
registra ``params["fallback"] = "first_pages(2)"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ficha.select.chunking import find_cut
from ficha.select.first_pages import FirstPagesSelector
from ficha.types import Chunk, Context, Document

SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", r"abstract"),
    ("limitations", r"limitations?(?:\s+and\s+future\s+work)?|threats\s+to\s+(?:the\s+)?validity"),
    (
        "methods",
        r"materials\s+and\s+methods|methods?|methodology|approach|proposed\s+(?:method|approach)",
    ),
    ("data", r"data\s+and\s+methods|data(?:sets?)?|data\s+(?:collection|description)"),
    ("results", r"results(?:\s+and\s+discussion)?"),
    ("experiments", r"experiments?|experimental\s+(?:setup|settings?|design|results)"),
    ("evaluation", r"evaluation"),
    ("discussion", r"discussion"),
    ("conclusion", r"conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks"),
)
"""(seção canônica, regex da palavra-chave), **em ordem de prioridade** para o orçamento."""

PRIORITY: dict[str, int] = {name: i for i, (name, _) in enumerate(SECTION_PATTERNS)}

_NUMBERING = r"(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVX]{1,5}\.?|[A-H]\.)"
_ALTERNATION = "|".join(f"(?P<{name}>{pat})" for name, pat in SECTION_PATTERNS)
_ALTERNATION_BARE = "|".join(f"(?P<{name}_b>{pat})" for name, pat in SECTION_PATTERNS)
HEADING_RE = re.compile(
    rf"^(?:{_NUMBERING}[ \t]+(?:{_ALTERNATION})\b"
    rf"|(?:{_ALTERNATION_BARE})[ \t]*(?=[.:\u2014\u2013]|$))",
    re.IGNORECASE | re.MULTILINE,
)
REFERENCES_RE = re.compile(
    rf"^(?:{_NUMBERING}[ \t]+)?(?:references|bibliography)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
MIN_WINDOW = 200
"""Janela mínima que vale a pena enviar quando o orçamento está acabando."""


@dataclass(frozen=True, slots=True)
class Heading:
    """Um cabeçalho de seção encontrado no texto."""

    section: str
    """Seção canônica (``methods``, ``limitations``...)."""
    label: str
    """Texto do cabeçalho como aparece no artigo (``3 Methods``)."""
    page: int
    start: int


def _canonical(m: re.Match[str]) -> str:
    for name, _ in SECTION_PATTERNS:
        if m.group(name) or m.group(f"{name}_b"):
            return name
    raise AssertionError("casamento sem grupo de seção")  # pragma: no cover


def find_headings(doc: Document) -> list[Heading]:
    """Todos os cabeçalhos de interesse, na ordem do artigo, até ``References``."""
    headings: list[Heading] = []
    for page in doc.pages:
        refs = REFERENCES_RE.search(page.text)
        limit = refs.start() if refs else len(page.text)
        for m in HEADING_RE.finditer(page.text, 0, limit):
            keyword = next(g for g in m.groups() if g)
            if not keyword[0].isupper():
                continue
            headings.append(Heading(_canonical(m), m.group(0).strip(), page.number, m.start()))
        if refs:
            break
    return headings


class KeywordSectionSelector:
    """Janelas de texto após cabeçalhos de seção, dentro de um orçamento de caracteres."""

    def __init__(self, window_chars: int = 1500, max_chars: int = 6000) -> None:
        if window_chars < MIN_WINDOW:
            raise ValueError(f"window_chars deve ser >= {MIN_WINDOW}, recebido {window_chars}")
        if max_chars < window_chars:
            raise ValueError("max_chars deve ser >= window_chars")
        self.window_chars = window_chars
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "keyword"

    # ------------------------------------------------------------------ janelas
    def _window(
        self, doc: Document, head: Heading, budget: int, all_heads: list[Heading]
    ) -> list[tuple[int, int, int, str]]:
        """Spans ``(página, início, fim, rótulo)`` da janela que começa em ``head``."""
        spans: list[tuple[int, int, int, str]] = []
        remaining = min(self.window_chars, budget)
        page_no, start, label = head.page, head.start, head.label
        while remaining >= MIN_WINDOW // 2:
            text = doc.page(page_no).text
            nxt = [
                h.start
                for h in all_heads
                if h.page == page_no and h.start >= start and h is not head
            ]
            stop = min([len(text), *nxt])
            hard = min(start + remaining, stop)
            end = hard if hard == stop else find_cut(text, start + (hard - start) // 2, hard)
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                spans.append((page_no, start, end, label))
                remaining -= end - start
            if hard < stop or nxt:
                break  # janela esgotada, ou a seção terminou nesta página
            # a seção continua na próxima página
            page_no, label = page_no + 1, f"{head.label} (cont.)"
            if page_no > doc.n_pages:
                break
            next_text = doc.page(page_no).text
            start = len(next_text) - len(next_text.lstrip())
        return spans

    @staticmethod
    def _merge(spans: list[tuple[int, int, int, str]]) -> list[tuple[int, int, int, str]]:
        """Une spans sobrepostos/encostados da mesma página, juntando os rótulos."""
        merged: list[tuple[int, int, int, str]] = []
        for page, start, end, label in sorted(spans):
            if merged and merged[-1][0] == page and start <= merged[-1][2]:
                p, s, e, lab = merged[-1]
                merged[-1] = (p, s, max(e, end), lab if label in lab else f"{lab} + {label}")
            else:
                merged.append((page, start, end, label))
        return merged

    # ------------------------------------------------------------------ seleção
    def select(self, doc: Document) -> Context:
        """Seleciona as seções de ``doc`` por palavra-chave."""
        headings = find_headings(doc)
        params: dict[str, object] = {
            "window_chars": self.window_chars,
            "max_chars": self.max_chars,
            "sections_found": [h.label for h in headings],
        }
        if not headings:
            fb = FirstPagesSelector(n_pages=2, max_chars=self.max_chars).select(doc)
            params.update(fallback="first_pages(2)", sections_selected=[])
            return Context(doc.arquivo, self.name, fb.chunks, params)

        # uma ocorrência (a primeira) por seção canônica, em ordem de prioridade
        first: dict[str, Heading] = {}
        for h in headings:
            first.setdefault(h.section, h)
        chosen = sorted(first.values(), key=lambda h: PRIORITY[h.section])
        if "abstract" not in first:
            start_page = next((p for p in doc.pages if p.text.strip()), doc.pages[0])
            chosen.insert(0, Heading("inicio", "início do artigo", start_page.number, 0))

        spans: list[tuple[int, int, int, str]] = []
        budget = self.max_chars
        selected: list[str] = []
        for head in chosen:
            if budget < MIN_WINDOW:
                break
            window = self._window(doc, head, budget, headings)
            if window:
                selected.append(head.label)
                spans.extend(window)
                budget -= sum(e - s for _, s, e, _ in window)

        chunks = tuple(
            Chunk(
                arquivo=doc.arquivo,
                page=page,
                text=doc.page(page).text[start:end],
                start=start,
                end=end,
                label=label,
            )
            for page, start, end, label in self._merge(spans)
        )
        params.update(fallback=None, sections_selected=selected)
        return Context(doc.arquivo, self.name, chunks, params)
