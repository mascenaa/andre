"""Diagnóstico offline: mede se as frases de limitação declaradas chegam ao contexto.

Motivação (execução real, 19 artigos): ``limitacao`` veio ``null`` em quase todas as fichas.
Antes de culpar o modelo, medimos se a **entrada** continha a informação: localizamos no texto
completo as frases em que os autores declaram limitações (regex :data:`LIMITATION_SENTENCE_RE`)
e contamos quantas estão dentro do contexto que cada estratégia envia. O modelo não pode
declarar o que não recebe — isso é um teto para qualquer prompt.

Tudo aqui é determinístico e roda sem modelo. O regex é um detector de **alta precisão**, não
de cobertura total: frases de limitação sem essas marcas não são contadas. Por isso ele serve
para comparar estratégias entre si, não como gabarito absoluto.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from ficha.types import Context, ContextSelector, Document

LIMITATION_SENTENCE_RE = re.compile(
    r"limitations? of (?:this|our|the present) (?:study|work|approach|method|analysis)"
    r"|(?:is|are|remains?) limited (?:by|to)"
    r"|we (?:acknowledge|note|caution|caveat)"
    r"|(?:a|one|another|the main) (?:limitation|caveat|drawback|shortcoming)"
    r"|beyond the scope of (?:this|the present)"
    r"|not (?:able|possible) to"
    r"|cannot (?:be|capture|account)"
    r"|(?:has|have) (?:several |some |a few |important |certain )?limitations",
    re.IGNORECASE,
)
"""Marcas lexicais de limitação declarada pelos autores (alta precisão)."""

KEY_CHARS = 80
"""Uma frase conta como "no contexto" se os seus primeiros 80 caracteres (normalizados)
aparecem no texto do contexto."""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n\s*\n")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Minúsculas e espaços colapsados (comparação tolerante a quebras de linha)."""
    return _WS_RE.sub(" ", text).strip().lower()


def split_sentences(text: str) -> list[str]:
    """Divisão simples em sentenças (ponto/interrogação/exclamação seguido de maiúscula)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def limitation_sentences(doc: Document) -> list[tuple[int, str]]:
    """Frases de limitação declarada no documento inteiro, como ``(página, frase)``."""
    out: list[tuple[int, str]] = []
    for page in doc.pages:
        for sentence in split_sentences(page.text):
            if LIMITATION_SENTENCE_RE.search(sentence):
                out.append((page.number, sentence))
    return out


@dataclass(frozen=True, slots=True)
class LimitationRecall:
    """Quantas frases de limitação do artigo chegaram ao contexto."""

    arquivo: str
    n_sentences_doc: int
    n_sentences_in_context: int
    pages_doc: tuple[int, ...]
    """Páginas do artigo que contêm frases de limitação."""
    recall: float | None
    """``n_sentences_in_context / n_sentences_doc``; ``None`` se o artigo não tem nenhuma."""


def limitation_sentence_recall(doc: Document, context: Context) -> LimitationRecall:
    """Recall das frases de limitação de ``doc`` dentro de ``context``."""
    sentences = limitation_sentences(doc)
    ctx_text = normalize(" ".join(c.text for c in context.chunks))
    hits = sum(1 for _, s in sentences if normalize(s)[:KEY_CHARS] in ctx_text)
    n = len(sentences)
    return LimitationRecall(
        arquivo=doc.arquivo,
        n_sentences_doc=n,
        n_sentences_in_context=hits,
        pages_doc=tuple(sorted({p for p, _ in sentences})),
        recall=hits / n if n else None,
    )


RECALL_COLUMNS: tuple[str, ...] = (
    "estrategia",
    "artigos_com_frases",
    "artigos_cobertos",
    "frases_total",
    "frases_no_contexto",
    "recall_frases",
    "chars_medios",
)


def recall_table(
    docs: Iterable[Document], selectors: Mapping[str, ContextSelector]
) -> pd.DataFrame:
    """Uma linha por estratégia com o recall agregado das frases de limitação.

    - ``artigos_com_frases``: artigos com ao menos uma frase de limitação no texto completo;
    - ``artigos_cobertos``: desses, quantos tiveram ao menos uma frase no contexto;
    - ``recall_frases``: ``frases_no_contexto / frases_total`` (micro, sobre todas as frases);
    - ``chars_medios``: tamanho médio do contexto (caracteres), para pôr o recall ao lado do
      custo (Seção 4.5).
    """
    docs = list(docs)
    rows: list[dict[str, object]] = []
    for name, selector in selectors.items():
        with_sentences = covered = total = in_ctx = 0
        chars: list[int] = []
        for doc in docs:
            ctx = selector.select(doc)
            chars.append(ctx.n_chars)
            r = limitation_sentence_recall(doc, ctx)
            total += r.n_sentences_doc
            in_ctx += r.n_sentences_in_context
            if r.n_sentences_doc:
                with_sentences += 1
                covered += int(r.n_sentences_in_context > 0)
        rows.append(
            {
                "estrategia": name,
                "artigos_com_frases": with_sentences,
                "artigos_cobertos": covered,
                "frases_total": total,
                "frases_no_contexto": in_ctx,
                "recall_frases": round(in_ctx / total, 3) if total else None,
                "chars_medios": round(sum(chars) / len(chars)) if chars else 0,
            }
        )
    return pd.DataFrame(rows, columns=list(RECALL_COLUMNS))
