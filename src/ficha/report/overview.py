"""Quadros descritivos do corpus e das estratégias de seleção (Seções 2 e 4.1).

São tabelas de exibição: páginas, caracteres e tokens por artigo, e quanto cada estratégia de
seleção envia ao modelo. Os tokens usam o contador passado (o tokenizador real no Colab, ou a
aproximação declarada de ~4 caracteres por token no modo de ensaio).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from ficha.types import ContextSelector, Document

TokenCounter = Callable[[str], int]


def corpus_frame(docs: Sequence[Document], count_tokens: TokenCounter) -> pd.DataFrame:
    """Uma linha por artigo (páginas, caracteres, tokens) e uma linha final ``TOTAL``."""
    rows: list[dict[str, object]] = [
        {
            "arquivo": d.arquivo,
            "paginas": d.n_pages,
            "caracteres": d.n_chars,
            "tokens": count_tokens(d.full_text),
        }
        for d in docs
    ]
    frame = pd.DataFrame(rows, columns=["arquivo", "paginas", "caracteres", "tokens"])
    total = {
        "arquivo": "TOTAL",
        "paginas": int(frame["paginas"].sum()),
        "caracteres": int(frame["caracteres"].sum()),
        "tokens": int(frame["tokens"].sum()),
    }
    return pd.concat([frame, pd.DataFrame([total])], ignore_index=True)


def selection_frame(
    docs: Sequence[Document],
    selectors: Sequence[ContextSelector],
    count_tokens: TokenCounter,
) -> pd.DataFrame:
    """Formato longo: uma linha por (estratégia, artigo) com o que é efetivamente enviado."""
    rows: list[dict[str, object]] = []
    for sel in selectors:
        for d in docs:
            ctx = sel.select(d)
            text = ctx.render()
            rows.append(
                {
                    "estrategia": sel.name,
                    "arquivo": d.arquivo,
                    "paginas_enviadas": ", ".join(str(p) for p in ctx.pages),
                    "n_chunks": len(ctx.chunks),
                    "caracteres": len(text),
                    "tokens": count_tokens(text),
                    "fracao_do_artigo": len(text) / d.n_chars if d.n_chars else 0.0,
                }
            )
    return pd.DataFrame(rows)


def strategy_summary(selection: pd.DataFrame) -> pd.DataFrame:
    """Resumo por estratégia: médias por artigo e total de tokens no corpus."""
    grouped = selection.groupby("estrategia", sort=False)
    return pd.DataFrame(
        {
            "chunks_medios": grouped["n_chunks"].mean().round(1),
            "caracteres_medios": grouped["caracteres"].mean().round(0).astype("int64"),
            "tokens_medios": grouped["tokens"].mean().round(0).astype("int64"),
            "tokens_corpus": grouped["tokens"].sum().astype("int64"),
            "fracao_media_do_artigo": grouped["fracao_do_artigo"].mean().round(3),
        }
    )
