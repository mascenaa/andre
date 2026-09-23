"""Normalização de texto usada em TODA comparação da auditoria.

Esta é a premissa da verificação de fidelidade (4.4a) e do motor de diferenças (4.4b-d):
duas strings são consideradas "o mesmo texto" quando ficam iguais depois de
:func:`normalize_for_match`. A normalização é determinística, sem dependências externas, e
remove apenas diferenças **tipográficas** que não mudam o conteúdo:

1. Unicode NFKC — desfaz ligaduras de PDF (``ﬁ`` → ``fi``), reticências (``…`` → ``...``),
   espaços especiais (NBSP → espaço) e formas de largura total.
2. Remoção de caracteres invisíveis (hífen suave, espaços de largura zero, BOM).
3. Aspas tipográficas → aspas retas (``“ ” ‘ ’ « »`` → ``" '``).
4. Hífens, travessões e sinal de menos unificados em ``-``; espaços ao redor de ``-`` removidos
   (``state - of - the - art`` ≡ ``state-of-the-art``, comum em extração de PDF).
5. ``casefold`` (minúsculas agressivas).
6. Qualquer sequência de espaço em branco (inclusive quebras de linha) → um espaço.
7. Espaço antes de pontuação de fechamento e depois de abertura removido
   (``resultado .`` ≡ ``resultado.``; ``( a )`` ≡ ``(a)``).

O que **não** é normalizado, de propósito: palavras, números e ordem. Um trecho parafraseado
continua diferente — é isso que a verificação de fidelidade quer pegar.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
    }
)
_DASHES = str.maketrans(dict.fromkeys("‐‑‒–—―−﹘", "-"))
_INVISIBLE = str.maketrans(dict.fromkeys("­​‌‍⁠﻿", ""))

_WS_RE = re.compile(r"\s+")
_SPACE_AROUND_HYPHEN_RE = re.compile(r" ?- ?")
_SPACE_BEFORE_CLOSE_RE = re.compile(r" ([.,;:!?)\]}])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{]) ")

PAGE_MARK_RE = re.compile(r"^\[p\. (\d+)\][ \t]*$", re.MULTILINE)
"""Marcador ``[p. N]`` em linha própria, produzido por :meth:`ficha.types.Context.render`."""


def normalize_for_match(text: str) -> str:
    """Normaliza ``text`` para comparação (ver docstring do módulo para as regras, em ordem).

    Idempotente: ``normalize_for_match(normalize_for_match(x)) == normalize_for_match(x)``.
    """
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_INVISIBLE).translate(_QUOTES).translate(_DASHES)
    t = t.casefold()
    t = _WS_RE.sub(" ", t).strip()
    t = _SPACE_AROUND_HYPHEN_RE.sub("-", t)
    t = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN_RE.sub(r"\1", t)
    return t


def split_context_blocks(context_text: str) -> list[tuple[int, str]]:
    """Divide o texto enviado em blocos ``(página, texto)``, na ordem em que aparecem.

    Uma mesma página pode aparecer em mais de um bloco (estratégias que selecionam vários
    chunks da mesma página). Texto antes do primeiro marcador é ignorado se estiver em branco;
    se não estiver (contexto sem marcadores), é devolvido com página ``0`` = "desconhecida".
    """
    parts = PAGE_MARK_RE.split(context_text)
    blocks: list[tuple[int, str]] = []
    if parts[0].strip():
        blocks.append((0, parts[0].strip()))
    # parts = [antes, num, texto, num, texto, ...]
    for i in range(1, len(parts) - 1, 2):
        blocks.append((int(parts[i]), parts[i + 1].strip()))
    return blocks


def split_context_pages(context_text: str) -> dict[int, str]:
    """Texto enviado agrupado por página: ``{página: texto}``.

    Blocos da mesma página são concatenados (separados por linha em branco) na ordem em que
    aparecem. Texto sem marcador (página ``0``) é excluído — não há página a atribuir.
    """
    out: dict[int, list[str]] = {}
    for page, text in split_context_blocks(context_text):
        if page == 0:
            continue
        out.setdefault(page, []).append(text)
    return {p: "\n\n".join(texts) for p, texts in out.items()}
