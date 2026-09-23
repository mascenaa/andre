r"""Limpeza do texto extraído do PDF (Seção 4.1, último parágrafo do enunciado).

Princípio: **determinística e conservadora**. A auditoria de fidelidade (4.4a) compara o
``evidencia.trecho`` devolvido pelo modelo com o texto enviado, que sai daqui. Cada regra
só mexe no que é claramente artefato de diagramação (quebra de linha, hífen de fim de linha,
cabeçalho/rodapé, número de página, ligatura tipográfica, espaço irregular) e nunca reescreve
palavras do autor.

Cada regra é uma função pequena e testável. :func:`clean_text` apenas as encadeia, numa
ordem que importa:

1. :func:`normalize_unicode` — NFKC + ligaturas + caracteres invisíveis.
2. :func:`normalize_spaces` — espaços exóticos viram espaço simples; ``\r`` vira ``\n``.
3. :func:`strip_page_numbers` — número de página solto na primeira/última linha.
4. :func:`dehyphenate` — ``exam-\nple`` → ``example`` (ver heurística na função).
5. :func:`rebuild_paragraphs` — quebra simples vira espaço; quebra de parágrafo fica.
6. :func:`collapse_spaces` — espaços repetidos e linhas em branco em excesso.

A remoção de cabeçalho/rodapé repetido depende de **várias páginas** ao mesmo tempo e por isso
fica em :func:`strip_repeated_lines`, chamada por :func:`ficha.ingest.pdf.load_pdf` antes de
``clean_text``.

Propriedade garantida (e testada): ``clean_text(clean_text(x)) == clean_text(x)``.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Collection, Iterable
from itertools import pairwise

# --------------------------------------------------------------------------- unicode

LIGATURES: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
"""Ligaturas tipográficas comuns em PDFs de LaTeX. NFKC já as decompõe; o mapa explícito
documenta a intenção e protege contra uma eventual mudança de normalização."""

_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_HYPHEN_EOL_RE = re.compile(r"\u00ad[ \t]*\n[ \t]*")


def normalize_unicode(text: str) -> str:
    """Aplica NFKC, desfaz ligaturas e remove caracteres invisíveis e de controle.

    NFKC também troca formas de compatibilidade (``²`` → ``2``, ``…`` → ``...``,
    espaço não separável → espaço). É aceitável porque o modelo recebe **este** texto e a
    fidelidade é medida contra ele. O hífen "suave" (``U+00AD``) só existe para indicar
    hifenização possível: no fim de linha ele junta as partes, no meio da linha é removido.
    """
    for lig, rep in LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = _SOFT_HYPHEN_EOL_RE.sub("", text)
    text = text.replace("\u00ad", "")
    text = _INVISIBLE_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


# --------------------------------------------------------------------------- espaços

_EXOTIC_SPACES_RE = re.compile(r"[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_LEADING_WS_RE = re.compile(r"^[ \t]+", re.MULTILINE)


def normalize_spaces(text: str) -> str:
    r"""Padroniza quebras de linha (``\r\n``/``\r`` → ``\n``) e espaços exóticos → espaço.

    Também remove espaço no início e no fim de cada linha, que no PDF é só posicionamento.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = _EXOTIC_SPACES_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    return _LEADING_WS_RE.sub("", text)


def collapse_spaces(text: str) -> str:
    """Espaços repetidos viram um; três ou mais quebras viram uma quebra de parágrafo."""
    text = re.sub(r" {2,}", " ", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _LEADING_WS_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- número de página

_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+|p\.\s*|-\s*)?\d{1,4}(?:\s*(?:/|of|de)\s*\d{1,4})?(?:\s*-)?$",
    re.IGNORECASE,
)


def is_page_number(line: str) -> bool:
    """``True`` se a linha é só um número de página (``12``, ``Page 3``, ``- 4 -``, ``3 of 10``)."""
    return _PAGE_NUMBER_RE.match(line.strip()) is not None


def strip_page_numbers(text: str) -> str:
    """Remove números de página soltos na **primeira ou última** linha não vazia do texto.

    Conservador de propósito: uma linha só com número no meio da página costuma ser célula de
    tabela, e aí é dado — não mexemos.
    """
    lines = text.split("\n")
    while True:
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        if not non_empty:
            break
        first, last = non_empty[0], non_empty[-1]
        if is_page_number(lines[first]):
            del lines[first]
        elif is_page_number(lines[last]):
            del lines[last]
        else:
            break
    return "\n".join(lines)


# --------------------------------------------------------------------------- hifenização

COMPOUND_PREFIXES: frozenset[str] = frozenset(
    {
        "self",
        "well",
        "state",
        "cross",
        "long",
        "short",
        "high",
        "low",
        "real",
        "large",
        "small",
        "end",
        "fine",
        "open",
        "one",
        "two",
        "three",
        "few",
        "zero",
        "rule",
        "task",
        "user",
        "time",
        "world",
    }
)
"""Palavras inteiras que em inglês costumam abrir compostos com hífen (``self-attention``,
``cross-validation``, ``few-shot``). Só são consultadas quando o próprio documento não dá
evidência sobre a palavra. Prefixos morfológicos (``re``, ``pre``, ``co``, ``multi``) ficam de
fora de propósito: são também pontos de hifenização tipográfica (``re-\nsults``), e juntar é o
erro menos danoso nesses casos."""

HYPHENATION_SUFFIXES: frozenset[str] = frozenset(
    {
        "tion",
        "tions",
        "sion",
        "sions",
        "ment",
        "ments",
        "ness",
        "ity",
        "ities",
        "ing",
        "ings",
        "ed",
        "er",
        "ers",
        "ly",
        "able",
        "ible",
        "ance",
        "ence",
        "al",
        "ally",
        "ous",
        "ive",
        "ize",
        "ized",
        "ization",
        "ic",
        "ical",
        "ist",
        "est",
    }
)
"""Sufixos que, sozinhos depois de um hífen de fim de linha, indicam hifenização tipográfica
(``evalua-\ntion``, ``high-\nly``): nenhum composto legítimo termina num sufixo solto."""

_WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_HYPHEN_EOL_RE = re.compile(
    r"(?<!\S)(?P<prefix>\S*?)(?P<left>[^\W_]+)[-\u2010]\n[ \t]*"
    r"(?P<right>[^\W_][\w']*)(?P<rest>\S*)(?P<trail>[ \t]+)?"
)


def build_vocabulary(texts: Iterable[str]) -> frozenset[str]:
    """Vocabulário (minúsculo) usado como evidência na hifenização.

    Inclui palavras simples, compostos inteiros (``state-of-the-art``) e cada par adjacente de
    um composto (``of-the``, ``the-art``), para reconhecer um composto quebrado no meio.
    """
    vocab: set[str] = set()
    for text in texts:
        for m in _WORD_RE.finditer(text):
            word = m.group(0).lower()
            vocab.add(word)
            parts = word.split("-")
            if len(parts) > 1:
                vocab.update(parts)
                vocab.update(f"{a}-{b}" for a, b in pairwise(parts))
    return frozenset(vocab)


def should_keep_hyphen(prefix: str, left: str, right: str, vocabulary: Collection[str]) -> bool:
    r"""Decide se o hífen de fim de linha é **legítimo** (mantém) ou de hifenização (remove).

    Heurística, na ordem (a primeira que decide vence):

    1. A continuação começa com maiúscula ou dígito, ou a primeira parte termina em dígito
       (``Bayes-\nOptimal``, ``Sentinel-\n2``, ``2019-\n2020``) → mantém.
    2. A palavra antes do hífen já é um composto (``state-of-the-\nart``) → mantém.
    3. O documento contém a forma junta (``example``) → junta.
    4. O documento contém a forma hifenizada (``fine-tuning``) → mantém.
    5. A continuação é um sufixo solto (:data:`HYPHENATION_SUFFIXES`) → junta.
    6. A primeira parte é palavra que abre compostos (:data:`COMPOUND_PREFIXES`) → mantém.
    7. Caso contrário → junta (é o caso mais comum: hifenização tipográfica).
    """
    if not right[0].islower() or not left[-1].isalpha():
        return True
    if "-" in prefix:
        return True
    joined = f"{left}{right}".lower()
    hyphenated = f"{left}-{right}".lower()
    if joined in vocabulary:
        return False
    if hyphenated in vocabulary:
        return True
    if right.lower() in HYPHENATION_SUFFIXES:
        return False
    return left.lower() in COMPOUND_PREFIXES


def dehyphenate(text: str, vocabulary: Collection[str] | None = None) -> str:
    r"""Resolve hífen de fim de linha, juntando a palavra ou mantendo o hífen legítimo.

    ``exam-\nple`` → ``example``; ``state-of-the-\nart`` → ``state-of-the-art``.
    Ver :func:`should_keep_hyphen`. A palavra reconstituída fica na linha de cima.

    ``vocabulary`` permite passar o vocabulário do documento inteiro (mais evidência); se
    ``None``, usa o do próprio texto. Não atravessa quebra de parágrafo (linha em branco).
    """
    vocab = build_vocabulary([text]) if vocabulary is None else vocabulary

    def _fix(m: re.Match[str]) -> str:
        prefix, left, right = m.group("prefix"), m.group("left"), m.group("right")
        sep = "-" if should_keep_hyphen(prefix, left, right, vocab) else ""
        # A palavra inteira sobe para a linha de cima e a quebra continua DEPOIS dela: assim o
        # comprimento das linhas (usado por ``rebuild_paragraphs``) não é distorcido.
        tail = "\n" if m.group("trail") else ""
        return f"{prefix}{left}{sep}{right}{m.group('rest')}{tail}"

    return _HYPHEN_EOL_RE.sub(_fix, text)


# --------------------------------------------------------------------------- parágrafos

_STARTS_BLOCK_RE = re.compile(r"^(?:[A-Z0-9À-Ý]|[•\-–—*]\s)")
SHORT_LINE_RATIO = 0.6
"""Uma linha com menos de 60% do comprimento da maior linha do bloco é "curta"."""


def _ends_paragraph(line: str, next_line: str, max_len: int) -> bool:
    """Heurística: diz se a quebra entre ``line`` e ``next_line`` é fim de parágrafo.

    Dentro de um bloco do PDF, um parágrafo termina quando a linha é **curta** (não chegou à
    margem) e a próxima começa com maiúscula, dígito ou marcador de lista. Isso também separa
    cabeçalhos colados ao texto (``Abstract`` / ``We study...``). Linhas cheias que terminam
    com ponto são meio de parágrafo, e viram espaço.
    """
    if not _STARTS_BLOCK_RE.match(next_line):
        return False
    return len(line) < SHORT_LINE_RATIO * max_len


def rebuild_paragraphs(text: str) -> str:
    """Reconstrói parágrafos: quebra de linha simples vira espaço, quebra dupla é preservada.

    A saída tem um parágrafo por linha, separados por uma linha em branco. Como a saída não
    tem mais quebras simples, aplicar de novo não muda nada (idempotência).
    """
    blocks = re.split(r"\n[ \t]*\n", text)
    out_blocks: list[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        max_len = max(len(ln) for ln in lines)
        paragraphs: list[list[str]] = [[lines[0]]]
        for prev, line in pairwise(lines):
            if _ends_paragraph(prev, line, max_len):
                paragraphs.append([line])
            else:
                paragraphs[-1].append(line)
        out_blocks.extend(" ".join(p) for p in paragraphs)
    return "\n\n".join(out_blocks)


# --------------------------------------------------------------------------- pipeline


def clean_text(raw: str, vocabulary: Collection[str] | None = None) -> str:
    """Limpa o texto de UMA página. Determinística, conservadora e idempotente.

    ``vocabulary`` (opcional) é o vocabulário do documento inteiro, usado como evidência pela
    hifenização (ver :func:`build_vocabulary`).
    """
    text = normalize_unicode(raw)
    text = normalize_spaces(text)
    text = strip_page_numbers(text)
    text = dehyphenate(text, vocabulary)
    text = rebuild_paragraphs(text)
    text = collapse_spaces(text)
    return strip_page_numbers(text).strip()


# --------------------------------------------------------------------------- cabeçalho/rodapé

EDGE_LINES = 3
"""Quantas linhas do topo e da base de cada página são candidatas a cabeçalho/rodapé."""
MAX_HEADER_LEN = 120
"""Cabeçalho/rodapé é linha curta; linhas maiores nunca são removidas."""
REPEAT_RATIO = 0.5
"""Fração mínima de páginas em que a linha precisa se repetir."""


def _line_key(line: str) -> str:
    """Chave de comparação: minúscula, espaços colapsados, dígitos → ``#``.

    Trocar dígitos permite reconhecer ``Journal X, p. 3`` e ``Journal X, p. 4`` como o mesmo
    rodapé, e também o número de página sozinho.
    """
    key = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line)).strip().lower()
    return re.sub(r"\d+", "#", key)


def _edge_indices(lines: list[str]) -> set[int]:
    non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
    return set(non_empty[:EDGE_LINES]) | set(non_empty[-EDGE_LINES:])


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove cabeçalhos/rodapés: linhas curtas que se repetem em ≥ 50% das páginas.

    Só olha as :data:`EDGE_LINES` primeiras e últimas linhas não vazias de cada página
    (é onde cabeçalho e rodapé moram) e exige que a repetição apareça em pelo menos duas
    páginas. Linhas no meio da página nunca são tocadas — uma frase repetida no corpo do
    texto é conteúdo, não diagramação. Recebe e devolve o texto **bruto** de cada página.
    """
    n = len(pages)
    if n < 2:
        return list(pages)
    split = [p.split("\n") for p in pages]
    counts: Counter[str] = Counter()
    for lines in split:
        keys = {
            _line_key(lines[i]) for i in _edge_indices(lines) if len(lines[i]) <= MAX_HEADER_LEN
        }
        counts.update(k for k in keys if k)
    threshold = max(2, math.ceil(REPEAT_RATIO * n))
    repeated = {k for k, c in counts.items() if c >= threshold}
    if not repeated:
        return list(pages)
    out: list[str] = []
    for lines in split:
        edges = _edge_indices(lines)
        kept = [
            ln
            for i, ln in enumerate(lines)
            if not (i in edges and len(ln) <= MAX_HEADER_LEN and _line_key(ln) in repeated)
        ]
        out.append("\n".join(kept))
    return out
