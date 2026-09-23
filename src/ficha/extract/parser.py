"""Parse e validação da saída do modelo — sem biblioteca de reparo externa (Seção 7).

A saída bruta nunca é alterada (ela vai intacta para ``ExtractionRecord.raw_output``); o
parser trabalha numa cópia e registra **cada** reparo aplicado, para que a comparação de prompts
possa contar quantas saídas precisaram de ajuda.

Etapas:

1. **Estrito**: ``json.loads`` do texto bruto + ``FichaExtraida.model_validate``.
2. **Reparos textuais**, aplicados um a um até o JSON carregar: remover cercas de código,
   extrair o primeiro objeto ``{...}`` balanceado, remover vírgulas finais, trocar aspas
   tipográficas por retas.
3. **Reparos estruturais** sobre o objeto: desembrulhar ``{"ficha": ...}`` /
   ``{"raciocinio": ..., "ficha": ...}``, ``pagina`` em texto → inteiro.
4. **Validação** pelo schema. Campo a mais é inválido (``extra="forbid"``) e **não** é reparado.
   Depois de validar, marca-se ``limitacao_sentinela`` (o modelo escreveu ``"N/A"``, ``"null"``...
   e o schema converteu para ``None``) e ``limitacao_ausente`` (a chave nem veio).

Definições usadas na comparação de prompts (Seção 4.2):

- ``status``: ``OK`` sem nenhum reparo; ``REPAIRED`` com pelo menos um; ``FAILED`` sem ficha.
- ``json_valid_first_try``: o texto bruto passa em ``json.loads`` estrito **e** satisfaz o schema
  sem nenhum reparo textual ou estrutural. Os dois reparos semânticos de ``limitacao`` não
  contam contra (o JSON já era válido e aceito pelo schema), mas deixam o status ``REPAIRED``.
- Com ``chain_of_thought=True`` o envelope ``{"raciocinio", "ficha"}`` é o formato pedido, então
  desembrulhá-lo não é reparo; já uma ficha sem envelope é (``cot_sem_envelope``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ficha.schema import FichaExtraida
from ficha.types import ParseStatus

# Nomes dos reparos (aparecem em ParseResult.repairs e no relatório).
REPAIR_FENCES = "cercas_de_codigo"
REPAIR_EXTRACT_OBJECT = "objeto_extraido_do_texto"
REPAIR_TRAILING_COMMA = "virgula_final"
REPAIR_SMART_QUOTES = "aspas_tipograficas"
REPAIR_UNWRAP = "envelope_ficha"
REPAIR_COT_NO_ENVELOPE = "cot_sem_envelope"
REPAIR_PAGINA_INT = "pagina_para_int"
REPAIR_LIMITACAO_SENTINEL = "limitacao_sentinela"
REPAIR_LIMITACAO_MISSING = "limitacao_ausente"

SEMANTIC_REPAIRS = frozenset({REPAIR_LIMITACAO_SENTINEL, REPAIR_LIMITACAO_MISSING})
"""Reparos que não tornam o JSON "inválido de primeira" (ver docstring do módulo)."""


@dataclass(slots=True)
class ParseResult:
    """Resultado do parse de uma saída do modelo."""

    status: ParseStatus
    ficha: FichaExtraida | None
    json_valid_first_try: bool
    repairs: list[str] = field(default_factory=list)
    error: str | None = None
    raciocinio: str | None = None
    """Conteúdo de ``raciocinio`` quando o modelo usou o envelope de cadeia de raciocínio."""

    @property
    def ok(self) -> bool:
        return self.ficha is not None


# --------------------------------------------------------------------------- reparos textuais

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"^\s*```[ \t]*(?:json|JSON)?[ \t]*\n?")


def strip_code_fences(text: str) -> str:
    """Conteúdo da primeira cerca de código markdown (ou remove a cerca aberta e não fechada)."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return _OPEN_FENCE_RE.sub("", text, count=1)


def extract_first_object(text: str) -> str | None:
    """Primeiro objeto ``{...}`` balanceado (ignorando chaves dentro de strings), ou ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None  # desbalanceado (ex.: saída truncada)


def remove_trailing_commas(text: str) -> str:
    """Remove vírgulas imediatamente antes de ``}`` ou ``]`` (fora de strings)."""
    out: list[str] = []
    in_string = False
    escaped = False
    n = len(text)
    for i, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                continue
        out.append(ch)
    return "".join(out)


def straighten_quotes(text: str) -> str:
    """Aspas duplas tipográficas (“ ” „ ″) → aspas retas. Apóstrofos ficam como estão."""
    return re.sub("[“”„″]", '"', text)


_TEXT_REPAIRS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    (REPAIR_FENCES, strip_code_fences),
    (REPAIR_EXTRACT_OBJECT, extract_first_object),
    (REPAIR_TRAILING_COMMA, remove_trailing_commas),
    (REPAIR_SMART_QUOTES, straighten_quotes),
    # Depois de trocar as aspas, a extração e a vírgula final podem passar a funcionar.
    (REPAIR_EXTRACT_OBJECT, extract_first_object),
    (REPAIR_TRAILING_COMMA, remove_trailing_commas),
)


def _describe_json_error(exc: json.JSONDecodeError) -> str:
    return f"JSON inválido: {exc.msg} (linha {exc.lineno}, coluna {exc.colno})"


def _load_json(text: str, repairs: list[str]) -> tuple[Any, bool, str | None]:
    """Carrega o JSON: ``(objeto, estrito_ok, erro)``. Registra reparos textuais aplicados."""
    try:
        return json.loads(text), True, None
    except json.JSONDecodeError as exc:
        last_error = _describe_json_error(exc)

    current = text
    for name, repair in _TEXT_REPAIRS:
        candidate = repair(current)
        if candidate is None or candidate == current:
            continue
        current = candidate
        if name not in repairs:
            repairs.append(name)
        try:
            return json.loads(current), False, None
        except json.JSONDecodeError as exc:
            last_error = _describe_json_error(exc)
    if "{" not in text:
        last_error = "Nenhum objeto JSON na saída"
    return None, False, last_error


# --------------------------------------------------------------------------- reparos estruturais

_PAGINA_RE = re.compile(
    r"\s*\[?\s*(?:p|pp|pg|pag|pág|page|pagina|página)?\.?\s*(\d+)\s*\]?\s*", re.IGNORECASE
)


def _unwrap(
    obj: dict[str, Any], chain_of_thought: bool, repairs: list[str]
) -> tuple[dict[str, Any], str | None]:
    """Desembrulha o envelope ``{"raciocinio"?, "ficha"}``, se for esse o formato."""
    if isinstance(obj.get("ficha"), dict) and set(obj) <= {"ficha", "raciocinio"}:
        if not chain_of_thought:
            repairs.append(REPAIR_UNWRAP)
        raciocinio = obj.get("raciocinio")
        return obj["ficha"], raciocinio if isinstance(raciocinio, str) else None
    if chain_of_thought:
        repairs.append(REPAIR_COT_NO_ENVELOPE)
    return obj, None


def _fix_pagina(obj: dict[str, Any], repairs: list[str]) -> None:
    """``evidencia.pagina`` em texto (``"3"``, ``"p. 3"``) ou float inteiro (``3.0``) → ``int``."""
    ev = obj.get("evidencia")
    if not isinstance(ev, dict):
        return
    pagina = ev.get("pagina")
    fixed: int | None = None
    if isinstance(pagina, str):
        m = _PAGINA_RE.fullmatch(pagina)
        if m:
            fixed = int(m.group(1))
    elif isinstance(pagina, float) and pagina.is_integer():
        fixed = int(pagina)
    if fixed is not None:
        ev["pagina"] = fixed
        repairs.append(REPAIR_PAGINA_INT)


def _summarize_validation_error(exc: ValidationError, max_items: int = 5) -> str:
    items = []
    for err in exc.errors()[:max_items]:
        loc = ".".join(str(p) for p in err["loc"]) or "(raiz)"
        items.append(f"{loc}: {err['msg']}")
    more = exc.error_count() - max_items
    suffix = f"; (+{more} erros)" if more > 0 else ""
    return "Schema inválido: " + "; ".join(items) + suffix


# --------------------------------------------------------------------------- API


def parse_completion(text: str, *, chain_of_thought: bool = False) -> ParseResult:
    """Transforma a saída bruta do modelo em :class:`ParseResult`. Nunca levanta exceção.

    ``chain_of_thought=True`` indica que o prompt pediu o envelope ``{"raciocinio", "ficha"}``.
    """
    repairs: list[str] = []
    obj, strict_ok, error = _load_json(text, repairs)
    if error is not None:
        return ParseResult(ParseStatus.FAILED, None, False, repairs, error)
    if not isinstance(obj, dict):
        return ParseResult(
            ParseStatus.FAILED,
            None,
            False,
            repairs,
            f"JSON não é um objeto (veio {type(obj).__name__})",
        )

    obj, raciocinio = _unwrap(obj, chain_of_thought, repairs)
    _fix_pagina(obj, repairs)
    limitacao_raw = obj.get("limitacao")
    limitacao_missing = "limitacao" not in obj

    try:
        ficha = FichaExtraida.model_validate(obj)
    except ValidationError as exc:
        return ParseResult(
            ParseStatus.FAILED,
            None,
            False,
            repairs,
            _summarize_validation_error(exc),
            raciocinio,
        )

    if limitacao_missing:
        repairs.append(REPAIR_LIMITACAO_MISSING)
    elif isinstance(limitacao_raw, str) and ficha.limitacao is None:
        repairs.append(REPAIR_LIMITACAO_SENTINEL)

    first_try = strict_ok and all(r in SEMANTIC_REPAIRS for r in repairs)
    status = ParseStatus.REPAIRED if repairs else ParseStatus.OK
    return ParseResult(status, ficha, first_try, repairs, None, raciocinio)
