"""Motor genérico de comparação entre duas execuções (reutilizado em 4.4b, 4.4c, 4.4d e 4.2).

Como a divergência é quantificada (declarado no ADR 0005):

- Os registros são pareados por ``arquivo``. Se um dos lados não tem ficha (parse ``FAILED``)
  ou não tem o arquivo, o par não é comparável e conta em ``n_unpaired`` — nunca como
  "igual" nem como "diferente".
- Cada ficha é comparada em :data:`FIELDS` (os campos da Seção 3, com ``evidencia`` aberta em
  ``trecho`` e ``pagina``; ``confianca`` não entra porque não vem do modelo).
- **Igualdade** de strings: iguais após :func:`normalize_for_match`.
  **Similaridade** de strings: ``rapidfuzz.fuzz.ratio/100`` sobre as formas normalizadas.
  ``pagina``: igual/similaridade 1 se o número é o mesmo, senão 0.
  ``None`` (só ``limitacao`` pode ser ``None``): ambos ``None`` → iguais (similaridade 1);
  apenas um ``None`` → diferentes (similaridade 0), porque "declara limitação" versus
  "não declara" é uma divergência de conteúdo, não de redação.
- Métricas: taxa de fichas **idênticas** (todos os campos iguais), taxa de mudança **por
  campo** e similaridade **média** por campo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from ficha.audit.normalize import normalize_for_match
from ficha.schema import CAMPOS_FICHA, FichaExtraida
from ficha.types import ExtractionRecord

FIELDS: tuple[str, ...] = (
    *(c for c in CAMPOS_FICHA if c != "evidencia"),
    "evidencia.trecho",
    "evidencia.pagina",
)
"""Campos comparados, na ordem da Seção 3."""

FieldValue = str | int | None


def field_value(ficha: FichaExtraida, name: str) -> FieldValue:
    """Valor de um campo de :data:`FIELDS` numa ficha (``evidencia.x`` acessa o aninhado)."""
    if name == "evidencia.trecho":
        return ficha.evidencia.trecho
    if name == "evidencia.pagina":
        return ficha.evidencia.pagina
    value: FieldValue = getattr(ficha, name)
    return value


def compare_values(a: FieldValue, b: FieldValue) -> tuple[bool, float]:
    """``(igual, similaridade)`` segundo as regras do módulo."""
    if a is None and b is None:
        return True, 1.0
    if a is None or b is None:
        return False, 0.0
    if isinstance(a, int) or isinstance(b, int):
        eq = a == b
        return eq, 1.0 if eq else 0.0
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if na == nb:
        return True, 1.0
    return False, round(fuzz.ratio(na, nb) / 100.0, 4)


@dataclass(frozen=True, slots=True)
class FieldDiff:
    """Comparação de UM campo de UM artigo entre as execuções A e B."""

    arquivo: str
    field: str
    a: FieldValue
    b: FieldValue
    equal: bool
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diff_fichas(arquivo: str, a: FichaExtraida, b: FichaExtraida) -> list[FieldDiff]:
    """Compara duas fichas em todos os :data:`FIELDS`."""
    out: list[FieldDiff] = []
    for name in FIELDS:
        va, vb = field_value(a, name), field_value(b, name)
        eq, sim = compare_values(va, vb)
        out.append(FieldDiff(arquivo, name, va, vb, eq, sim))
    return out


@dataclass(frozen=True, slots=True)
class DiffReport:
    """Divergência entre duas execuções, quantificada por ficha e por campo."""

    label_a: str
    label_b: str
    n_pairs: int
    """Artigos com ficha válida nos DOIS lados (comparáveis)."""
    n_identical: int
    """Pares em que todos os :data:`FIELDS` são iguais."""
    rate_identical: float
    """``n_identical / n_pairs`` (0.0 se não há pares)."""
    field_change_rate: dict[str, float]
    """Por campo: fração dos pares em que o campo mudou."""
    field_mean_similarity: dict[str, float]
    """Por campo: similaridade média entre A e B nos pares."""
    per_arquivo: dict[str, list[FieldDiff]]
    n_unpaired: int
    """Artigos presentes em algum lado sem ficha comparável (FAILED ou ausente)."""
    unpaired: dict[str, str] = field(default_factory=dict)
    """``{arquivo: motivo}`` de cada artigo não pareado."""

    def changed_fields(self, arquivo: str) -> list[str]:
        """Campos que mudaram para um artigo pareado (vazio se idêntico)."""
        return [d.field for d in self.per_arquivo.get(arquivo, []) if not d.equal]

    def n_changed(self, arquivo: str) -> int | None:
        """Número de campos que mudaram; ``None`` se o artigo não foi pareado."""
        if arquivo not in self.per_arquivo:
            return None
        return len(self.changed_fields(arquivo))

    @property
    def changed_fields_by_arquivo(self) -> dict[str, int]:
        """``{arquivo: nº de campos que mudaram}`` para os pares."""
        return {a: len(self.changed_fields(a)) for a in self.per_arquivo}

    def most_unstable_fields(self) -> list[tuple[str, float]]:
        """``[(campo, taxa de mudança)]`` do mais instável ao mais estável.

        Desempate: menor similaridade média primeiro; depois a ordem de :data:`FIELDS`.
        """
        order = {f: i for i, f in enumerate(FIELDS)}
        return sorted(
            self.field_change_rate.items(),
            key=lambda kv: (-kv[1], self.field_mean_similarity.get(kv[0], 1.0), order[kv[0]]),
        )

    def disagreements(self, min_similarity: float = 1.0) -> list[FieldDiff]:
        """Diferenças com similaridade abaixo de ``min_similarity``.

        Com o padrão 1.0, toda diferença (após normalização) é listada; com 0.8, apenas as
        que mudam o conteúdo de forma substantiva.
        """
        return [
            d
            for diffs in self.per_arquivo.values()
            for d in diffs
            if not d.equal and d.similarity < min_similarity
        ]

    def to_frame(self) -> pd.DataFrame:
        """Formato longo: uma linha por (arquivo, campo)."""
        cols = ["arquivo", "field", "a", "b", "equal", "similarity"]
        rows = [d.to_dict() for diffs in self.per_arquivo.values() for d in diffs]
        return pd.DataFrame(rows, columns=cols)

    def field_frame(self) -> pd.DataFrame:
        """Resumo por campo: ``field, change_rate, mean_similarity`` (mais instável primeiro)."""
        rows = [
            {
                "field": f,
                "change_rate": rate,
                "mean_similarity": self.field_mean_similarity.get(f, float("nan")),
            }
            for f, rate in self.most_unstable_fields()
        ]
        return pd.DataFrame(rows, columns=["field", "change_rate", "mean_similarity"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_pairs": self.n_pairs,
            "n_identical": self.n_identical,
            "rate_identical": self.rate_identical,
            "n_unpaired": self.n_unpaired,
            "unpaired": dict(self.unpaired),
            "field_change_rate": dict(self.field_change_rate),
            "field_mean_similarity": dict(self.field_mean_similarity),
            "most_unstable_fields": self.most_unstable_fields(),
        }


def index_by_arquivo(run: Sequence[ExtractionRecord], side: str) -> dict[str, ExtractionRecord]:
    """``{arquivo: registro}``; erro explícito se houver dois registros do mesmo artigo.

    Uma execução deve ter um registro por artigo. Misturar repetições numa mesma lista faria o
    pareamento escolher uma delas silenciosamente — preferimos falhar alto.
    """
    out: dict[str, ExtractionRecord] = {}
    for r in run:
        if r.arquivo in out:
            raise ValueError(
                f"Execução {side} tem mais de um registro para {r.arquivo!r} "
                f"(run_ids {out[r.arquivo].run_id!r} e {r.run_id!r}). Passe uma execução por vez."
            )
        out[r.arquivo] = r
    return out


def _label(run: Sequence[ExtractionRecord], default: str) -> str:
    ids = sorted({r.run_id for r in run})
    return ids[0] if len(ids) == 1 else default


def diff_runs(
    run_a: Sequence[ExtractionRecord],
    run_b: Sequence[ExtractionRecord],
    label_a: str | None = None,
    label_b: str | None = None,
) -> DiffReport:
    """Compara duas execuções pareando por ``arquivo`` (ver regras no docstring do módulo)."""
    ia, ib = index_by_arquivo(run_a, "A"), index_by_arquivo(run_b, "B")
    per_arquivo: dict[str, list[FieldDiff]] = {}
    unpaired: dict[str, str] = {}
    for arquivo in sorted(set(ia) | set(ib)):
        ra, rb = ia.get(arquivo), ib.get(arquivo)
        if ra is None or rb is None:
            unpaired[arquivo] = "ausente em A" if ra is None else "ausente em B"
            continue
        if ra.ficha is None or rb.ficha is None:
            lados = [s for s, r in (("A", ra), ("B", rb)) if r.ficha is None]
            unpaired[arquivo] = f"extração falhou em {' e '.join(lados)}"
            continue
        per_arquivo[arquivo] = diff_fichas(arquivo, ra.ficha, rb.ficha)

    n_pairs = len(per_arquivo)
    n_identical = sum(all(d.equal for d in diffs) for diffs in per_arquivo.values())
    change: dict[str, float] = {}
    sim: dict[str, float] = {}
    for f in FIELDS:
        diffs_f = [d for diffs in per_arquivo.values() for d in diffs if d.field == f]
        change[f] = sum(not d.equal for d in diffs_f) / n_pairs if n_pairs else 0.0
        sim[f] = sum(d.similarity for d in diffs_f) / n_pairs if n_pairs else 0.0
    return DiffReport(
        label_a=label_a or _label(run_a, "A"),
        label_b=label_b or _label(run_b, "B"),
        n_pairs=n_pairs,
        n_identical=n_identical,
        rate_identical=n_identical / n_pairs if n_pairs else 0.0,
        field_change_rate=change,
        field_mean_similarity=sim,
        per_arquivo=per_arquivo,
        n_unpaired=len(unpaired),
        unpaired=unpaired,
    )
