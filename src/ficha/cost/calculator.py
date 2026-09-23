"""Custo (Seção 4.5), com premissa de preço declarada e visível junto de cada resultado.

Contas:

- ``usd_input  = input_tokens  / 1_000_000 × preço_entrada_por_Mtok``
- ``usd_output = output_tokens / 1_000_000 × preço_saída_por_Mtok``
- ``usd_total  = usd_input + usd_output``

O custo **real** soma o ``Usage`` de cada chamada efetivamente feita — inclusive repetições
de estabilidade, a outra estratégia, a outra temperatura e a outra variante de prompt
(passe todos os registros). A alternativa **ingênua** manda o artigo inteiro em cada
chamada, com o mesmo número de chamadas por artigo, e usa uma estimativa declarada de
tokens de saída. Mesmo quando o modelo roda de graça (Qwen no Colab), a premissa responde
"quanto isso custaria" num provedor pago.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ficha.config import Settings
from ficha.types import Document, ExtractionRecord

_MTOK = 1_000_000


@dataclass(frozen=True, slots=True)
class PricePremise:
    """Premissa de preço em USD por 1 milhão de tokens."""

    input_per_mtok: float
    output_per_mtok: float
    source: str

    @classmethod
    def from_settings(cls, settings: Settings) -> PricePremise:
        """Lê a premissa de :class:`ficha.config.Settings` (``FICHA_PRICE_*``)."""
        return cls(
            input_per_mtok=settings.price_input_per_mtok,
            output_per_mtok=settings.price_output_per_mtok,
            source=settings.price_source,
        )

    def describe(self) -> str:
        """Frase para exibir junto de qualquer número de custo."""
        return (
            f"Premissa: US$ {self.input_per_mtok:.2f} por 1M tokens de entrada e "
            f"US$ {self.output_per_mtok:.2f} por 1M tokens de saída (fonte: {self.source})."
        )

    def price(self, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        """``(usd_entrada, usd_saída)``."""
        return (
            input_tokens / _MTOK * self.input_per_mtok,
            output_tokens / _MTOK * self.output_per_mtok,
        )


@dataclass(frozen=True, slots=True)
class CostReport:
    """Tokens e custo de um conjunto de chamadas."""

    label: str
    n_calls: int
    input_tokens: int
    output_tokens: int
    usd_input: float
    usd_output: float
    usd_total: float
    premise: PricePremise

    @classmethod
    def build(
        cls,
        label: str,
        n_calls: int,
        input_tokens: int,
        output_tokens: int,
        premise: PricePremise,
    ) -> CostReport:
        usd_in, usd_out = premise.price(input_tokens, output_tokens)
        return cls(
            label=label,
            n_calls=n_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd_input=usd_in,
            usd_output=usd_out,
            usd_total=usd_in + usd_out,
            premise=premise,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "usd_input": self.usd_input,
            "usd_output": self.usd_output,
            "usd_total": self.usd_total,
            "premissa": self.premise.describe(),
        }


def cost_of_records(
    records: Sequence[ExtractionRecord], premise: PricePremise, label: str = "real"
) -> CostReport:
    """Custo real: soma o ``Usage`` de cada chamada registrada (uma por registro)."""
    return CostReport.build(
        label=label,
        n_calls=len(records),
        input_tokens=sum(r.usage.input_tokens for r in records),
        output_tokens=sum(r.usage.output_tokens for r in records),
        premise=premise,
    )


def cost_by_run(records: Sequence[ExtractionRecord], premise: PricePremise) -> list[CostReport]:
    """Um :class:`CostReport` por ``run_id`` (ordem alfabética), para a tabela detalhada."""
    runs: dict[str, list[ExtractionRecord]] = {}
    for r in records:
        runs.setdefault(r.run_id, []).append(r)
    return [cost_of_records(rs, premise, label=rid) for rid, rs in sorted(runs.items())]


def naive_cost(
    docs: Sequence[Document],
    count_tokens: Callable[[str], int],
    premise: PricePremise,
    n_calls_per_doc: int = 1,
    expected_output_tokens: int = 300,
    prompt_overhead_tokens: int = 0,
    label: str = "ingênuo (artigo inteiro)",
) -> CostReport:
    """Custo da alternativa ingênua: o artigo INTEIRO no prompt de cada chamada.

    - entrada por chamada = ``count_tokens(doc.full_text) + prompt_overhead_tokens``;
    - saída por chamada = ``expected_output_tokens`` (estimativa declarada: uma ficha JSON);
    - ``n_calls_per_doc`` deve igualar o nº de chamadas por artigo da execução real, para que
      a comparação seja do mesmo experimento com outra entrada.

    Não considera o limite de contexto do modelo: artigos que não caberiam são contados
    mesmo assim — é o custo *se* fosse possível.
    """
    per_doc_in = [count_tokens(d.full_text) + prompt_overhead_tokens for d in docs]
    n_calls = len(docs) * n_calls_per_doc
    return CostReport.build(
        label=label,
        n_calls=n_calls,
        input_tokens=sum(per_doc_in) * n_calls_per_doc,
        output_tokens=expected_output_tokens * n_calls,
        premise=premise,
    )


def naive_cost_from_records(
    records: Sequence[ExtractionRecord],
    docs: Sequence[Document] | Mapping[str, Document],
    count_tokens: Callable[[str], int],
    premise: PricePremise,
    label: str = "ingênuo (artigo inteiro, mesmas chamadas)",
) -> CostReport:
    """Alternativa ingênua espelhando EXATAMENTE as chamadas reais.

    Para cada registro (repetições, subconjunto de temperatura etc. incluídos):

    - entrada = ``usage.input_tokens - count_tokens(context_text) + count_tokens(full_text)``,
      isto é, o mesmo prompt (instruções, few-shot, delimitadores) com o artigo inteiro no
      lugar da seleção (piso em 0 para aproximações de contagem);
    - saída = ``usage.output_tokens`` (a mesma ficha seria devolvida);
    - ``n_calls = len(records)``.

    Levanta ``KeyError`` se algum ``arquivo`` não estiver em ``docs``.
    """
    by_name = dict(docs) if isinstance(docs, Mapping) else {d.arquivo: d for d in docs}
    full_tokens: dict[str, int] = {}
    input_tokens = 0
    for r in records:
        if r.arquivo not in by_name:
            raise KeyError(
                f"naive_cost_from_records: {r.arquivo!r} (run {r.run_id!r}) não está em docs; "
                "passe o Document de todos os artigos das execuções."
            )
        if r.arquivo not in full_tokens:
            full_tokens[r.arquivo] = count_tokens(by_name[r.arquivo].full_text)
        overhead = r.usage.input_tokens - count_tokens(r.context_text)
        input_tokens += max(0, overhead + full_tokens[r.arquivo])
    return CostReport.build(
        label=label,
        n_calls=len(records),
        input_tokens=input_tokens,
        output_tokens=sum(r.usage.output_tokens for r in records),
        premise=premise,
    )


@dataclass(frozen=True, slots=True)
class CostComparison:
    """Quantas vezes a alternativa ingênua é mais cara que a real."""

    real: CostReport
    naive: CostReport
    ratio: float
    """``naive.usd_total / real.usd_total`` (``inf`` se o real custa 0)."""
    token_ratio: float
    """``naive.input_tokens / real.input_tokens`` — independe da premissa de preço."""
    savings_usd: float
    savings_pct: float
    """``savings_usd / naive.usd_total`` (0-1)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "real_usd": self.real.usd_total,
            "naive_usd": self.naive.usd_total,
            "ratio": self.ratio,
            "token_ratio": self.token_ratio,
            "savings_usd": self.savings_usd,
            "savings_pct": self.savings_pct,
            "premissa": self.real.premise.describe(),
        }

    def describe(self) -> str:
        """Frase pronta para o relatório."""
        return (
            f"Enviar o artigo inteiro custaria US$ {self.naive.usd_total:.4f} contra "
            f"US$ {self.real.usd_total:.4f} efetivamente gastos: {self.ratio:.1f}x mais caro "
            f"({self.token_ratio:.1f}x mais tokens de entrada). {self.real.premise.describe()}"
        )


def compare_costs(real: CostReport, naive: CostReport) -> CostComparison:
    """Compara o custo real com o ingênuo (mesma premissa esperada)."""
    if real.premise != naive.premise:
        raise ValueError("Custos com premissas de preço diferentes não são comparáveis.")
    ratio = naive.usd_total / real.usd_total if real.usd_total else float("inf")
    token_ratio = naive.input_tokens / real.input_tokens if real.input_tokens else float("inf")
    savings = naive.usd_total - real.usd_total
    return CostComparison(
        real=real,
        naive=naive,
        ratio=ratio,
        token_ratio=token_ratio,
        savings_usd=savings,
        savings_pct=savings / naive.usd_total if naive.usd_total else 0.0,
    )


def cost_table(reports: Sequence[CostReport]) -> pd.DataFrame:
    """Tabela com uma linha por relatório e a premissa em coluna própria (sempre visível)."""
    cols = [
        "label",
        "n_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "usd_input",
        "usd_output",
        "usd_total",
        "premissa",
    ]
    return pd.DataFrame([r.to_dict() for r in reports], columns=cols)
