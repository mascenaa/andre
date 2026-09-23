"""Figuras do notebook (e, se couberem, do relatório): matplotlib, estilo sóbrio.

Cada função salva um PNG e devolve o ``Path``. O backend ``Agg`` é forçado para que as
figuras sejam geradas também sem display (CI, ``nbconvert --execute``).

As funções aceitam ``DataFrame`` em formas tolerantes, porque os quadros vêm da auditoria:

- **comparação de prompts**: uma linha por métrica (no índice ou numa coluna de texto) e uma
  coluna numérica por variante — ou o transposto (uma linha por variante); a função usa as
  colunas numéricas como séries e o rótulo textual como categoria.
- **taxa de mudança por campo**: uma coluna de texto (o campo) e uma coluna numérica (a taxa),
  ou o índice como campo.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # o backend precisa ser fixado antes do pyplot
import pandas as pd
from matplotlib.axes import Axes

from ficha.schema import Confianca, Ficha

PALETTE: tuple[str, ...] = ("#1F3A5F", "#8FA9C4", "#B5838D", "#6D6875", "#A3B18A")
"""Tons sóbrios (azul-marinho, azul-acinzentado, rosé, cinza, oliva)."""

METRIC_LABELS: dict[str, str] = {
    "rate_json_valid_first_try": "JSON válido de primeira",
    "rate_parse_ok": "parse ok",
    "rate_parse_repaired": "parse reparado",
    "rate_parse_failed": "parse falhou",
    "rate_limitacao_null": "limitação null",
    "rate_null_when_expected": "null quando devido",
    "rate_fidelity": "trecho encontrado",
    "rate_page_ok": "página correta",
    "rate_identical_entre_variantes": "fichas idênticas",
    "change_rate": "taxa de mudança",
}
"""Rótulos em português para as métricas conhecidas da auditoria (o resto fica como está)."""

_CONF_COLORS = {"alta": "#4F7942", "media": "#C9A227", "baixa": "#9E2A2B", "sem": "#9A9A9A"}


def _out(out_path: Path | str | None, default_name: str) -> Path:
    if out_path is None:
        return Path(tempfile.mkdtemp(prefix="ficha_fig_")) / default_name
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _label_and_values(frame: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Separa rótulos (coluna textual ou índice) das colunas numéricas."""
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("O quadro não tem colunas numéricas para plotar.")
    text_cols = [c for c in frame.columns if c not in numeric.columns]
    labels = [str(v) for v in frame[text_cols[0]]] if text_cols else [str(i) for i in frame.index]
    return [METRIC_LABELS.get(label, label) for label in labels], numeric


def _style(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, loc="left", color="#222222")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_prompt_comparison(
    frame: pd.DataFrame,
    out_path: Path | str | None = None,
    title: str = "Comparação das versões do prompt",
) -> Path:
    """Barras horizontais agrupadas: cada métrica × cada variante do prompt."""
    labels, numeric = _label_and_values(frame)
    path = _out(out_path, "comparacao_prompts.png")
    n_series = len(numeric.columns)
    height = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.45 * len(labels) * max(1, n_series / 1.5)))
    positions = range(len(labels))
    for k, col in enumerate(numeric.columns):
        ys = [p + k * height for p in positions]
        values = [float(v) for v in numeric[col]]
        bars = ax.barh(ys, values, height=height, color=PALETTE[k % len(PALETTE)], label=str(col))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_yticks([p + height * (n_series - 1) / 2 for p in positions], labels)
    ax.invert_yaxis()
    if float(numeric.max().max()) <= 1:
        ax.set_xlim(0, 1.1)
    _style(ax, title)
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=n_series,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_field_change_rates(
    frame: pd.DataFrame,
    out_path: Path | str | None = None,
    title: str = "Taxa de mudança por campo entre execuções",
) -> Path:
    """Barras horizontais com a taxa de mudança de cada campo (primeira coluna numérica)."""
    labels, numeric = _label_and_values(frame)
    values = [float(v) for v in numeric.iloc[:, 0]]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    path = _out(out_path, "mudanca_por_campo.png")
    fig, ax = plt.subplots(figsize=(6.0, 0.6 + 0.35 * len(labels)))
    bars = ax.barh([labels[i] for i in order], [values[i] for i in order], color=PALETTE[0])
    ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.invert_yaxis()
    col = str(numeric.columns[0])
    ax.set_xlabel(METRIC_LABELS.get(col, col), fontsize=8)
    if values and max(values) <= 1:
        ax.set_xlim(0, 1.08)
    _style(ax, title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confidence_distribution(
    fichas: Sequence[Ficha],
    out_path: Path | str | None = None,
    title: str = "Distribuição da confiança atribuída pela regra",
) -> Path:
    """Contagem de fichas por nível de confiança (inclui "sem" para não auditadas)."""
    counts = Counter(f.confianca.value if f.confianca else "sem" for f in fichas)
    levels = [c.value for c in Confianca] + (["sem"] if counts.get("sem") else [])
    values = [counts.get(level, 0) for level in levels]
    path = _out(out_path, "distribuicao_confianca.png")
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    bars = ax.bar(levels, values, color=[_CONF_COLORS[level] for level in levels], width=0.6)
    ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_ylabel("fichas", fontsize=8)
    ax.set_ylim(0, max([*values, 1]) * 1.2)
    _style(ax, title)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
