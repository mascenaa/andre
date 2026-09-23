"""Relatório PDF de no máximo 3 páginas (Seção 5, entregável 3), com ``reportlab``.

Decisões:

- **A4, fonte com acentos.** Usamos DejaVu Sans (TrueType, cobre "ção", "ê", "≤", "→") quando
  ela está disponível — ela vem junto do ``matplotlib``. Sem ela, caímos para Helvetica, que
  cobre Latin-1 (todos os acentos do português) mas não símbolos como "→"; nesse caso esses
  símbolos são trocados por equivalentes ASCII.
- **Identificação na primeira página**: título + integrantes logo no topo (Seção 5).
- **Seções numeradas** na ordem dos itens da Seção 5, tabelas compactas, rodapé com
  "página X de Y".
- **Limite rígido de páginas.** Depois de gerar, contamos as páginas com ``pymupdf``. Se o
  conteúdo tiver figuras e estourar o limite, tentamos de novo sem figuras; se ainda assim
  passar, levantamos :class:`ReportTooLongError` — o PDF fica gravado para inspeção, mas a
  entrega não pode seguir assim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ficha.report.content import AuditBlock, ReportContent, TableLike, format_cell, table_rows

PAGE_SIZE = A4
MARGIN = 1.5 * cm
CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN
FIGURE_MAX_HEIGHT = 5.5 * cm
"""Altura máxima de uma figura no relatório (o limite de 3 páginas manda)."""

SECTION_TITLES: tuple[str, ...] = (
    "1. Estratégia de seleção do contexto (4.1)",
    "2. Prompt: duas versões e a comparação (4.2)",
    "3. Modelo e temperatura (4.2–4.3)",
    "4. Auditoria: as quatro verificações (4.4)",
    "5. Fichas que não defenderíamos",
    "6. Custo (4.5)",
    "7. O que faríamos diferente com mais uma semana",
    "8. Declaração de uso de IA",
)
"""Títulos das seções, na ordem dos itens da Seção 5 (usados também nos testes)."""

_ASCII_FALLBACK = {"→": "->", "≤": "<=", "≥": ">=", "×": "x", "≈": "~", "—": "-", "–": "-"}


class ReportTooLongError(RuntimeError):
    """O PDF gerado passou do limite de páginas do enunciado."""

    def __init__(self, n_pages: int, max_pages: int, path: Path) -> None:
        self.n_pages = n_pages
        self.max_pages = max_pages
        self.path = path
        super().__init__(
            f"Relatório com {n_pages} páginas (limite: {max_pages}). Arquivo: {path}. "
            "Encurte justificativas/tabelas antes de entregar."
        )


@lru_cache(maxsize=1)
def _fonts() -> tuple[str, str]:
    """Registra DejaVu Sans (se disponível) e devolve ``(regular, negrito)``."""
    try:
        import matplotlib

        ttf_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(ttf_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(ttf_dir / "DejaVuSans-Bold.ttf")))
    except Exception:  # matplotlib ausente ou fonte não encontrada: Latin-1 basta p/ acentos
        return "Helvetica", "Helvetica-Bold"
    pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold")
    return "DejaVuSans", "DejaVuSans-Bold"


class _Styles:
    """Estilos tipográficos compactos (o limite de 3 páginas manda)."""

    def __init__(self) -> None:
        regular, bold = _fonts()
        self.unicode_ok = regular.startswith("DejaVu")
        self.title = ParagraphStyle("title", fontName=bold, fontSize=13, leading=16)
        self.subtitle = ParagraphStyle(
            "subtitle", fontName=regular, fontSize=8, leading=10, textColor=colors.grey
        )
        self.members = ParagraphStyle(
            "members", fontName=regular, fontSize=9, leading=12, spaceBefore=4
        )
        self.h2 = ParagraphStyle(
            "h2",
            fontName=bold,
            fontSize=9.5,
            leading=12,
            spaceBefore=11,
            spaceAfter=4,
            textColor=colors.HexColor("#1F3A5F"),
            keepWithNext=1,  # título nunca fica órfão no fim da página
        )
        self.body = ParagraphStyle("body", fontName=regular, fontSize=8, leading=10.5, spaceAfter=3)
        self.cell = ParagraphStyle("cell", fontName=regular, fontSize=7, leading=8.4)
        self.cell_head = ParagraphStyle("cell_head", fontName=bold, fontSize=7, leading=8.4)
        self.caption = ParagraphStyle(
            "caption", fontName=regular, fontSize=6.5, leading=8, alignment=TA_CENTER
        )
        self.footer_font = regular


class _NumberedCanvas(Canvas):  # type: ignore[misc]
    """Canvas que conhece o total de páginas, para o rodapé "página X de Y"."""

    footer_label = "Atividade de Construção I"
    footer_font = "Helvetica"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved: list[dict[str, Any]] = []

    def showPage(self) -> None:  # noqa: N802 — nome imposto pelo reportlab
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.setFont(self.footer_font, 7)
            self.setFillColor(colors.grey)
            self.drawString(MARGIN, 0.8 * cm, self.footer_label)
            self.drawRightString(
                PAGE_SIZE[0] - MARGIN, 0.8 * cm, f"página {self._pageNumber} de {total}"
            )
            super().showPage()
        super().save()


class _Builder:
    """Transforma um :class:`ReportContent` em flowables do platypus."""

    def __init__(self, content: ReportContent, with_figures: bool) -> None:
        self.c = content
        self.s = _Styles()
        self.with_figures = with_figures

    # ---------------------------------------------------------------- primitivas
    def text(self, value: str) -> str:
        """Escapa para o mini-HTML do reportlab e troca símbolos sem glifo na fonte."""
        if not self.s.unicode_ok:
            for k, v in _ASCII_FALLBACK.items():
                value = value.replace(k, v)
        # Quebras de linha do texto (ex.: a regra de confiança em tópicos) são preservadas.
        return escape(value).replace("\n", "<br/>")

    def para(self, value: str, style: ParagraphStyle | None = None) -> Paragraph:
        return Paragraph(self.text(value), style or self.s.body)

    def lead(self, label: str, value: str) -> Paragraph:
        """Parágrafo com rótulo em negrito ("<b>Rótulo.</b> texto")."""
        return Paragraph(f"<b>{self.text(label)}</b> {self.text(value)}", self.s.body)

    def table(self, data: TableLike, width: float = CONTENT_WIDTH) -> Table:
        headers, rows = table_rows(data)
        return self._grid(headers, rows, width)

    def _grid(self, headers: list[str], rows: list[list[str]], width: float) -> Table:
        cells: list[list[Flowable]] = [[Paragraph(self.text(h), self.s.cell_head) for h in headers]]
        cells += [[Paragraph(self.text(v), self.s.cell) for v in row] for row in rows]
        # Largura proporcional ao conteúdo, com piso e teto para não esmagar colunas curtas.
        weights = []
        for j, h in enumerate(headers):
            longest = max([len(h), *(len(r[j]) for r in rows)]) if rows else len(h)
            weights.append(min(max(longest, len(h) + 2, 6), 70))
        total = sum(weights) or 1
        # Tabelas estreitas (poucos números) não precisam ocupar a largura toda.
        width = min(width, total * 4.6 + 12 * len(headers))
        col_widths = [width * w / total for w in weights]
        tbl = Table(cells, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EBF0")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#1F3A5F")),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.3, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        return tbl

    def numbers(self, data: TableLike | Mapping[str, Any]) -> Flowable:
        """Números de um bloco: dicionário vira linha compacta; tabela vira grade."""
        if isinstance(data, Mapping):
            parts = [
                f"<b>{self.text(str(k))}</b>: {self.text(format_cell(v))}" for k, v in data.items()
            ]
            return Paragraph(" · ".join(parts), self.s.body)
        return self.table(data)

    def section(self, index: int) -> Paragraph:
        return Paragraph(self.text(SECTION_TITLES[index]), self.s.h2)

    # ---------------------------------------------------------------- seções
    def header(self) -> list[Flowable]:
        c = self.c
        return [
            self.para(c.titulo, self.s.title),
            self.para(c.subtitulo, self.s.subtitle),
            Spacer(1, 6),
            Paragraph(
                f"<b>Integrantes:</b> {self.text(', '.join(n for n in c.integrantes if n))}",
                self.s.members,
            ),
            Spacer(1, 5),
        ]

    def estrategia(self) -> list[Flowable]:
        c = self.c
        out: list[Flowable] = [
            self.section(0),
            self.lead("Escolhida:", c.estrategia_escolhida),
            self.lead("Por quê:", c.estrategia_justificativa),
        ]
        if c.estrategia_tabela is not None:
            out.append(self.table(c.estrategia_tabela))
        if c.estrategia_evidencia:
            out += [Spacer(1, 5), self.lead("Evidência medida:", c.estrategia_evidencia)]
        return out

    def prompt(self) -> list[Flowable]:
        c = self.c
        versions = self._grid(
            ["versão", "técnicas", "descrição"],
            [[v.nome, v.features, v.descricao] for v in (c.prompt_v1, c.prompt_v2)],
            CONTENT_WIDTH,
        )
        return [
            self.section(1),
            versions,
            Spacer(1, 5),
            self.lead("O que muda:", c.prompt_diff),
            self.table(c.prompt_comparacao),
            Spacer(1, 5),
            self.lead("O que a comparação mostrou:", c.prompt_conclusao),
        ]

    def modelo(self) -> list[Flowable]:
        c = self.c
        return [
            self.section(2),
            self.lead("Modelo:", f"{c.modelo}. {c.modelo_justificativa}"),
            self.lead("Temperatura:", c.temperatura_justificativa),
        ]

    def _audit_block(self, block: AuditBlock) -> Flowable:
        return KeepTogether(
            [self.lead(f"{block.titulo}.", block.conclusao), self.numbers(block.numeros)]
        )

    def auditoria(self) -> list[Flowable]:
        c = self.c
        out: list[Flowable] = [self.section(3)]
        for i, block in enumerate(c.resultados_auditoria.blocks()):
            out += [self._audit_block(block), Spacer(1, 5)]
            if i == 0 and c.destaque_auditoria:  # logo após a fidelidade, onde foi detectado
                out += [Paragraph(f"<b>{self.text(c.destaque_auditoria)}</b>", self.s.body)]
        out.append(self.lead("Regra de confiança (declarada):", c.regra_confianca))
        if self.with_figures and c.figuras:
            out.append(self.figures(c.figuras))
        return out

    def figures(self, paths: Sequence[Path]) -> Flowable:
        """Figuras lado a lado, com altura limitada a :data:`FIGURE_MAX_HEIGHT`."""
        n = len(paths)
        width = CONTENT_WIDTH / n
        imgs: list[Flowable] = []
        for p in paths:
            img = Image(str(p))
            ratio = img.imageHeight / img.imageWidth if img.imageWidth else 0.6
            draw_w = width - 6
            draw_h = draw_w * ratio
            if draw_h > FIGURE_MAX_HEIGHT:  # preserva a proporção
                draw_w, draw_h = FIGURE_MAX_HEIGHT / ratio, FIGURE_MAX_HEIGHT
            img.drawWidth, img.drawHeight = draw_w, draw_h
            imgs.append(img)
        tbl = Table([imgs], colWidths=[width] * n, hAlign="CENTER")
        tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return tbl

    def nao_defensaveis(self) -> list[Flowable]:
        headers, rows = table_rows(self.c.fichas_nao_defensaveis)
        body: Flowable = (
            self._grid(headers, rows, CONTENT_WIDTH)
            if rows
            else self.para("Nenhuma ficha foi classificada como não defensável.")
        )
        return [self.section(4), body]

    def custo(self) -> list[Flowable]:
        cost = self.c.custo
        razao = format_cell(round(cost.razao_ingenuo_real, 1))
        out: list[Flowable] = [
            self.section(5),
            (
                self.para(cost.premissa)
                if cost.premissa.lower().startswith("premissa")
                else self.lead("Premissa de preço:", cost.premissa)
            ),
            self.table(cost.tabela),
            Spacer(1, 5),
            self.lead("Razão ingênuo/real:", f"a alternativa ingênua custa {razao}× a real."),
        ]
        if cost.comentario:
            out.append(self.para(cost.comentario))
        return out

    def futuro(self) -> list[Flowable]:
        items = [self.para(f"• {item}") for item in self.c.o_que_faria_diferente]
        return [self.section(6), *items]

    def declaracao(self) -> list[Flowable]:
        return [self.section(7), self.para(self.c.declaracao_uso_ia)]

    def story(self) -> list[Flowable]:
        return [
            *self.header(),
            *self.estrategia(),
            *self.prompt(),
            *self.modelo(),
            *self.auditoria(),
            *self.nao_defensaveis(),
            *self.custo(),
            *self.futuro(),
            *self.declaracao(),
        ]


def count_pages(path: Path) -> int:
    """Número de páginas de um PDF (via ``pymupdf``)."""
    with pymupdf.open(path) as pdf:  # type: ignore[no-untyped-call]
        return int(pdf.page_count)


def _render(content: ReportContent, out_path: Path, with_figures: bool) -> int:
    builder = _Builder(content, with_figures=with_figures)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=content.titulo,
        author=", ".join(content.integrantes),
        subject="Atividade de Construção I — ficha comparativa auditável",
    )
    canvas_cls = type(
        "_Canvas",
        (_NumberedCanvas,),
        {"footer_font": builder.s.footer_font, "footer_label": content.subtitulo},
    )
    doc.build(builder.story(), canvasmaker=canvas_cls)
    return count_pages(out_path)


def build_report_pdf(content: ReportContent, out_path: Path, max_pages: int = 3) -> Path:
    """Gera o relatório PDF e garante o limite de páginas.

    Args:
        content: conteúdo com todos os itens da Seção 5.
        out_path: caminho do PDF (``AtividadeI_<sobrenomes>.pdf``); diretórios são criados.
        max_pages: limite de páginas (3, pelo enunciado).

    Returns:
        ``out_path``.

    Raises:
        ReportTooLongError: se, mesmo sem figuras, o PDF passar de ``max_pages``.

    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pages = _render(content, out_path, with_figures=True)
    if n_pages > max_pages and content.figuras:
        n_pages = _render(replace(content, figuras=[]), out_path, with_figures=False)
    if n_pages > max_pages:
        raise ReportTooLongError(n_pages, max_pages, out_path)
    return out_path
