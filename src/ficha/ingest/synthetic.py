"""Artigos científicos **sintéticos** em PDF, para testes e para o modo de ensaio do notebook.

Os PDFs imitam os defeitos que a limpeza precisa tratar em artigos reais:

- cabeçalho repetido no topo de todas as páginas (nome da revista);
- número de página solto no rodapé;
- **hifenização forçada em fim de linha**: palavras longas que não cabem na linha são quebradas
  com hífen (``evalua-`` / ``tion``), como faz o TeX; compostos (``state-of-the-art``) são
  quebrados no hífen legítimo;
- estrutura de artigo real (título, autores, Abstract, Introduction, Related Work, Data, Methods,
  Results, Limitations **ou não**, Discussion, Conclusion), com três estilos de numeração de seção
  (``3 Methods``, ``3. Methods``, ``III. METHODS``).

Artigos de índice ímpar (``artigo_01``, ``artigo_03``...) declaram limitações; os de índice par
não declaram nenhuma e **não contêm a palavra "limitation"** — o valor correto de ``limitacao``
para eles é ``null``. Isso vai para ``synthetic_manifest.json`` como gabarito.

Tudo é determinístico dado o ``seed``. O layout é feito à mão (e não com ``Paragraph`` do
reportlab) justamente para controlar onde cada linha quebra.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from ficha.ingest.clean import COMPOUND_PREFIXES

PAGE_W, PAGE_H = letter
MARGIN_X = 72.0
MARGIN_TOP = 72.0
MARGIN_BOTTOM = 72.0
TEXT_W = PAGE_W - 2 * MARGIN_X

BODY_FONT, BODY_SIZE, BODY_LEADING = "Times-Roman", 10.5, 14.0
HEAD_FONT, HEAD_SIZE = "Times-Bold", 12.0
TITLE_FONT, TITLE_SIZE = "Times-Bold", 16.0
PARA_GAP = 9.0
MIN_HYPHEN_WORD = 8
"""Só palavras com pelo menos 8 letras são hifenizadas."""

MANIFEST_NAME = "synthetic_manifest.json"


# --------------------------------------------------------------------------- conteúdo

TOPICS: list[dict[str, str]] = [
    {
        "title": "Predicting Thirty-Day Hospital Readmission from Electronic Health Records",
        "problem": "forecasting hospital readmission within thirty days of discharge",
        "data": (
            "We use the MIMIC-IV database, covering admissions from 2008 to 2019 and comprising "
            "431,231 hospitalizations from a single tertiary hospital in Boston."
        ),
        "method": (
            "Structured features are combined with TF-IDF representations of discharge notes "
            "and fed to a gradient boosted classifier tuned with five-fold cross-validation."
        ),
        "metric": "area under the ROC curve (AUROC) and area under the precision-recall curve",
        "result": (
            "The model reaches an AUROC of 0.81 on the held-out test set, outperforming the "
            "LACE index baseline by 0.09."
        ),
        "limitation": (
            "Data come from a single institution, so external validity is unknown, and the "
            "clinical notes were written in English only."
        ),
    },
    {
        "title": "Aspect-Level Sentiment Classification of Product Reviews",
        "problem": "classifying the sentiment expressed about specific product aspects",
        "data": (
            "We collected 182,000 customer reviews of consumer electronics published on an "
            "online marketplace between January 2016 and December 2018."
        ),
        "method": (
            "A pretrained transformer encoder is fine-tuned with an auxiliary aspect-extraction "
            "objective and a weighted cross-entropy loss."
        ),
        "metric": "macro-averaged F1 score and accuracy on a manually annotated test split",
        "result": (
            "Our approach obtains a macro F1 of 0.742, an improvement of 3.1 points over the "
            "strongest baseline."
        ),
        "limitation": (
            "The annotation was performed by only two annotators, and the model was evaluated "
            "on a single product category."
        ),
    },
    {
        "title": "Credit Default Prediction with Interpretable Ensembles",
        "problem": "predicting default of credit card clients in a transparent way",
        "data": (
            "The dataset contains 30,000 credit card holders from a Taiwanese bank observed "
            "between April and September 2005, with 23 explanatory variables."
        ),
        "method": (
            "We train monotonic gradient boosting ensembles and compare them with logistic "
            "regression using Shapley additive explanations."
        ),
        "metric": "Kolmogorov-Smirnov statistic and the area under the ROC curve",
        "result": (
            "The monotonic ensemble achieves a KS statistic of 0.47 while keeping every "
            "feature effect monotonic."
        ),
        "limitation": (
            "The period covered is short and predates recent changes in consumer credit "
            "regulation, which may reduce the relevance of our findings."
        ),
    },
    {
        "title": "Crop Yield Estimation from Multispectral Satellite Imagery",
        "problem": "estimating soybean yield at municipality level before harvest",
        "data": (
            "We combine Sentinel-2 multispectral imagery from 2017 to 2022 with official "
            "yield statistics for 1,112 municipalities."
        ),
        "method": (
            "A convolutional network processes monthly composites and a recurrent layer "
            "aggregates the temporal sequence of the growing season."
        ),
        "metric": "root mean squared error (RMSE) in kilograms per hectare",
        "result": (
            "The model reaches an RMSE of 312 kg/ha, a reduction of 18 percent relative to "
            "the regression baseline."
        ),
        "limitation": (
            "Cloud cover leaves gaps in the imagery for some regions, and the approach was not "
            "tested on crops other than soybean."
        ),
    },
    {
        "title": "Customer Churn Modeling with Survival Analysis",
        "problem": "anticipating when telecom subscribers will cancel their contracts",
        "data": (
            "The study uses anonymized records of 95,400 prepaid subscribers of a regional "
            "telecom operator observed over 24 months."
        ),
        "method": (
            "We fit a random survival forest and a Cox proportional hazards model with "
            "time-varying covariates derived from usage logs."
        ),
        "metric": "concordance index (C-index) and the integrated Brier score",
        "result": (
            "The random survival forest achieves a C-index of 0.79 against 0.72 for the Cox model."
        ),
        "limitation": (
            "Contract terminations caused by relocation could not be distinguished from "
            "voluntary churn in the available records."
        ),
    },
    {
        "title": "Traffic Speed Forecasting with Graph Neural Networks",
        "problem": "forecasting traffic speed on highway sensors up to one hour ahead",
        "data": (
            "We use the PeMS-Bay dataset with readings from 325 loop detectors in the San "
            "Francisco Bay Area collected every five minutes over six months."
        ),
        "method": (
            "A spatio-temporal graph neural network with diffusion convolution models the "
            "road network and a sequence-to-sequence decoder produces the forecasts."
        ),
        "metric": "mean absolute error (MAE) at 15, 30 and 60 minute horizons",
        "result": (
            "At the 60 minute horizon the model obtains an MAE of 1.95 mph, compared with "
            "2.24 mph for the strongest recurrent baseline."
        ),
        "limitation": (
            "Sensor failures were imputed with simple interpolation, and the model was not "
            "evaluated under incidents or extreme weather."
        ),
    },
]

FILLER_SENTENCES: list[str] = [
    "Previous research has emphasized the importance of reproducibility and careful "
    "experimental documentation.",
    "Several contributions rely on handcrafted representations whose generalization to "
    "unseen institutions remains questionable.",
    "Recent approaches adopt state-of-the-art architectures and report substantial "
    "improvements on established benchmarks.",
    "Interpretability has become a central requirement for practitioners deploying "
    "predictive systems in regulated environments.",
    "The computational requirements of these methods vary considerably across "
    "configurations and hardware platforms.",
    "A comprehensive characterization of preprocessing decisions is rarely provided by "
    "the original authors.",
    "Comparative benchmarking studies frequently disagree about the relative performance "
    "of competing techniques.",
    "Our implementation follows established recommendations for hyperparameter "
    "optimization and early stopping.",
    "The experimental protocol was designed to isolate the contribution of each "
    "architectural component.",
    "Additional experiments confirmed that the conclusions are consistent across random "
    "initializations of the network.",
    "Practitioners typically combine quantitative evaluation with qualitative inspection "
    "of representative examples.",
    "The literature distinguishes between predictive accuracy and the operational "
    "usefulness of the resulting recommendations.",
    "Transformations of the input variables were standardized using statistics computed "
    "exclusively on the training partition.",
    "The organizational context in which predictions are consumed influences the choice "
    "of decision thresholds.",
]

LIMITATIONS_FILLER = (
    "Our study has limitations that should be considered when interpreting the results."
)


# --------------------------------------------------------------------------- estrutura


@dataclass(frozen=True, slots=True)
class Block:
    """Um elemento do artigo antes do layout."""

    kind: str
    """``title``, ``authors``, ``heading`` ou ``para``."""
    text: str


@dataclass(frozen=True, slots=True)
class DrawOp:
    """Uma linha a desenhar numa página."""

    x: float
    y: float
    text: str
    font: str
    size: float


@dataclass(slots=True)
class SyntheticArticle:
    """Gabarito de um artigo sintético (vai para o manifest)."""

    arquivo: str
    title: str
    has_limitations: bool
    header: str
    heading_style: str
    n_pages: int = 0
    hyphenated_words: list[str] = field(default_factory=list)
    """Palavras quebradas com hífen tipográfico (devem sair **juntas** da limpeza)."""
    compound_breaks: list[str] = field(default_factory=list)
    """Compostos quebrados no hífen legítimo (devem sair **com** hífen da limpeza)."""
    fields: dict[str, str] = field(default_factory=dict)


_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
HEADING_STYLES = ("arabic", "arabic_dot", "roman_upper")


def _heading(n: int, name: str, style: str) -> str:
    if style == "arabic":
        return f"{n} {name}"
    if style == "arabic_dot":
        return f"{n}. {name}"
    return f"{_ROMAN[n - 1]}. {name.upper()}"


def _filler(rng: random.Random, k: int) -> str:
    return " ".join(rng.sample(FILLER_SENTENCES, k))


def build_blocks(
    topic: dict[str, str],
    *,
    has_limitations: bool,
    style: str,
    n_related: int,
    n_discussion: int,
    rng_seed: int,
) -> list[Block]:
    """Monta a sequência de blocos do artigo (sem layout). Determinístico."""
    rng = random.Random(rng_seed)
    blocks = [
        Block("title", topic["title"]),
        Block("authors", "A. Researcher, B. Scientist and C. Analyst · Institute of Data Studies"),
        Block("heading", "Abstract"),
        Block(
            "para",
            f"This paper addresses the problem of {topic['problem']}. {topic['method']} "
            f"Performance is measured by the {topic['metric']}. {topic['result']}",
        ),
    ]
    sections: list[tuple[str, list[str]]] = [
        (
            "Introduction",
            [
                f"We study the problem of {topic['problem']}. {_filler(rng, 2)} "
                "Our contribution is a reproducible pipeline whose evaluation methodology is "
                "described in detail.",
            ],
        ),
        ("Related Work", [_filler(rng, 4) for _ in range(n_related)]),
        ("Data", [f"{topic['data']} {_filler(rng, 2)}"]),
        ("Methods", [f"{topic['method']} {_filler(rng, 3)}"]),
        (
            "Results",
            [
                f"We report the {topic['metric']}. {topic['result']} {_filler(rng, 2)}",
            ],
        ),
    ]
    if has_limitations:
        sections.append(("Limitations", [f"{LIMITATIONS_FILLER} {topic['limitation']}"]))
    sections.append(("Discussion", [_filler(rng, 4) for _ in range(n_discussion)]))
    sections.append(
        (
            "Conclusion",
            [
                f"We presented an approach to {topic['problem']}. "
                "Future work will extend the evaluation to additional settings.",
            ],
        )
    )
    n = 0
    for name, paras in sections:
        if not paras:
            continue
        n += 1
        blocks.append(Block("heading", _heading(n, name, style)))
        blocks.extend(Block("para", p) for p in paras)
    return blocks


# --------------------------------------------------------------------------- layout


def _width(text: str, font: str = BODY_FONT, size: float = BODY_SIZE) -> float:
    return float(stringWidth(text, font, size))


def _split_word(word: str, avail: float) -> tuple[str, str, bool] | None:
    """Tenta quebrar ``word`` para caber em ``avail`` pontos. Devolve (início, resto, composto).

    Composto (tem hífen): quebra no último hífen legítimo que cabe. Palavra simples longa: quebra
    tipográfica com ``-`` deixando ao menos 4 letras antes e 3 depois, e evitando que o início
    seja uma palavra que abre compostos (senão a limpeza, corretamente, manteria o hífen).
    """
    if "-" in word:
        parts = word.split("-")
        for i in range(len(parts) - 1, 0, -1):
            head = "-".join(parts[:i]) + "-"
            if _width(head) <= avail:
                return head, "-".join(parts[i:]), True
        return None
    core = word.rstrip(".,;:")
    if len(core) < MIN_HYPHEN_WORD or not core.isalpha():
        return None
    for k in range(len(core) - 3, 3, -1):
        head = core[:k]
        if head.lower() in COMPOUND_PREFIXES:
            continue
        if _width(head + "-") <= avail:
            return head + "-", word[k:], False
    return None


def layout(blocks: list[Block], article: SyntheticArticle | None = None) -> list[list[DrawOp]]:
    """Distribui os blocos em páginas, quebrando linhas (e palavras) manualmente."""
    pages: list[list[DrawOp]] = [[]]
    y = PAGE_H - MARGIN_TOP

    def new_page() -> None:
        nonlocal y
        pages.append([])
        y = PAGE_H - MARGIN_TOP

    for i, block in enumerate(blocks):
        if block.kind == "title":
            pages[-1].append(
                DrawOp(PAGE_W / 2, y, block.text, TITLE_FONT, TITLE_SIZE)  # centralizado
            )
            y -= TITLE_SIZE + 10
            continue
        if block.kind == "authors":
            pages[-1].append(DrawOp(PAGE_W / 2, y, block.text, "Times-Italic", 10.5))
            y -= 28
            continue
        if block.kind == "heading":
            # mantém o cabeçalho junto de pelo menos duas linhas do parágrafo seguinte
            if y - HEAD_SIZE - 3 * BODY_LEADING < MARGIN_BOTTOM:
                new_page()
            pages[-1].append(DrawOp(MARGIN_X, y, block.text, HEAD_FONT, HEAD_SIZE))
            y -= HEAD_SIZE + 8
            continue
        # parágrafo: quebra de linha manual, com hifenização forçada
        words = block.text.split()
        line = ""
        j = 0
        while j < len(words):
            word = words[j]
            candidate = f"{line} {word}" if line else word
            if _width(candidate) <= TEXT_W:
                line = candidate
                j += 1
                continue
            avail = TEXT_W - _width(f"{line} " if line else "")
            split = _split_word(word, avail) if line else None
            if split is not None:
                head, rest, compound = split
                line = f"{line} {head}"
                words[j] = rest
                if article is not None:
                    target = article.compound_breaks if compound else article.hyphenated_words
                    target.append(word.rstrip(".,;:"))
            if y < MARGIN_BOTTOM:
                new_page()
            pages[-1].append(DrawOp(MARGIN_X, y, line, BODY_FONT, BODY_SIZE))
            y -= BODY_LEADING
            line = ""
        if line:
            if y < MARGIN_BOTTOM:
                new_page()
            pages[-1].append(DrawOp(MARGIN_X, y, line, BODY_FONT, BODY_SIZE))
            y -= BODY_LEADING
        if i < len(blocks) - 1:
            y -= PARA_GAP
    return pages


def render_pdf(path: Path, pages: list[list[DrawOp]], header: str, title: str) -> None:
    """Desenha as páginas com cabeçalho repetido e número de página no rodapé."""
    c = Canvas(str(path), pagesize=letter, invariant=True)
    c.setTitle(title)
    c.setAuthor("Synthetic Generator")
    for n, ops in enumerate(pages, start=1):
        c.setFont("Helvetica", 8)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 40, header)
        for op in ops:
            c.setFont(op.font, op.size)
            if op.font in (TITLE_FONT, "Times-Italic") and op.x == PAGE_W / 2:
                c.drawCentredString(op.x, op.y, op.text)
            else:
                c.drawString(op.x, op.y, op.text)
        c.setFont("Helvetica", 9)
        c.drawCentredString(PAGE_W / 2, 40, str(n))
        c.showPage()
    c.save()


# --------------------------------------------------------------------------- API pública


def write_synthetic_article(out_dir: Path, index: int, seed: int = 42) -> SyntheticArticle:
    """Gera ``artigo_<index:02d>.pdf`` em ``out_dir`` e devolve o gabarito do artigo.

    O número de páginas-alvo (3 a 5) é sorteado com a semente; o corpo cresce com parágrafos
    de "Related Work"/"Discussion" até atingir o alvo.
    """
    rng = random.Random(f"{seed}:{index}")
    topic = TOPICS[(index - 1 + seed) % len(TOPICS)]
    has_limitations = index % 2 == 1
    style = HEADING_STYLES[(index - 1) % len(HEADING_STYLES)]
    target_pages = rng.randint(3, 5)
    header = f"Synthetic Journal of Applied Data Science · Vol. {10 + index} · 2026"
    arquivo = f"artigo_{index:02d}.pdf"

    n_related, n_discussion = 1, 1
    block_seed = rng.randint(0, 2**31)
    while True:
        blocks = build_blocks(
            topic,
            has_limitations=has_limitations,
            style=style,
            n_related=n_related,
            n_discussion=n_discussion,
            rng_seed=block_seed,
        )
        pages = layout(blocks)
        if len(pages) >= target_pages or n_related + n_discussion > 60:
            break
        if n_related <= n_discussion:
            n_related += 1
        else:
            n_discussion += 1

    article = SyntheticArticle(
        arquivo=arquivo,
        title=topic["title"],
        has_limitations=has_limitations,
        header=header,
        heading_style=style,
        fields={
            "problema": topic["problem"],
            "dados": topic["data"],
            "metodo": topic["method"],
            "metrica": topic["metric"],
            "limitacao": topic["limitation"] if has_limitations else "",
        },
    )
    pages = layout(blocks, article)
    article.n_pages = len(pages)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_pdf(out_dir / arquivo, pages, header, topic["title"])
    return article


def generate_synthetic_corpus(out_dir: Path, n: int = 5, seed: int = 42) -> list[SyntheticArticle]:
    """Gera ``n`` artigos e grava ``synthetic_manifest.json`` (gabarito) em ``out_dir``."""
    out_dir = Path(out_dir)
    articles = [write_synthetic_article(out_dir, i, seed) for i in range(1, n + 1)]
    manifest = {"seed": seed, "n": n, "articles": [asdict(a) for a in articles]}
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return articles


def make_synthetic_corpus(out_dir: Path, n: int = 5, seed: int = 42) -> list[Path]:
    """Gera ``n`` artigos sintéticos em ``out_dir`` e devolve os caminhos, ordenados."""
    out_dir = Path(out_dir)
    return [out_dir / a.arquivo for a in generate_synthetic_corpus(out_dir, n, seed)]
