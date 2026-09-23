"""Fixtures compartilhadas. Nenhum teste aqui precisa de rede, GPU ou PDF real."""

from __future__ import annotations

import pytest

from ficha.llm.fake import FakeBehavior, FakeLLM
from ficha.types import Chunk, Context, Document, GenerationParams, Page

ABSTRACT = (
    "Abstract. We study the problem of forecasting hospital readmission within thirty days "
    "of discharge using routinely collected electronic health records. Existing approaches "
    "rely on hand-crafted risk scores that generalize poorly across institutions."
)
INTRO = (
    "1 Introduction. Hospital readmission is a costly outcome and a widely used quality "
    "indicator. We propose a gradient boosted model trained on structured records and free "
    "text notes. Our contribution is a pipeline that is reproducible and inexpensive to deploy."
)
DATA = (
    "2 Data. We use the MIMIC-IV database, covering 2008 to 2019 and comprising 431,231 "
    "admissions from a single tertiary hospital in Boston. Records with missing discharge "
    "summaries were excluded, leaving 380,112 admissions."
)
METHODS = (
    "3 Methods. Structured features are combined with TF-IDF representations of discharge "
    "notes and fed to a LightGBM classifier. Hyperparameters were tuned with five-fold "
    "cross-validation on the training split."
)
RESULTS = (
    "4 Results. The model reaches an AUROC of 0.81 and an AUPRC of 0.34 on the held-out "
    "test set, outperforming the LACE index baseline by 0.09 AUROC."
)
LIMITATIONS = (
    "5 Limitations. Our study has limitations. Data come from a single institution, so "
    "external validity is unknown, and the free text notes were written in English only."
)
CONCLUSION = (
    "6 Conclusion. We presented a practical readmission model. Future work will address "
    "multi-site validation."
)


def make_document(arquivo: str = "artigo_01.pdf", with_limitations: bool = True) -> Document:
    """Artigo sintético em inglês, com seções reconhecíveis e paginação."""
    pages = [
        Page(1, f"{ABSTRACT}\n\n{INTRO}"),
        Page(2, f"{DATA}\n\n{METHODS}"),
        Page(3, RESULTS + ("\n\n" + LIMITATIONS if with_limitations else "")),
        Page(4, CONCLUSION),
    ]
    return Document(arquivo=arquivo, pages=tuple(pages), metadata={"synthetic": True})


@pytest.fixture
def doc() -> Document:
    return make_document()


@pytest.fixture
def doc_sem_limitacao() -> Document:
    return make_document("artigo_02.pdf", with_limitations=False)


@pytest.fixture
def docs(doc: Document, doc_sem_limitacao: Document) -> list[Document]:
    return [doc, doc_sem_limitacao]


def make_context(doc: Document, strategy: str = "first_pages", n_pages: int = 3) -> Context:
    chunks = tuple(
        Chunk(arquivo=doc.arquivo, page=p.number, text=p.text, start=0, end=len(p.text))
        for p in doc.pages[:n_pages]
    )
    return Context(arquivo=doc.arquivo, strategy=strategy, chunks=chunks, params={"n": n_pages})


@pytest.fixture
def context(doc: Document) -> Context:
    return make_context(doc)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Cliente falso perfeito: JSON válido, trecho fiel, null quando devido, estável."""
    return FakeLLM()


@pytest.fixture
def fake_llm_ruim() -> FakeLLM:
    """Cliente falso com os erros que a auditoria precisa pegar."""
    return FakeLLM(
        behavior=FakeBehavior(
            broken_json_rate=0.3,
            invent_trecho_rate=0.3,
            invent_limitacao_rate=0.5,
            unstable_rate=0.3,
            truncate_rate=0.1,
        )
    )


@pytest.fixture
def params() -> GenerationParams:
    return GenerationParams(temperature=0.0, seed=42, max_new_tokens=512)
