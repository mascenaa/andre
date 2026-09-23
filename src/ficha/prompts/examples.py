"""Exemplos few-shot (Seção 4.2): fichas preenchidas corretamente, incluindo um caso ``null``.

Regras seguidas pelos exemplos — e verificadas em ``tests/test_prompts_examples.py``:

- A entrada é produzida por :meth:`ficha.types.Context.render`, exatamente no formato que o
  modelo verá nos artigos reais (cada trecho precedido de ``[p. N]``).
- ``evidencia.trecho`` é **cópia literal** de um trecho da entrada, e ``evidencia.pagina`` é o
  número do marcador ``[p. N]`` sob o qual ele aparece.
- Exemplo 1: os autores **declaram** uma limitação → ``limitacao`` a resume.
- Exemplo 2: o artigo **não declara** limitação (fala de "future work", que não é limitação)
  → ``limitacao`` é ``null``; e o período dos dados não é informado → "não informado".
- Domínios diferentes dos artigos de teste, para o modelo não copiar conteúdo do exemplo.
"""

from __future__ import annotations

from dataclasses import dataclass

from ficha.schema import Evidencia, FichaExtraida
from ficha.types import Chunk, Context


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """Um par (entrada, saída correta) mostrado ao modelo."""

    name: str
    context_text: str
    """Texto no formato de :meth:`Context.render` (com marcadores ``[p. N]``)."""
    output: FichaExtraida
    raciocinio: str
    """Raciocínio curto usado só na variante com cadeia de raciocínio."""


def _render(arquivo: str, pages: list[tuple[int, str]]) -> str:
    chunks = tuple(
        Chunk(arquivo=arquivo, page=n, text=text, start=0, end=len(text)) for n, text in pages
    )
    return Context(arquivo=arquivo, strategy="few_shot", chunks=chunks).render()


# ------------------------------------------------------------------ exemplo 1: com limitação

_EX1_P1 = (
    "Abstract. Early detection of credit card fraud remains difficult because fraudulent "
    "transactions are rare and fraud patterns drift over time. We propose a sequence model "
    "that represents each cardholder's recent transactions and flags anomalous purchases in "
    "real time."
)
_EX1_P2 = (
    "3 Experimental setup. We use 2.4 million anonymised transactions from a European "
    "issuer, collected between January 2021 and June 2022, of which 0.17% are labelled as "
    "fraud. A two-layer LSTM is trained on sequences of the last 30 transactions per card. "
    "Performance is reported as area under the precision-recall curve (AUPRC) on the final "
    "three months, which are held out in time."
)
_EX1_P3 = (
    "6 Discussion. The LSTM improves AUPRC from 0.41 to 0.58 over a gradient boosting "
    "baseline. A limitation of this study is that labels come from chargebacks, so frauds "
    "never disputed by customers are missing from the ground truth."
)

EXAMPLE_WITH_LIMITATION = FewShotExample(
    name="com_limitacao",
    context_text=_render("exemplo_fraude.pdf", [(1, _EX1_P1), (2, _EX1_P2), (4, _EX1_P3)]),
    output=FichaExtraida(
        problema="Detectar fraudes em cartão de crédito em tempo real, apesar da raridade das "
        "fraudes e da mudança de padrão ao longo do tempo.",
        dados="2,4 milhões de transações anonimizadas de um emissor europeu, de jan/2021 a "
        "jun/2022, com 0,17% rotuladas como fraude.",
        metodo="LSTM de duas camadas sobre as últimas 30 transações de cada cartão, comparada "
        "a um baseline de gradient boosting.",
        metrica="Área sob a curva precisão-revocação (AUPRC) nos três meses finais, separados "
        "no tempo.",
        limitacao="Os rótulos vêm de chargebacks, então fraudes nunca contestadas pelos "
        "clientes ficam fora da verdade de referência.",
        evidencia=Evidencia(
            trecho="A limitation of this study is that labels come from chargebacks, so frauds "
            "never disputed by customers are missing from the ground truth.",
            pagina=4,
        ),
    ),
    raciocinio="O problema está no resumo (p. 1); dados, método e métrica estão na seção 3 "
    "(p. 2); a p. 4 traz a frase 'A limitation of this study is...', que é uma limitação "
    "declarada pelos próprios autores, então limitacao não é null.",
)

# ------------------------------------------------------------------ exemplo 2: sem limitação

_EX2_P1 = (
    "Abstract. Retailers need accurate weekly demand forecasts to plan inventory across "
    "hundreds of stores. We introduce a temporal fusion transformer that combines store "
    "attributes, promotions and holiday calendars to forecast sales at the product level."
)
_EX2_P2 = (
    "4 Evaluation. The model is trained on sales records from 312 supermarkets of a Brazilian "
    "chain. Forecast accuracy is measured with the weighted mean absolute percentage error "
    "(WMAPE) over a 13-week horizon. Our model reduces WMAPE by 11% relative to exponential "
    "smoothing. In future work we plan to add weather data to the model."
)

EXAMPLE_WITHOUT_LIMITATION = FewShotExample(
    name="sem_limitacao",
    context_text=_render("exemplo_varejo.pdf", [(1, _EX2_P1), (5, _EX2_P2)]),
    output=FichaExtraida(
        problema="Prever a demanda semanal por produto em centenas de lojas para planejar o "
        "estoque.",
        dados="Registros de vendas de 312 supermercados de uma rede brasileira; período e "
        "volume não informados.",
        metodo="Temporal fusion transformer combinando atributos da loja, promoções e "
        "calendário de feriados.",
        metrica="WMAPE (erro percentual absoluto médio ponderado) em horizonte de 13 semanas, "
        "comparado a suavização exponencial.",
        limitacao=None,
        evidencia=Evidencia(
            trecho="Forecast accuracy is measured with the weighted mean absolute percentage "
            "error (WMAPE) over a 13-week horizon.",
            pagina=5,
        ),
    ),
    raciocinio="Problema e método estão no resumo (p. 1); dados e métrica na seção 4 (p. 5). "
    "O texto não declara nenhuma limitação — 'future work' é plano, não limitação —, então "
    "limitacao é null. O período dos dados não aparece, então dados diz 'não informado'.",
)

FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    EXAMPLE_WITH_LIMITATION,
    EXAMPLE_WITHOUT_LIMITATION,
)
"""Exemplos na ordem em que aparecem no prompt (com limitação, depois ``null``)."""
