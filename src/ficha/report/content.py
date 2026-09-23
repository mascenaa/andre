"""Conteúdo do relatório PDF (Seção 5, entregável 3).

:class:`ReportContent` tem **exatamente** os itens que a Seção 5 exige, cada um num campo
nomeado — assim, esquecer um item é um erro de construção do objeto, e não uma seção que some
silenciosamente do PDF. O mapeamento item do enunciado → campo é:

========================================================  ==================================
Item da Seção 5                                           Campo
========================================================  ==================================
identificação dos integrantes (primeira página)           ``integrantes``
estratégia escolhida em 4.1 e por quê                     ``estrategia_escolhida``,
                                                          ``estrategia_justificativa``
duas versões do prompt, o que muda, o que a comparação    ``prompt_v1``, ``prompt_v2``,
mostrou                                                   ``prompt_diff``, ``prompt_comparacao``,
                                                          ``prompt_conclusao``
qual modelo e por quê (+ temperatura, Seção 4.2/4.3)      ``modelo``, ``modelo_justificativa``,
                                                          ``temperatura_justificativa``
resultados das quatro verificações de 4.4, com números    ``resultados_auditoria``
regra de confiança declarada (Seção 4.4, final)           ``regra_confianca``
fichas não defensáveis e o motivo                         ``fichas_nao_defensaveis``
custo de 4.5 (premissa visível, razão ingênuo/real)       ``custo``
o que faria diferente com mais uma semana                 ``o_que_faria_diferente``
declaração de uso de IA (Seção 7)                         ``declaracao_uso_ia``
========================================================  ==================================

Tabelas podem ser passadas como ``pandas.DataFrame`` ou como ``list[dict]``;
:func:`table_rows` normaliza as duas formas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TableLike = pd.DataFrame | Sequence[Mapping[str, Any]]
"""Tabela aceita pelo relatório: ``DataFrame`` ou lista de dicionários."""


def format_cell(value: Any) -> str:
    """Formata um valor de célula para o PDF (números com vírgula decimal, None vazio)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "sim" if value else "não"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer() and abs(value) >= 1:
            return f"{int(value):,}".replace(",", ".")
        text = f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1 else f"{value:,.2f}"
        # Estilo pt-BR: vírgula decimal, ponto de milhar.
        return text.replace(",", "_").replace(".", ",").replace("_", ".")
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "; ".join(format_cell(v) for v in value)
    return str(value)


def table_rows(table: TableLike) -> tuple[list[str], list[list[str]]]:
    """Normaliza uma tabela para ``(cabeçalho, linhas)`` de strings já formatadas.

    Um ``DataFrame`` com índice nomeado (ou não numérico) tem o índice promovido a coluna,
    para que rótulos como o nome da métrica não se percam.
    """
    if isinstance(table, pd.DataFrame):
        frame = table
        if frame.index.name is not None or not pd.api.types.is_integer_dtype(frame.index):
            frame = frame.reset_index()
        headers = [str(c) for c in frame.columns]
        rows = [[format_cell(v) for v in rec] for rec in frame.itertuples(index=False)]
        return headers, rows
    headers_list: list[str] = []
    for rec in table:
        for key in rec:
            if str(key) not in headers_list:
                headers_list.append(str(key))
    rows = [[format_cell(rec.get(h)) for h in headers_list] for rec in table]
    return headers_list, rows


@dataclass(slots=True)
class PromptVersion:
    """Uma das duas versões do prompt comparadas na Seção 4.2."""

    nome: str
    """Identificador da variante (ex.: ``v_full``)."""
    features: str
    """Técnicas ligadas/desligadas (ex.: saída de ``PromptFeatures.describe()``)."""
    descricao: str = ""
    """Uma frase sobre o que caracteriza esta versão."""


@dataclass(slots=True)
class AuditBlock:
    """Resultado de uma verificação da Seção 4.4: números + conclusão em uma frase."""

    titulo: str
    numeros: TableLike | Mapping[str, Any]
    """Tabela compacta ou dicionário ``métrica → valor``."""
    conclusao: str


@dataclass(slots=True)
class AuditResults:
    """As quatro verificações mínimas da Seção 4.4 (obrigatórias) e a de vazamento (opcional)."""

    fidelidade: AuditBlock
    estabilidade: AuditBlock
    entrada: AuditBlock
    temperatura: AuditBlock
    vazamento: AuditBlock | None = None
    """Opcional: e) campos copiados dos exemplos few-shot (além do mínimo do enunciado)."""

    def blocks(self) -> list[AuditBlock]:
        """Os blocos na ordem a)–d) do enunciado, mais e) quando houver."""
        base = [self.fidelidade, self.estabilidade, self.entrada, self.temperatura]
        return base + ([self.vazamento] if self.vazamento is not None else [])


@dataclass(slots=True)
class CostSection:
    """Custo da Seção 4.5, com a premissa de preço visível junto do resultado."""

    tabela: TableLike
    """Tokens e custo: real (com repetições) vs. ingênuo (artigos inteiros)."""
    premissa: str
    """Premissa de preço declarada (USD/Mtok de entrada e saída + fonte)."""
    razao_ingenuo_real: float
    """Quantas vezes a alternativa ingênua é mais cara que a execução real."""
    comentario: str = ""


@dataclass(slots=True)
class ReportContent:
    """Tudo o que a Seção 5 exige no relatório — um campo por item (ver docstring do módulo)."""

    titulo: str
    integrantes: list[str]
    estrategia_escolhida: str
    estrategia_justificativa: str
    prompt_v1: PromptVersion
    prompt_v2: PromptVersion
    prompt_diff: str
    prompt_comparacao: TableLike
    prompt_conclusao: str
    modelo: str
    modelo_justificativa: str
    temperatura_justificativa: str
    resultados_auditoria: AuditResults
    regra_confianca: str
    fichas_nao_defensaveis: TableLike
    """Tabela com (pelo menos) ``arquivo`` e ``motivos``."""
    custo: CostSection
    o_que_faria_diferente: list[str]
    declaracao_uso_ia: str
    estrategia_tabela: TableLike | None = None
    """Opcional: comparação das estratégias de seleção (páginas, chars, tokens)."""
    estrategia_evidencia: str = ""
    """Opcional: evidência medida sobre a estratégia (ex.: quantas frases de limitação chegam
    ao contexto de cada estratégia). Aparece na seção 1."""
    destaque_auditoria: str = ""
    """Opcional: o caso que a rubrica pede em destaque — uma invenção convincente, como foi
    detectada e por que a detecção funcionou. Parágrafo próprio, em negrito, na seção 4."""
    subtitulo: str = "Computação Cognitiva · Ciência de Dados e Negócios · ESPM · 2026.2 · Prof. André Insardi"
    figuras: list[Path] = field(default_factory=list)
    """Figuras opcionais; só entram no PDF se couberem nas 3 páginas."""

    def __post_init__(self) -> None:
        if not [n for n in self.integrantes if n.strip()]:
            raise ValueError("O relatório precisa identificar os integrantes (Seção 5).")


def sample_content() -> ReportContent:
    """Conteúdo de exemplo, realista no tamanho, para testes e para ``ficha report --exemplo``.

    Os números são ilustrativos; no notebook, cada um vem de um objeto da auditoria.
    """
    return ReportContent(
        titulo="Atividade de Construção I — Ficha comparativa de 19 artigos",
        integrantes=["Ana Souza", "Bruno Conceição", "Carla Ribeiro", "Diego Araújo"],
        estrategia_escolhida="semantic (trechos por similaridade), com keyword como alternativa",
        estrategia_justificativa=(
            "Segmentamos cada artigo em blocos de 1.200 caracteres com sobreposição de 200 "
            "(uma ideia completa cabe no bloco; a sobreposição evita cortar a frase de "
            "evidência) e enviamos os 6 blocos mais próximos de consultas sobre problema, "
            "dados, método, métrica e limitação. As primeiras páginas raramente trazem "
            "limitações declaradas, e a busca por palavra-chave falha quando a seção não se "
            "chama 'Limitations'."
        ),
        estrategia_tabela=[
            {"estrategia": "first_pages", "chars_medios": 11800, "tokens_medios": 2950},
            {"estrategia": "keyword", "chars_medios": 8400, "tokens_medios": 2100},
            {"estrategia": "semantic", "chars_medios": 7200, "tokens_medios": 1800},
        ],
        prompt_v1=PromptVersion(
            "v_full",
            "papel, delimitadores, few-shot (com null), abstenção",
            "Versão final: todas as técnicas exigidas.",
        ),
        prompt_v2=PromptVersion(
            "v_sem_fewshot", "papel, delimitadores, abstenção", "Mesma versão sem os exemplos."
        ),
        prompt_diff="Única diferença: o bloco de exemplos (few-shot) é removido na v2.",
        prompt_comparacao=pd.DataFrame(
            {
                "metrica": ["JSON válido de primeira", "null correto em limitacao", "discordância"],
                "v_full": [0.95, 0.83, 0.21],
                "v_sem_fewshot": [0.79, 0.50, 0.21],
            }
        ).set_index("metrica"),
        prompt_conclusao=(
            "O few-shot elevou a taxa de JSON válido de 79% para 95% e a abstenção correta de "
            "50% para 83%; escolhemos v_full pela medição, não por intuição."
        ),
        modelo="Qwen/Qwen2.5-3B-Instruct (float16, Colab T4)",
        modelo_justificativa=(
            "Cabe na T4 em meia precisão e é o equilíbrio recomendado; o 1,5B erra demais e o "
            "7B exige quantização em 4 bits."
        ),
        temperatura_justificativa=(
            "Temperatura 0,0: extração pede a saída mais provável e estável; 4.4d mede 0,7."
        ),
        resultados_auditoria=AuditResults(
            fidelidade=AuditBlock(
                "a) Fidelidade",
                {"fichas com trecho literal": "16/19", "trechos inventados": 3},
                "Três trechos não existem no texto enviado: são invenções.",
            ),
            estabilidade=AuditBlock(
                "b) Estabilidade",
                [
                    {"campo": "metodo", "taxa_mudanca": 0.16},
                    {"campo": "limitacao", "taxa_mudanca": 0.11},
                    {"campo": "problema", "taxa_mudanca": 0.05},
                ],
                "Com t=0 e seed fixa, 17/19 fichas idênticas; 'metodo' é o campo mais instável.",
            ),
            entrada=AuditBlock(
                "c) Efeito da entrada",
                {"discordância first_pages × semantic": "7/19", "limitacao divergente": 6},
                "first_pages perde a limitação declarada em 6 artigos; defendemos semantic.",
            ),
            temperatura=AuditBlock(
                "d) Efeito da temperatura",
                {"JSON válido t=0,0": "95%", "JSON válido t=0,7": "74%", "subconjunto": 6},
                "t=0,7 piora validade e fidelidade: a escolha de t=0,0 se sustenta.",
            ),
        ),
        regra_confianca=(
            "alta: trecho literal + estável + JSON ok + mesma resposta nas duas entradas; "
            "media: uma falha leve; baixa: trecho inventado, parse falho ou limitação instável."
        ),
        fichas_nao_defensaveis=[
            {"arquivo": "artigo_04.pdf", "motivos": "trecho de evidência não existe no texto"},
            {"arquivo": "artigo_11.pdf", "motivos": "limitação muda entre execuções"},
            {"arquivo": "artigo_15.pdf", "motivos": "JSON falhou; ficha reparada"},
            {"arquivo": "artigo_17.pdf", "motivos": "limitação provavelmente inventada"},
        ],
        custo=CostSection(
            tabela=[
                {"cenario": "real (4 execuções + subconjunto)", "tokens": 171000, "usd": 0.62},
                {"cenario": "ingênuo (artigos inteiros ×4)", "tokens": 1400000, "usd": 4.35},
            ],
            premissa="USD 3,00/Mtok entrada e 15,00/Mtok saída (Claude Sonnet 4.5, 2026-09).",
            razao_ingenuo_real=7.0,
            comentario="O Qwen local não tem custo monetário; a premissa torna o custo comparável.",
        ),
        o_que_faria_diferente=[
            "Comparar o Qwen local com um modelo por API nas mesmas 19 fichas.",
            "Anotar à mão um gabarito de 5 artigos para medir acurácia, não só consistência.",
            "Separar raciocínio e resposta em campos distintos e medir se ajuda.",
        ],
        declaracao_uso_ia=(
            "Usamos assistentes de IA para gerar partes do código e revisar o texto; todas as "
            "decisões e números foram verificados pelo grupo."
        ),
    )
