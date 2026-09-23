"""Esquema da ficha (Seção 3 do enunciado).

Duas classes, de propósito:

- :class:`FichaExtraida` é **exatamente o que o modelo devolve**. Não contém ``arquivo``
  (nós já sabemos) nem ``confianca`` (que é atribuída pela auditoria, nunca pelo modelo —
  ver Seção 4.4). ``extra="forbid"`` faz JSON com campos a mais falhar na validação, o que
  conta como saída inválida na comparação de prompts.
- :class:`Ficha` é a linha final da tabela: ``arquivo`` + campos extraídos + ``confianca``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Confianca(StrEnum):
    """Nível de confiança atribuído pela auditoria segundo regra declarada."""

    ALTA = "alta"
    MEDIA = "media"
    BAIXA = "baixa"


class Evidencia(BaseModel):
    """Trecho literal do artigo que sustenta a ficha, e a página onde ele está."""

    model_config = ConfigDict(extra="forbid")

    trecho: str = Field(
        ...,
        min_length=20,
        description="Texto literal do artigo que sustenta os campos da ficha.",
    )
    pagina: int = Field(..., ge=1, description="Página (1-based) onde o trecho aparece.")


class FichaExtraida(BaseModel):
    """Saída estruturada esperada do modelo. Ver docstring do módulo."""

    model_config = ConfigDict(extra="forbid")

    problema: str = Field(
        ..., min_length=1, description="O que o trabalho tenta resolver, em uma frase."
    )
    dados: str = Field(
        ..., min_length=1, description="Que dados usa (fonte, período, volume, se informado)."
    )
    metodo: str = Field(..., min_length=1, description="A abordagem principal.")
    metrica: str = Field(..., min_length=1, description="Como o resultado é medido.")
    limitacao: str | None = Field(
        None,
        description="Limitação que os AUTORES declaram. null se o artigo não declara nenhuma.",
    )
    evidencia: Evidencia

    @field_validator("limitacao", mode="before")
    @classmethod
    def _normaliza_limitacao(cls, v: Any) -> Any:
        """Trata sentinelas textuais de ausência como ``None``.

        Modelos pequenos às vezes escrevem "null", "N/A" ou "" em vez do literal JSON ``null``.
        Aceitar essas formas evita descartar uma ficha correta por um detalhe de sintaxe —
        mas isso é registrado no parse como ``repaired``, não como ``ok`` (ver ``extract.parser``).
        """
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "na",
            "não declarada",
            "nao declarada",
        }:
            return None
        return v

    @classmethod
    def json_schema_for_prompt(cls) -> dict[str, Any]:
        """Schema JSON enxuto para embutir no prompt.

        Remove os ``title`` gerados pelo pydantic (em todos os níveis), a descrição interna
        da classe, e torna ``limitacao`` explicitamente obrigatória (o modelo tem de escrever
        a chave, com ``null`` quando não houver limitação declarada — ausência da chave não
        é abstenção, é omissão).
        """
        schema = cls.model_json_schema()

        def _strip(node: Any) -> Any:
            if isinstance(node, dict):
                node.pop("title", None)
                return {k: _strip(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_strip(v) for v in node]
            return node

        schema = dict(_strip(schema))
        schema.pop("description", None)
        required = list(schema.get("required", []))
        if "limitacao" not in required:
            required.append("limitacao")
        schema["required"] = required
        return schema


class Ficha(BaseModel):
    """Linha final da tabela comparativa: exatamente os campos da Seção 3."""

    model_config = ConfigDict(extra="forbid")

    arquivo: str
    problema: str
    dados: str
    metodo: str
    metrica: str
    limitacao: str | None
    evidencia: Evidencia
    confianca: Confianca | None = Field(
        None,
        description="Atribuída pela auditoria (Seção 4.4). None enquanto não auditada.",
    )

    @classmethod
    def from_extraida(
        cls, arquivo: str, extraida: FichaExtraida, confianca: Confianca | None = None
    ) -> Ficha:
        """Monta a ficha final a partir da saída do modelo."""
        return cls(arquivo=arquivo, confianca=confianca, **extraida.model_dump())

    def to_row(self) -> dict[str, Any]:
        """Achata a ficha para uma linha de tabela (CSV/XLSX)."""
        return {
            "arquivo": self.arquivo,
            "problema": self.problema,
            "dados": self.dados,
            "metodo": self.metodo,
            "metrica": self.metrica,
            "limitacao": self.limitacao,
            "evidencia_trecho": self.evidencia.trecho,
            "evidencia_pagina": self.evidencia.pagina,
            "confianca": self.confianca.value if self.confianca else None,
        }


CAMPOS_FICHA: tuple[str, ...] = ("problema", "dados", "metodo", "metrica", "limitacao", "evidencia")
"""Campos que a auditoria compara entre execuções (na ordem da Seção 3)."""
