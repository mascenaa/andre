"""Configuração central via variáveis de ambiente / ``.env`` / Secrets do Colab.

Nada de chave em código (Seção 7). Toda escolha que o relatório precisa declarar
(modelo, temperatura, semente, premissa de preço) vive aqui e é impressa pelo notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["fake", "qwen_local", "anthropic", "openai_compat"]


class Settings(BaseSettings):
    """Parâmetros do pipeline. Prefixo ``FICHA_`` nas variáveis de ambiente."""

    model_config = SettingsConfigDict(
        env_prefix="FICHA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- caminhos
    data_dir: Path = Field(default=Path("data"))

    # --- modelo
    model_backend: Backend = "fake"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    quantize_4bit: bool = Field(
        default=False, description="Necessário para o 7B na T4 (16 GB). Ver Seção 4.3."
    )
    max_new_tokens: int = 1024

    # --- reprodutibilidade
    seed: int = 42
    temperature: float = Field(
        default=0.0,
        description=(
            "Temperatura padrão para extração. 0.0 porque a tarefa é de fidelidade, não de "
            "criatividade: queremos a saída mais provável, estável entre execuções. "
            "A Seção 4.4d mede o efeito de subir esse valor."
        ),
    )
    temperature_alt: float = Field(
        default=0.7, description="Temperatura alternativa usada na verificação 4.4d."
    )

    # --- seleção de contexto (Seção 4.1)
    first_pages_n: int = 3
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200
    semantic_top_k: int = 6
    hybrid_max_chars: int | None = Field(
        default=12000,
        description=(
            "Orçamento da estratégia híbrida (semântica + seções finais por palavra-chave). "
            "Maior que o semântico porque a auditoria real mostrou que as limitações declaradas "
            "ficam nas páginas finais e não chegavam ao modelo (ver ADR 0002, revisão)."
        ),
    )
    semantic_max_chars: int | None = Field(
        default=8000,
        description=(
            "Orçamento de caracteres do contexto semântico (~2 mil tokens). Sem teto, "
            "8 consultas × top_k=6 num artigo de 25 páginas chegam a 30–50 mil caracteres, "
            "o que anula a economia que justifica a estratégia (Seções 4.1 e 4.5)."
        ),
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "Multilíngue de propósito: as consultas são escritas em português e os artigos "
            "estão em inglês. Roda em CPU em minutos (FAQ do enunciado)."
        ),
    )

    # --- custo (Seção 4.5) — premissa declarada, visível junto do resultado
    price_input_per_mtok: float = Field(default=3.0, description="USD por 1M tokens de entrada.")
    price_output_per_mtok: float = Field(default=15.0, description="USD por 1M tokens de saída.")
    price_source: str = Field(
        default="Tabela pública Anthropic, Claude Sonnet 4.5, consultada em 2026-09",
        description="De onde veio a premissa de preço. Aparece no relatório.",
    )

    # --- chaves (nunca logadas: SecretStr)
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # --- caminhos derivados
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.runs_dir, self.outputs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def declaracao(self) -> dict[str, object]:
        """Tudo que o relatório precisa declarar, sem segredos."""
        return {
            "modelo": f"{self.model_backend}:{self.model_name}",
            "quantize_4bit": self.quantize_4bit,
            "temperatura": self.temperature,
            "temperatura_alternativa": self.temperature_alt,
            "seed": self.seed,
            "embedding_model": self.embedding_model,
            "premissa_preco_usd_por_mtok": {
                "entrada": self.price_input_per_mtok,
                "saida": self.price_output_per_mtok,
                "fonte": self.price_source,
            },
        }


def get_settings(**overrides: object) -> Settings:
    """Carrega as configurações (com sobrescritas explícitas para testes/notebook)."""
    return Settings(**overrides)  # type: ignore[arg-type]
