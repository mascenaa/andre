"""Fábrica de clientes: escolhe a implementação por ``Settings.model_backend``.

Falha cedo e com mensagem acionável quando falta o extra (``pip install "ficha[local]"`` /
``"ficha[api]"``) ou a chave (variável de ambiente ou Secrets do Colab — nunca no código).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from pydantic import SecretStr

from ficha.config import Settings
from ficha.llm.fake import FakeBehavior, FakeLLM
from ficha.types import LLMClient


class ClientConfigError(RuntimeError):
    """Configuração do cliente de modelo incompleta (extra não instalado ou chave ausente)."""


def _is_installed(module: str) -> bool:
    """O módulo pode ser importado? (sem importá-lo — ``torch`` leva segundos para carregar)."""
    return importlib.util.find_spec(module) is not None


def _require_modules(backend: str, extra: str, *modules: str) -> None:
    missing = [m for m in modules if not _is_installed(m)]
    if missing:
        raise ClientConfigError(
            f"O backend {backend!r} precisa de {', '.join(missing)}, que não está instalado. "
            f'Instale com: pip install "ficha[{extra}]"'
        )


def _require_key(backend: str, key: SecretStr | None, env_var: str) -> SecretStr:
    if key is None or not key.get_secret_value().strip():
        raise ClientConfigError(
            f"O backend {backend!r} precisa da chave {env_var}. Defina-a como variável de "
            "ambiente (ou nos Secrets do Colab). Nunca escreva a chave no código ou no notebook."
        )
    return key


def build_client(settings: Settings, **overrides: Any) -> LLMClient:
    """Instancia o cliente de modelo descrito por ``settings``.

    ``overrides`` sobrescreve campos pontualmente sem recriar ``Settings``: ``backend``,
    ``model_name``, ``quantize_4bit``, ``max_new_tokens``, ``api_key``, ``base_url`` e, só para o
    ``fake``, ``behavior`` (:class:`~ficha.llm.fake.FakeBehavior`) e ``name``.
    """
    backend = overrides.pop("backend", settings.model_backend)
    model_name: str = overrides.pop("model_name", settings.model_name)
    max_new_tokens: int = overrides.pop("max_new_tokens", settings.max_new_tokens)

    if backend == "fake":
        behavior: FakeBehavior = overrides.pop("behavior", FakeBehavior())
        name: str | None = overrides.pop("name", None)
        _reject_unknown(overrides)
        return FakeLLM(behavior=behavior) if name is None else FakeLLM(behavior=behavior, name=name)

    if backend == "qwen_local":
        _require_modules(backend, "local", "torch", "transformers", "accelerate")
        from ficha.llm.qwen_local import QwenLocalClient

        quantize: bool = overrides.pop("quantize_4bit", settings.quantize_4bit)
        if quantize:
            _require_modules(backend, "local", "bitsandbytes")
        device: str | None = overrides.pop("device", None)
        _reject_unknown(overrides)
        return QwenLocalClient(
            model_name=model_name,
            quantize_4bit=quantize,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    if backend == "anthropic":
        _require_modules(backend, "api", "anthropic")
        from ficha.llm.anthropic_client import AnthropicClient

        key = _require_key(
            backend, overrides.pop("api_key", settings.anthropic_api_key), "ANTHROPIC_API_KEY"
        )
        _reject_unknown(overrides)
        return AnthropicClient(model_name=model_name, api_key=key, max_new_tokens=max_new_tokens)

    if backend == "openai_compat":
        _require_modules(backend, "api", "openai")
        from ficha.llm.openai_compat import OpenAICompatClient

        base_url: str | None = overrides.pop("base_url", settings.openai_base_url)
        api_key: SecretStr | None = overrides.pop("api_key", settings.openai_api_key)
        if base_url is None:
            # Sem base_url o destino é a própria OpenAI, que exige chave.
            api_key = _require_key(backend, api_key, "OPENAI_API_KEY")
        _reject_unknown(overrides)
        return OpenAICompatClient(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_new_tokens=max_new_tokens,
        )

    raise ClientConfigError(
        f"Backend desconhecido: {backend!r}. Use fake | qwen_local | anthropic | openai_compat."
    )


def _reject_unknown(overrides: dict[str, Any]) -> None:
    if overrides:
        raise TypeError(f"Sobrescritas não reconhecidas para este backend: {sorted(overrides)}")
