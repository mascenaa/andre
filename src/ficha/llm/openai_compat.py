"""Cliente para endpoints compatíveis com a API da OpenAI (Groq, Together, Ollama local...).

- ``seed`` é enviado quando ``params.seed`` existe; se o provedor recusar o parâmetro, a
  chamada é refeita sem ele e o cliente passa a omiti-lo (registrado em ``Completion.raw``).
- ``Usage`` vem de ``response.usage`` quando o provedor informa; senão, aproximação declarada.
- ``count_tokens`` é sempre a aproximação declarada (não há endpoint padrão de contagem).
- A chave é :class:`pydantic.SecretStr` e nunca aparece em ``repr`` nem em logs. Para Ollama,
  que não exige chave, ``api_key=None`` usa um valor fictício.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import SecretStr

from ficha.llm.tokens import approx_token_count
from ficha.types import Completion, GenerationParams, Usage


class OpenAICompatClient:
    """Implementa :class:`ficha.types.LLMClient` sobre o SDK ``openai`` (import preguiçoso)."""

    def __init__(
        self,
        model_name: str,
        api_key: SecretStr | None = None,
        base_url: str | None = None,
        max_new_tokens: int = 1024,
        supports_seed: bool = True,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                'O backend openai_compat exige o SDK: pip install "ficha[api]"'
            ) from exc
        self._model_name = model_name
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.supports_seed = supports_seed
        key = api_key.get_secret_value() if api_key is not None else "sem-chave"
        self._client: Any = openai.OpenAI(api_key=key, base_url=base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _create(self, kwargs: dict[str, Any]) -> Any:
        return self._client.chat.completions.create(**kwargs)

    def generate(self, system: str | None, user: str, params: GenerationParams) -> Completion:
        """Uma chamada a ``chat.completions``; refaz sem ``seed`` se o provedor recusar."""
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": min(params.max_new_tokens, self.max_new_tokens),
        }
        seed_sent = self.supports_seed and params.seed is not None
        if seed_sent:
            kwargs["seed"] = params.seed

        t0 = time.perf_counter()
        try:
            resp = self._create(kwargs)
        except Exception as exc:
            if not seed_sent or "seed" not in str(exc).lower():
                raise
            self.supports_seed = False
            seed_sent = False
            kwargs.pop("seed")
            resp = self._create(kwargs)
        latency = time.perf_counter() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        n_in = getattr(usage, "prompt_tokens", None)
        n_out = getattr(usage, "completion_tokens", None)
        estimated = n_in is None or n_out is None
        if n_in is None:
            n_in = approx_token_count((system or "") + user)
        if n_out is None:
            n_out = approx_token_count(text)
        return Completion(
            text=text,
            usage=Usage(input_tokens=int(n_in), output_tokens=int(n_out)),
            model=str(getattr(resp, "model", None) or self._model_name),
            latency_s=latency,
            params=params,
            raw={
                "id": getattr(resp, "id", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "system_fingerprint": getattr(resp, "system_fingerprint", None),
                "seed_sent": seed_sent,
                "usage_estimated": estimated,
            },
        )

    def count_tokens(self, text: str) -> int:
        """Aproximação declarada (~4 caracteres por token)."""
        return approx_token_count(text)

    def __repr__(self) -> str:
        return (
            f"OpenAICompatClient(model_name={self._model_name!r}, base_url={self.base_url!r}, "
            "api_key=**********)"
        )
