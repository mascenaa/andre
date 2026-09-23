"""Cliente por API da Anthropic (Messages API) — alternativa declarada da Seção 4.3.

- O papel vai no parâmetro ``system=`` (não como mensagem), como a API espera.
- ``Usage`` vem de ``response.usage`` — tokens **cobrados**, base exata do custo (4.5).
- A API não tem semente: com ``temperature=0`` a saída é quase determinística, mas não há
  garantia. Isso é declarado; a verificação de estabilidade (4.4b) mede o efeito real.
- Só ``temperature`` é enviado (não ``top_p``): modelos recentes recusam os dois juntos.
- A chave é :class:`pydantic.SecretStr` e nunca aparece em ``repr``, logs ou ``Completion.raw``.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import SecretStr

from ficha.llm.tokens import approx_token_count
from ficha.types import Completion, GenerationParams, Usage


class AnthropicClient:
    """Implementa :class:`ficha.types.LLMClient` sobre o SDK ``anthropic`` (import preguiçoso)."""

    def __init__(self, model_name: str, api_key: SecretStr, max_new_tokens: int = 1024) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError('O backend anthropic exige o SDK: pip install "ficha[api]"') from exc
        self._model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._client: Any = anthropic.Anthropic(api_key=api_key.get_secret_value())

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, system: str | None, user: str, params: GenerationParams) -> Completion:
        """Uma chamada à Messages API; o texto de todos os blocos ``text`` é concatenado."""
        max_tokens = min(params.max_new_tokens, self.max_new_tokens)
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": max_tokens,
            "temperature": params.temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system is not None:
            kwargs["system"] = system

        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        text = "".join(
            getattr(block, "text", "")
            for block in resp.content
            if getattr(block, "type", "text") == "text"
        )
        stop = getattr(resp, "stop_reason", None)
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=int(resp.usage.input_tokens),
                output_tokens=int(resp.usage.output_tokens),
            ),
            model=str(getattr(resp, "model", self._model_name)),
            latency_s=latency,
            params=params,
            raw={
                "id": getattr(resp, "id", None),
                "stop_reason": stop,
                "finish_reason": "length" if stop == "max_tokens" else "stop",
            },
        )

    def count_tokens(self, text: str) -> int:
        """Contagem pelo endpoint ``messages.count_tokens``; cai na aproximação declarada.

        O endpoint conta a mensagem inteira (com a moldura do chat), então o número é
        levemente maior que o do texto puro — adequado para estimar custo.
        """
        try:
            resp = self._client.messages.count_tokens(
                model=self._model_name, messages=[{"role": "user", "content": text}]
            )
            return int(resp.input_tokens)
        except Exception:
            return approx_token_count(text)

    def __repr__(self) -> str:
        return f"AnthropicClient(model_name={self._model_name!r}, api_key=**********)"
