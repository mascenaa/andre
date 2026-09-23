from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ficha.types import GenerationParams, LLMClient

SECRET = "sk-teste-NAO-VAZAR"


# ------------------------------------------------------------------ Anthropic


class _FakeMessages:
    def __init__(self, fail_count=False):
        self.calls: list[dict] = []
        self.fail_count = fail_count

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="msg_1",
            model=kwargs["model"],
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text='{"a": '),
                SimpleNamespace(type="text", text="1}"),
            ],
            usage=SimpleNamespace(input_tokens=120, output_tokens=30),
        )

    def count_tokens(self, **kwargs):
        if self.fail_count:
            raise RuntimeError("sem endpoint")
        return SimpleNamespace(input_tokens=42)


def _install_anthropic(monkeypatch, fail_count=False):
    created: dict = {}

    class FakeAnthropic:
        def __init__(self, api_key):
            created["api_key"] = api_key
            self.messages = _FakeMessages(fail_count)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    return created


def test_anthropic_generate(monkeypatch):
    created = _install_anthropic(monkeypatch)
    from ficha.llm.anthropic_client import AnthropicClient

    client = AnthropicClient("claude-x", SecretStr(SECRET), max_new_tokens=800)
    assert isinstance(client, LLMClient)
    assert created["api_key"] == SECRET

    comp = client.generate("papel", "pergunta", GenerationParams(temperature=0.0))
    call = client._client.messages.calls[0]
    assert call["system"] == "papel"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 800  # teto do cliente < 1024 dos params
    assert "top_p" not in call
    assert call["messages"] == [{"role": "user", "content": "pergunta"}]
    assert comp.text == '{"a": 1}'
    assert (comp.usage.input_tokens, comp.usage.output_tokens) == (120, 30)
    assert comp.raw["finish_reason"] == "stop"
    assert SECRET not in repr(client)
    assert SECRET not in str(comp.raw)


def test_anthropic_sem_papel_omite_system(monkeypatch):
    _install_anthropic(monkeypatch)
    from ficha.llm.anthropic_client import AnthropicClient

    client = AnthropicClient("claude-x", SecretStr(SECRET))
    client.generate(None, "pergunta", GenerationParams())
    assert "system" not in client._client.messages.calls[0]


def test_anthropic_count_tokens_com_fallback(monkeypatch):
    _install_anthropic(monkeypatch)
    from ficha.llm.anthropic_client import AnthropicClient

    assert AnthropicClient("claude-x", SecretStr(SECRET)).count_tokens("x" * 400) == 42

    _install_anthropic(monkeypatch, fail_count=True)
    assert AnthropicClient("claude-x", SecretStr(SECRET)).count_tokens("x" * 400) == 100


# ------------------------------------------------------------------ OpenAI-compatível


def _install_openai(monkeypatch, reject_seed=False, with_usage=True):
    state: dict = {"calls": []}

    class Completions:
        def create(self, **kwargs):
            state["calls"].append(kwargs)
            if reject_seed and "seed" in kwargs:
                raise ValueError("Unsupported parameter: 'seed'")
            usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10) if with_usage else None
            return SimpleNamespace(
                id="c1",
                model=kwargs["model"],
                system_fingerprint="fp",
                usage=usage,
                choices=[
                    SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))
                ],
            )

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            state["api_key"], state["base_url"] = api_key, base_url
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return state


def test_openai_compat_envia_seed(monkeypatch):
    state = _install_openai(monkeypatch)
    from ficha.llm.openai_compat import OpenAICompatClient

    client = OpenAICompatClient("qwen2.5:3b", SecretStr(SECRET), "http://localhost:11434/v1")
    comp = client.generate("papel", "pergunta", GenerationParams(seed=42, temperature=0.0))
    call = state["calls"][0]
    assert call["seed"] == 42
    assert call["messages"][0] == {"role": "system", "content": "papel"}
    assert (comp.usage.input_tokens, comp.usage.output_tokens) == (50, 10)
    assert comp.raw["seed_sent"] is True
    assert state["api_key"] == SECRET
    assert SECRET not in repr(client)


def test_openai_compat_refaz_sem_seed_quando_recusado(monkeypatch):
    state = _install_openai(monkeypatch, reject_seed=True)
    from ficha.llm.openai_compat import OpenAICompatClient

    client = OpenAICompatClient("m", None, "http://x/v1")
    comp = client.generate(None, "pergunta", GenerationParams(seed=42))
    assert len(state["calls"]) == 2
    assert "seed" not in state["calls"][1]
    assert comp.raw["seed_sent"] is False
    assert client.supports_seed is False
    assert state["calls"][1]["messages"] == [{"role": "user", "content": "pergunta"}]


def test_openai_compat_sem_usage_aproxima(monkeypatch):
    _install_openai(monkeypatch, with_usage=False)
    from ficha.llm.openai_compat import OpenAICompatClient

    comp = OpenAICompatClient("m", None, "http://x/v1").generate(
        None, "x" * 400, GenerationParams()
    )
    assert comp.usage.input_tokens == 100
    assert comp.raw["usage_estimated"] is True


def test_sdk_ausente_gera_erro_claro(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import passa a falhar
    from ficha.llm.anthropic_client import AnthropicClient

    with pytest.raises(ImportError, match=r"ficha\[api\]"):
        AnthropicClient("claude-x", SecretStr(SECRET))
