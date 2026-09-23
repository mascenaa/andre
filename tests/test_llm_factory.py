from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from ficha.config import Settings
from ficha.llm import factory
from ficha.llm.factory import ClientConfigError, build_client
from ficha.llm.fake import FakeBehavior, FakeLLM
from ficha.types import LLMClient


@pytest.fixture
def settings(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def sem_extras(monkeypatch):
    monkeypatch.setattr(factory, "_is_installed", lambda _m: False)


@pytest.fixture
def com_extras(monkeypatch):
    monkeypatch.setattr(factory, "_is_installed", lambda _m: True)


def test_fake(settings):
    client = build_client(settings)
    assert isinstance(client, FakeLLM)
    assert isinstance(client, LLMClient)


def test_fake_com_comportamento(settings):
    behavior = FakeBehavior(broken_json_rate=1.0)
    client = build_client(settings, behavior=behavior, name="fake/ruim")
    assert isinstance(client, FakeLLM)
    assert client.behavior is behavior
    assert client.model_name == "fake/ruim"


def test_sobrescrita_desconhecida_falha(settings):
    with pytest.raises(TypeError, match="nao_existe"):
        build_client(settings, nao_existe=1)


def test_backend_desconhecido(settings):
    with pytest.raises(ClientConfigError, match="desconhecido"):
        build_client(settings, backend="gpt-magico")


@pytest.mark.parametrize(
    ("backend", "extra"),
    [("qwen_local", "local"), ("anthropic", "api"), ("openai_compat", "api")],
)
def test_sem_extras_mensagem_clara(settings, sem_extras, backend, extra):
    with pytest.raises(ClientConfigError, match=rf'pip install "ficha\[{extra}\]"'):
        build_client(settings, backend=backend)


def test_qwen_local_com_extras_nao_carrega_modelo(settings, com_extras):
    client = build_client(settings, backend="qwen_local", quantize_4bit=True)
    from ficha.llm.qwen_local import QwenLocalClient

    assert isinstance(client, QwenLocalClient)
    assert client.quantize_4bit is True
    assert not client.loaded
    assert client.model_name.startswith(settings.model_name)


def test_anthropic_sem_chave(settings, com_extras):
    with pytest.raises(ClientConfigError, match="ANTHROPIC_API_KEY") as exc:
        build_client(settings, backend="anthropic")
    assert "Nunca" in str(exc.value)


def test_anthropic_com_chave_do_ambiente(monkeypatch, com_extras):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-do-ambiente")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=lambda api_key: SimpleNamespace(key=api_key)),
    )
    s = Settings(_env_file=None, model_name="claude-x")  # type: ignore[call-arg]
    client = build_client(s, backend="anthropic")
    assert client.model_name == "claude-x"
    assert "sk-do-ambiente" not in repr(client)


def test_openai_sem_base_url_exige_chave(settings, com_extras):
    with pytest.raises(ClientConfigError, match="OPENAI_API_KEY"):
        build_client(settings, backend="openai_compat")


def test_openai_ollama_sem_chave(settings, com_extras, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=lambda api_key, base_url: SimpleNamespace(base_url=base_url)),
    )
    client = build_client(
        settings, backend="openai_compat", base_url="http://localhost:11434/v1", model_name="q"
    )
    assert client.model_name == "q"


def test_chave_em_override_tambem_vale(settings, com_extras, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda api_key: SimpleNamespace())
    )
    client = build_client(settings, backend="anthropic", api_key=SecretStr("k"))
    assert client.model_name == settings.model_name
