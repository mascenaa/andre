from __future__ import annotations

import pytest

from ficha.llm import tokens
from ficha.llm.tokens import HFTokenCounter, approx_token_count


def test_approx_token_count_premissa_4_chars():
    assert approx_token_count("") == 0
    assert approx_token_count("abc") == 1
    assert approx_token_count("a" * 400) == 100
    assert approx_token_count("a" * 400, chars_per_token=2.0) == 200


def test_approx_token_count_rejeita_premissa_invalida():
    with pytest.raises(ValueError):
        approx_token_count("abc", chars_per_token=0)


def test_hf_token_counter_usa_tokenizador_preguicoso(monkeypatch):
    loaded: list[str] = []

    class FakeTok:
        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return text.split()

    def fake_load(name):
        loaded.append(name)
        return FakeTok()

    monkeypatch.setattr(tokens, "_load_tokenizer", fake_load)
    counter = HFTokenCounter("Qwen/Qwen2.5-3B-Instruct")
    assert loaded == []  # nada carregado ao instanciar
    assert counter("um dois tres") == 3
    assert loaded == ["Qwen/Qwen2.5-3B-Instruct"]
    assert "Qwen2.5-3B" in repr(counter)
