from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from ficha.llm.qwen_local import (
    QwenLocalClient,
    build_messages,
    estimate_memory_gb,
    finish_reason,
    fits_in_memory,
    generation_kwargs,
    params_billions_from_name,
)
from ficha.types import GenerationParams, LLMClient

# ------------------------------------------------------------------ mensagens


def test_build_messages_system_e_user():
    msgs = build_messages("papel", "pergunta")
    assert msgs == [
        {"role": "system", "content": "papel"},
        {"role": "user", "content": "pergunta"},
    ]


def test_build_messages_sem_papel_envia_system_vazio():
    # Evita que o chat template do Qwen injete o papel padrão "You are Qwen...".
    msgs = build_messages(None, "pergunta")
    assert msgs[0] == {"role": "system", "content": ""}
    assert msgs[1]["content"] == "pergunta"


# ------------------------------------------------------------------ kwargs de geração


def test_generation_kwargs_temperatura_zero_e_gulosa():
    kw = generation_kwargs(GenerationParams(temperature=0.0, max_new_tokens=512))
    assert kw["do_sample"] is False
    assert kw["temperature"] is None and kw["top_p"] is None and kw["top_k"] is None
    assert kw["max_new_tokens"] == 512
    assert kw["repetition_penalty"] == 1.0


def test_generation_kwargs_temperatura_positiva_amostra():
    kw = generation_kwargs(GenerationParams(temperature=0.7, top_p=0.9, max_new_tokens=256))
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 0.9
    assert kw["top_k"] == 0
    assert kw["repetition_penalty"] == 1.0


def test_generation_kwargs_respeita_teto_do_cliente():
    kw = generation_kwargs(GenerationParams(max_new_tokens=2048), max_new_tokens=1024)
    assert kw["max_new_tokens"] == 1024


def test_finish_reason():
    assert finish_reason(1024, 1024) == "length"
    assert finish_reason(300, 1024) == "stop"


# ------------------------------------------------------------------ memória (Seção 4.3)


@pytest.mark.parametrize(
    ("billions", "quant", "cabe"),
    [
        (1.5, False, True),
        (3.0, False, True),
        (7.0, False, False),  # 14 GB só de pesos + contexto > 16 GB
        (7.0, True, True),  # 7B só quantizado em 4 bits
        (1.5, True, True),
        (3.0, True, True),
    ],
)
def test_fits_in_memory_na_t4(billions, quant, cabe):
    assert fits_in_memory(billions, quant, free_gb=16.0) is cabe


def test_fits_in_memory_com_contagem_real_do_7b():
    assert not fits_in_memory(7.62, False, 16.0)
    assert fits_in_memory(7.62, True, 16.0)


def test_estimate_memory_regra_de_bolso_2gb_por_bilhao():
    assert estimate_memory_gb(3.0, False, context_overhead_gb=0) == pytest.approx(6.0)
    assert estimate_memory_gb(7.0, True, context_overhead_gb=0) < 7.0


def test_params_billions_from_name():
    assert params_billions_from_name("Qwen/Qwen2.5-3B-Instruct") == pytest.approx(3.09)
    assert params_billions_from_name("Qwen/Qwen2.5-1.5B-Instruct") == pytest.approx(1.54)
    assert params_billions_from_name("org/Outro-14B-Chat") == pytest.approx(14.0)
    assert params_billions_from_name("modelo-sem-tamanho") is None


# ------------------------------------------------------------------ cliente (sem torch real)


def test_instanciar_nao_carrega_nada():
    client = QwenLocalClient("Qwen/Qwen2.5-7B-Instruct", quantize_4bit=True)
    assert isinstance(client, LLMClient)
    assert not client.loaded
    assert client.model_name == "Qwen/Qwen2.5-7B-Instruct@4bit"
    assert "loaded=False" in repr(client)


class _Enc(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        return "PROMPT"

    def __call__(self, text, return_tensors):
        return _Enc(input_ids=np.array([[1, 2, 3, 4, 5]]))

    def decode(self, ids, skip_special_tokens):
        return '{"ok": true}'

    def encode(self, text, add_special_tokens=False):
        return list(text)


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.array([[1, 2, 3, 4, 5, 9, 9, 9]])


def _fake_torch(seeds):
    return SimpleNamespace(
        manual_seed=lambda s: seeds.append(s),
        cuda=SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda s: None),
        inference_mode=nullcontext,
    )


@pytest.mark.parametrize(("temp", "sample"), [(0.0, False), (0.7, True)])
def test_generate_usage_real_e_semente(monkeypatch, temp, sample):
    seeds: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(seeds))
    client = QwenLocalClient("Qwen/Qwen2.5-3B-Instruct", max_new_tokens=3)
    client._tokenizer, client._model = _FakeTokenizer(), _FakeModel()

    comp = client.generate("papel", "pergunta", GenerationParams(temperature=temp, seed=7))

    assert comp.text == '{"ok": true}'
    assert comp.usage.input_tokens == 5
    assert comp.usage.output_tokens == 3
    assert comp.raw["finish_reason"] == "length"  # 3 de 3 tokens: bateu no teto
    assert client._model.kwargs["do_sample"] is sample
    assert seeds == ([7] if sample else [])
    assert client._tokenizer.messages[0] == {"role": "system", "content": "papel"}
    assert client.count_tokens("abcd") == 4
