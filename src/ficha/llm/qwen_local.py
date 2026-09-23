"""Qwen2.5-Instruct local via ``transformers`` (Seção 4.3 — GPU T4 do Colab).

Organização:

- **Funções puras** (sem ``torch``), testáveis em qualquer máquina: :func:`build_messages`,
  :func:`generation_kwargs`, :func:`estimate_memory_gb`, :func:`fits_in_memory`,
  :func:`params_billions_from_name`.
- :func:`describe_gpu` — informação da GPU para o notebook imprimir (``None`` sem GPU/torch).
- :class:`QwenLocalClient` — o cliente em si. ``torch``/``transformers`` só são importados ao
  carregar o modelo; instanciar a classe não custa nada.

Regra de bolso do enunciado: em meia precisão, ~2 GB por bilhão de parâmetros, mais o espaço
do contexto. Por isso, na T4 (~16 GB), 1.5B e 3B cabem em fp16 e o 7B só quantizado em 4 bits.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ficha.types import Completion, GenerationParams, Usage

# --------------------------------------------------------------------------- memória

GB_PER_BILLION_FP16 = 2.0
"""Regra de bolso da Seção 4.3: 2 bytes por parâmetro em meia precisão."""

GB_PER_BILLION_4BIT = 0.75
"""4 bits = 0,5 byte por parâmetro, mais folga para as camadas que o bitsandbytes mantém em
16 bits (embeddings e ``lm_head``, grandes no Qwen por causa do vocabulário de ~152 mil tokens)."""

CONTEXT_OVERHEAD_GB = 2.5
"""Folga para contexto (cache KV de alguns milhares de tokens), ativações e o próprio runtime
CUDA. É o "ainda sobra o espaço do contexto" do enunciado."""

T4_MEMORY_GB = 16.0

QWEN_PARAMS_BILLIONS: dict[str, float] = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.49,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
}
"""Contagens reais de parâmetros (cartões dos modelos no Hugging Face)."""

_BILLIONS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB]\b")


def params_billions_from_name(model_name: str) -> float | None:
    """Número de parâmetros (em bilhões) pelo nome do modelo.

    Usa a tabela de contagens reais quando conhece o modelo; senão, lê o ``NB`` do nome
    (ex.: ``...-3B-Instruct`` → 3.0). ``None`` se não der para inferir.
    """
    if model_name in QWEN_PARAMS_BILLIONS:
        return QWEN_PARAMS_BILLIONS[model_name]
    m = _BILLIONS_RE.search(model_name.replace("-", " "))
    return float(m.group(1)) if m else None


def estimate_memory_gb(
    params_billions: float,
    quantize_4bit: bool,
    context_overhead_gb: float = CONTEXT_OVERHEAD_GB,
) -> float:
    """Memória de vídeo estimada (GB) para carregar o modelo e gerar com contexto médio."""
    per_billion = GB_PER_BILLION_4BIT if quantize_4bit else GB_PER_BILLION_FP16
    return params_billions * per_billion + context_overhead_gb


def fits_in_memory(
    params_billions: float,
    quantize_4bit: bool,
    free_gb: float,
    context_overhead_gb: float = CONTEXT_OVERHEAD_GB,
) -> bool:
    """O modelo cabe em ``free_gb`` de memória de vídeo? (regra de bolso da Seção 4.3).

    Exemplos na T4 (16 GB): 1.5B e 3B em fp16 cabem; 7B em fp16 não (14 GB só de pesos,
    sem espaço para o contexto); 7B em 4 bits cabe.
    """
    return estimate_memory_gb(params_billions, quantize_4bit, context_overhead_gb) <= free_gb


# --------------------------------------------------------------------------- mensagens e geração


def build_messages(system: str | None, user: str) -> list[dict[str, str]]:
    """Mensagens no formato de chat do ``tokenizer.apply_chat_template``.

    Quando ``system`` é ``None`` (técnica *papel* desligada), enviamos uma mensagem de sistema
    **vazia** em vez de omiti-la: o chat template do Qwen2.5 insere, na ausência de mensagem de
    sistema, o papel padrão "You are Qwen, created by Alibaba Cloud. You are a helpful
    assistant." — o que contaminaria a comparação "com papel × sem papel".
    """
    return [
        {"role": "system", "content": system if system is not None else ""},
        {"role": "user", "content": user},
    ]


def generation_kwargs(
    params: GenerationParams, max_new_tokens: int | None = None
) -> dict[str, Any]:
    """Argumentos de ``model.generate`` a partir dos parâmetros declarados.

    - ``temperature == 0`` → decodificação gulosa (``do_sample=False``): determinística, escolhe
      sempre o token mais provável. ``temperature/top_p/top_k`` vão como ``None`` para anular os
      valores de amostragem do ``generation_config.json`` do Qwen (0.7 / 0.8 / 20).
    - ``temperature > 0`` → amostragem com ``temperature`` e ``top_p`` declarados; ``top_k=0``
      desliga o top-k padrão do Qwen, para que só os parâmetros declarados atuem.
    - ``repetition_penalty=1.0`` sempre: o padrão do Qwen (1.05) penaliza tokens que **já estão
      no prompt** — ou seja, penaliza justamente copiar o trecho literal do artigo, que é o que
      ``evidencia.trecho`` exige.
    """
    limit = (
        params.max_new_tokens
        if max_new_tokens is None
        else min(params.max_new_tokens, max_new_tokens)
    )
    kwargs: dict[str, Any] = {"max_new_tokens": limit, "repetition_penalty": 1.0}
    if params.temperature <= 0:
        kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        kwargs.update(do_sample=True, temperature=params.temperature, top_p=params.top_p, top_k=0)
    return kwargs


def finish_reason(n_output_tokens: int, max_new_tokens: int) -> str:
    """``"length"`` se a geração parou no limite de tokens (JSON provavelmente truncado)."""
    return "length" if n_output_tokens >= max_new_tokens else "stop"


# --------------------------------------------------------------------------- GPU


def describe_gpu() -> dict[str, Any] | None:
    """Nome e memória da GPU CUDA, ou ``None`` se não houver GPU (ou ``torch``).

    Para o notebook imprimir no início — e o relatório declarar em que hardware rodou.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "nome": torch.cuda.get_device_name(0),
        "memoria_total_gb": round(total_b / 1e9, 2),
        "memoria_livre_gb": round(free_b / 1e9, 2),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }


# --------------------------------------------------------------------------- cliente


class QwenLocalClient:
    """Cliente :class:`ficha.types.LLMClient` para Qwen2.5-Instruct local.

    O modelo é carregado na primeira chamada (ou explicitamente com :meth:`load`), em fp16 na
    GPU com ``device_map="auto"``; com ``quantize_4bit=True`` usa ``BitsAndBytesConfig`` (necessário
    para o 7B na T4). ``Usage`` traz a contagem **real** de tokens pelo tokenizador do modelo.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        quantize_4bit: bool = False,
        max_new_tokens: int = 1024,
        device: str | None = None,
        check_memory: bool = True,
    ) -> None:
        self._model_name = model_name
        self.quantize_4bit = quantize_4bit
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.check_memory = check_memory
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def model_name(self) -> str:
        suffix = "@4bit" if self.quantize_4bit else ""
        return f"{self._model_name}{suffix}"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Carrega tokenizador e modelo (idempotente). Falha cedo se o modelo não couber."""
        if self._model is not None:
            return
        try:
            import torch as _torch
            import transformers as _transformers
        except ImportError as exc:
            raise ImportError(
                'O backend qwen_local exige torch e transformers: pip install "ficha[local]"'
            ) from exc

        # Tipados como Any: o pacote não depende dos stubs (nem da presença) do torch/transformers.
        torch: Any = _torch
        tf: Any = _transformers
        gpu = describe_gpu()
        if self.check_memory and gpu is not None:
            billions = params_billions_from_name(self._model_name)
            if billions is not None and not fits_in_memory(
                billions, self.quantize_4bit, gpu["memoria_livre_gb"]
            ):
                need = estimate_memory_gb(billions, self.quantize_4bit)
                raise MemoryError(
                    f"{self._model_name} precisa de ~{need:.1f} GB e a GPU tem "
                    f"{gpu['memoria_livre_gb']} GB livres. Use quantize_4bit=True "
                    "(FICHA_QUANTIZE_4BIT=true) ou um modelo menor."
                )

        load_kwargs: dict[str, Any] = {}
        if self.quantize_4bit:
            load_kwargs["quantization_config"] = tf.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16 if gpu is not None else torch.float32

        if self.device is None:
            load_kwargs["device_map"] = "auto"

        self._tokenizer = tf.AutoTokenizer.from_pretrained(self._model_name)
        model = tf.AutoModelForCausalLM.from_pretrained(self._model_name, **load_kwargs)
        if self.device is not None and not self.quantize_4bit:
            model = model.to(self.device)
        model.eval()
        self._model = model

    def count_tokens(self, text: str) -> int:
        """Contagem exata pelo tokenizador do Qwen (carrega só o tokenizador, se preciso)."""
        if self._tokenizer is None:
            from ficha.llm.tokens import HFTokenCounter

            return HFTokenCounter(self._model_name).count(text)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(self, system: str | None, user: str, params: GenerationParams) -> Completion:
        """Gera a resposta; gulosa com ``temperature == 0``, amostrada com semente senão."""
        self.load()
        import torch as _torch

        torch: Any = _torch
        tok, model = self._tokenizer, self._model
        prompt_text = tok.apply_chat_template(
            build_messages(system, user), tokenize=False, add_generation_prompt=True
        )
        enc = tok(prompt_text, return_tensors="pt").to(model.device)
        kwargs = generation_kwargs(params, self.max_new_tokens)
        if kwargs["do_sample"] and params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, **kwargs, pad_token_id=tok.eos_token_id)
        latency = time.perf_counter() - t0

        n_in = int(enc["input_ids"].shape[1])
        new_tokens = out[0, n_in:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        n_out = int(new_tokens.shape[0])
        return Completion(
            text=text,
            usage=Usage(input_tokens=n_in, output_tokens=n_out),
            model=self.model_name,
            latency_s=latency,
            params=params,
            raw={
                "finish_reason": finish_reason(n_out, kwargs["max_new_tokens"]),
                "generation_kwargs": {k: v for k, v in kwargs.items() if v is not None},
            },
        )

    def __repr__(self) -> str:
        return (
            f"QwenLocalClient(model_name={self._model_name!r}, quantize_4bit={self.quantize_4bit}, "
            f"max_new_tokens={self.max_new_tokens}, loaded={self.loaded})"
        )
