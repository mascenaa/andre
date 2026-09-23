"""Cliente falso e determinístico.

Serve a dois propósitos:

1. **Testes** de toda a cadeia (parser, auditoria, custo, relatório) sem GPU nem rede.
2. **Modo de ensaio** do notebook: roda de ponta a ponta em segundos com PDFs sintéticos,
   provando que o pipeline funciona antes de gastar GPU/API com os 19 artigos reais.

O comportamento é controlado por :class:`FakeBehavior`, o que permite simular exatamente os
erros que a auditoria da Seção 4.4 precisa detectar: JSON quebrado, trecho inventado,
limitação inventada quando deveria ser ``null``, e instabilidade entre execuções.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

from ficha.types import Completion, GenerationParams, Usage

_PAGE_MARK_RE = re.compile(r"^\[p\. (\d+)\]\s*$", re.MULTILINE)
_DELIM_RE = re.compile(r"<<<ARTIGO>>>(.*?)<<<FIM_ARTIGO>>>", re.DOTALL)
_ARQUIVO_RE = re.compile(r"[\w\-]+\.pdf")
_PLAIN_ARTICLE_RE = re.compile(r"^Artigo:[ \t]*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class FakeBehavior:
    """Como o cliente falso deve se comportar. Tudo desligado = ficha correta e estável."""

    broken_json_rate: float = 0.0
    """Fração de chamadas que devolvem JSON malformado (vírgula sobrando + cerca de código)."""
    invent_trecho_rate: float = 0.0
    """Fração de chamadas em que ``evidencia.trecho`` NÃO existe no contexto enviado."""
    invent_limitacao_rate: float = 0.0
    """Fração de chamadas em que ``limitacao`` é inventada embora o texto não declare nenhuma."""
    unstable_rate: float = 0.0
    """Fração de chamadas em que um campo textual varia (simula não-determinismo)."""
    truncate_rate: float = 0.0
    """Fração de chamadas em que a saída é cortada no meio do JSON, como quando um modelo
    pequeno bate em ``max_new_tokens`` (``Completion.raw["finish_reason"] == "length"``).
    Nenhum reparo recupera um objeto incompleto, então o resultado é ``FAILED`` de verdade:
    serve para o modo de ensaio do notebook exibir um caso FAILED."""
    temperature_noise: bool = True
    """Se True, temperatura > 0 aumenta as taxas acima proporcionalmente (efeito 4.4d)."""
    chars_per_token: float = 4.0
    """Aproximação declarada para ``count_tokens``."""


@dataclass(slots=True)
class FakeLLM:
    """Implementa :class:`ficha.types.LLMClient` sem depender de nada externo."""

    behavior: FakeBehavior = field(default_factory=FakeBehavior)
    name: str = "fake/deterministic-v1"
    calls: list[tuple[str | None, str, GenerationParams]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.name

    def count_tokens(self, text: str) -> int:
        return max(1, round(len(text) / self.behavior.chars_per_token))

    # ------------------------------------------------------------------ geração
    def generate(self, system: str | None, user: str, params: GenerationParams) -> Completion:
        t0 = time.perf_counter()
        self.calls.append((system, user, params))
        rng = self._rng(user, params)
        article = self._article_text(user)
        pages = self._pages(article)
        first_page = pages[0][0] if pages else 1
        body = pages[0][1] if pages else article

        text, truncated = self._build_ficha_json(body, first_page, rng, params)
        usage = Usage(
            input_tokens=self.count_tokens((system or "") + user),
            output_tokens=self.count_tokens(text),
        )
        return Completion(
            text=text,
            usage=usage,
            model=self.name,
            latency_s=time.perf_counter() - t0,
            params=params,
            raw={"fake": True, "finish_reason": "length" if truncated else "stop"},
        )

    # ------------------------------------------------------------------ internos
    def _rng(self, user: str, params: GenerationParams) -> random.Random:
        # Determinístico para (prompt, seed, temperatura): mesma entrada → mesma saída,
        # exatamente como um modelo com seed fixa e temperatura 0 deveria se comportar.
        # Com temperatura > 0 e temperature_noise, adicionamos entropia ao seed para
        # simular amostragem.
        base = f"{user}|{params.seed}|{params.temperature}"
        if params.temperature > 0 and self.behavior.temperature_noise:
            base += f"|{time.perf_counter_ns()}"
        return random.Random(base)

    def _rate(self, rate: float, params: GenerationParams) -> float:
        if params.temperature > 0 and self.behavior.temperature_noise:
            return min(1.0, rate + params.temperature * 0.3)
        return rate

    @staticmethod
    def _article_text(user: str) -> str:
        """Localiza o artigo no prompt.

        Com delimitadores, o texto do último par ``<<<ARTIGO>>>``/``<<<FIM_ARTIGO>>>``. Sem eles
        (variante ``v_sem_delimitadores``), o que vem depois da **última** linha ``Artigo:`` —
        os exemplos few-shot vêm antes e também têm marcadores ``[p. N]``.
        """
        # O último bloco delimitado é o artigo (o prompt termina nele); isso tolera uma menção
        # aos marcadores na prosa da instrução.
        delimited = list(_DELIM_RE.finditer(user))
        if delimited:
            return delimited[-1].group(1)
        plain = list(_PLAIN_ARTICLE_RE.finditer(user))
        return user[plain[-1].end() :] if plain else user

    @staticmethod
    def _pages(article: str) -> list[tuple[int, str]]:
        """Divide o contexto pelos marcadores ``[p. N]`` gerados por ``Context.render``."""
        out: list[tuple[int, str]] = []
        parts = _PAGE_MARK_RE.split(article)
        # parts = [antes, num, texto, num, texto, ...]
        for i in range(1, len(parts) - 1, 2):
            out.append((int(parts[i]), parts[i + 1].strip()))
        return out

    @staticmethod
    def _sentence(body: str, rng: random.Random, min_len: int = 40) -> str:
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) >= min_len
        ]
        if not sentences:
            return body[: max(min_len, 80)]
        return rng.choice(sentences[: max(1, len(sentences) // 2)])

    def _build_ficha_json(
        self, body: str, page: int, rng: random.Random, params: GenerationParams
    ) -> tuple[str, bool]:
        """JSON da ficha (possivelmente estragado) e se a saída foi truncada."""
        b = self.behavior
        declara_limitacao = re.search(r"limitation", body, re.IGNORECASE) is not None
        trecho = self._sentence(body, rng)

        if rng.random() < self._rate(b.invent_trecho_rate, params):
            trecho = (
                "The proposed framework achieves state-of-the-art results "
                "across all benchmarks considered."
            )

        limitacao: str | None
        if declara_limitacao:
            limitacao = "Os autores reconhecem limitações declaradas no texto."
        elif rng.random() < self._rate(b.invent_limitacao_rate, params):
            limitacao = "O tamanho reduzido da amostra limita a generalização dos resultados."
        else:
            limitacao = None

        variante = ""
        if rng.random() < self._rate(b.unstable_rate, params):
            variante = f" (variante {rng.randint(1, 9)})"

        ficha = {
            "problema": f"Resolver o problema descrito no artigo{variante}.",
            "dados": "Dados descritos na seção de dados do artigo.",
            "metodo": f"Método principal descrito no artigo{variante}.",
            "metrica": "Métrica reportada nos resultados.",
            "limitacao": limitacao,
            "evidencia": {"trecho": trecho, "pagina": page},
        }
        text = json.dumps(ficha, ensure_ascii=False, indent=2)

        if rng.random() < self._rate(b.broken_json_rate, params):
            # Erros típicos de modelo pequeno: cerca de markdown + vírgula final sobrando.
            text = (
                "```json\n"
                + text.replace('"pagina": ' + str(page), '"pagina": ' + str(page) + ",")
                + "\n```"
            )

        # Sorteado por último para não alterar a sequência das taxas anteriores.
        if rng.random() < self._rate(b.truncate_rate, params):
            # Corta entre 30% e 80% do texto: o objeto fica sempre desbalanceado.
            cut = int(len(text) * rng.uniform(0.3, 0.8))
            return text[:cut], True
        return text, False
