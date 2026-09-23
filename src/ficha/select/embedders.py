"""Embedders para a seleção semântica (Seção 4.1, terceiro caminho).

A busca vetorial aqui **não é o produto** (enunciado, 4.1): é só o modo de escolher o que entra
no prompt. Por isso o contrato é mínimo — texto → vetor normalizado (L2), de modo que o produto
escalar é o cosseno.

Duas implementações:

- :class:`SentenceTransformerEmbedder` — o padrão para os artigos reais. O modelo padrão
  (``Settings.embedding_model``) é **multilíngue** (``paraphrase-multilingual-MiniLM-L12-v2``)
  porque as consultas são escritas em português e os artigos estão em inglês: um modelo só de
  inglês trataria "o que este trabalho não consegue fazer" como ruído. Roda em CPU.
- :class:`HashingEmbedder` — bag-of-words + n-gramas de caracteres com *feature hashing*
  determinístico. Sem download, sem GPU, instantâneo: usado nos testes e no modo de ensaio.
  Os n-gramas de caracteres (sem acento) dão alguma ponte entre cognatos PT↔EN
  (``limitações``/``limitations``, ``métrica``/``metric``), mas **não** entendem paráfrase:
  é um substituto funcional, não semântico.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class Embedder(Protocol):
    """Contrato de um embedder: textos → matriz ``(n, d)`` com linhas de norma L2 = 1."""

    @property
    def name(self) -> str:
        """Identificador do modelo, registrado em ``Context.params``."""
        ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Vetoriza ``texts``. Linhas normalizadas (L2); vetor nulo fica nulo."""
        ...


def l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normaliza cada linha para norma 1 (linhas nulas permanecem nulas)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------- hashing

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS: frozenset[str] = frozenset(
    {
        # inglês
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "in",
        "on",
        "to",
        "for",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "we",
        "our",
        "it",
        "its",
        "as",
        "at",
        "which",
        "not",
        "can",
        "use",
        "used",
        "using",
        "than",
        "into",
        "over",
        "under",
        "also",
        "such",
        # português (sem acento: o texto é normalizado antes)
        "o",
        "os",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "e",
        "ou",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "para",
        "por",
        "com",
        "que",
        "se",
        "ao",
        "aos",
        "este",
        "esta",
        "estes",
        "estas",
        "isso",
        "isto",
        "como",
        "foi",
        "foram",
        "ser",
        "sao",
        "nao",
        "mais",
        "pelo",
        "pela",
        "pelos",
        "pelas",
    }
)


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class HashingEmbedder:
    """Embedder determinístico por *feature hashing* (sem download). Ver docstring do módulo.

    Cada texto vira contagens de palavras (sem stopwords) e de n-gramas de caracteres
    (``ngram_range``) das palavras com pelo menos 4 letras; cada feature cai num de ``dim``
    baldes via CRC32 (estável entre execuções, ao contrário de ``hash()`` do Python) com sinal
    também derivado do hash, e o peso é ``1 + log(contagem)``.
    """

    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dim < 8:
            raise ValueError(f"dim deve ser >= 8, recebido {dim}")
        self.dim = dim
        self.ngram_range = ngram_range

    @property
    def name(self) -> str:
        lo, hi = self.ngram_range
        return f"hashing-bow-char{lo}{hi}-d{self.dim}"

    def features(self, text: str) -> Counter[str]:
        """Features (palavras e n-gramas de caracteres) de um texto."""
        feats: Counter[str] = Counter()
        lo, hi = self.ngram_range
        for tok in _TOKEN_RE.findall(_strip_accents(text)):
            if tok in STOPWORDS:
                continue
            feats[f"w:{tok}"] += 1
            if len(tok) >= 4:
                padded = f"<{tok}>"
                for n in range(lo, hi + 1):
                    for i in range(len(padded) - n + 1):
                        feats[f"c:{padded[i : i + n]}"] += 1
        return feats

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Vetoriza ``texts`` em ``(len(texts), dim)``, linhas normalizadas."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self.features(text).items():
                h = zlib.crc32(feat.encode("utf-8"))
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + np.log(count))
        return l2_normalize(out)


# --------------------------------------------------------------------------- sentence-transformers


class SentenceTransformerEmbedder:
    """Embedder com ``sentence-transformers``. O modelo só é carregado na primeira chamada.

    O import também é tardio: quem não usa a estratégia semântica não precisa do pacote
    (extra ``embeddings`` do ``pyproject``) nem paga o tempo de carga.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depende do ambiente
                raise ImportError(
                    "A seleção semântica precisa de `sentence-transformers` "
                    "(pip install -e '.[embeddings]'). Para testes/ensaio use HashingEmbedder."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Vetoriza ``texts`` com o modelo (normalização L2 feita pelo próprio encode)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
