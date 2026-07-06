"""BGE Chinese embedding model support for MemPalace.

BAAI General Embedding (BGE) models are optimized for Chinese and multilingual
retrieval. This module provides ChromaDB-compatible embedding functions for:

* ``BAAI/bge-small-zh-v1.5`` — 512-dim, ~100 MB, fast (default for ``bge-zh``)
* ``BAAI/bge-base-zh-v1.5`` — 768-dim, ~400 MB, higher quality
* ``BAAI/bge-large-zh-v1.5`` — 1024-dim, ~1.3 GB, best quality

BGE models use an instruction prefix for queries to distinguish query vs document
embeddings, which significantly improves retrieval quality.

Install the optional dependency:
    pip install mempalace[bge]

Select via config or env:
    export MEMPALACE_EMBEDDING_MODEL=bge-small-zh
    mempalace init ~/my-palace

Or in ``~/.mempalace/config.json``:
    {"embedding_model": "bge-small-zh"}

Switching from another model requires re-embedding:
    mempalace repair rebuild-index
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, Optional

logger = logging.getLogger(__name__)

# Map of user-facing model name -> HuggingFace model ID + dims
_BGE_MODELS = {
    "bge-small-zh": {
        "hf_id": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "desc": "BGE Small Chinese — 512-dim, fast, good for demos",
    },
    "bge-base-zh": {
        "hf_id": "BAAI/bge-base-zh-v1.5",
        "dim": 768,
        "desc": "BGE Base Chinese — 768-dim, balanced quality/speed",
    },
    "bge-large-zh": {
        "hf_id": "BAAI/bge-large-zh-v1.5",
        "dim": 1024,
        "desc": "BGE Large Chinese — 1024-dim, best quality",
    },
}

# BGE query instruction prefix (v1.5 models)
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEEmbeddingFunction:
    """ChromaDB-compatible embedding function using BAAI BGE models.

    Supports query/document differentiation via instruction prefixes for
    improved retrieval quality. Model is lazy-loaded on first use.

    Parameters
    ----------
    model_key : str
        One of ``"bge-small-zh"``, ``"bge-base-zh"``, ``"bge-large-zh"``.
    device : str or None
        Device for model inference (``"cpu"``, ``"cuda"``, ``"auto"``).
        Default ``"auto"`` picks the best available device.
    """

    _MODEL_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_key: str = "bge-small-zh",
        device: Optional[str] = None,
    ):
        if model_key not in _BGE_MODELS:
            valid = ", ".join(_BGE_MODELS)
            raise ValueError(
                f"Unknown BGE model key {model_key!r}. Valid keys: {valid}"
            )
        self._model_key = model_key
        self._model_info = _BGE_MODELS[model_key]
        self._device = device or "auto"
        self._model = None
        self._load_lock = threading.Lock()

    @staticmethod
    def name() -> str:
        # ChromaDB persists EF name on the collection; this is the stable
        # identity used for collection → embedder matching.
        return "bge_zh"

    @property
    def model_name(self) -> str:
        return self._model_key

    @property
    def dimension(self) -> int:
        return self._model_info["dim"]

    def _lazy_load(self):
        """Load the BGE model on first use (thread-safe)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "BGE embedding requires sentence-transformers. "
                    "Install with: pip install mempalace[bge]"
                )

            hf_id = self._model_info["hf_id"]
            device = None if self._device == "auto" else self._device
            logger.info(
                "Loading BGE model %s (%d-dim) on device %s…",
                hf_id,
                self._model_info["dim"],
                self._device,
            )
            self._model = SentenceTransformer(hf_id, device=device)

    def __call__(self, input: str | list[str] | None) -> list[list[float]]:  # noqa: A002
        """Embed documents (ChromaDB EF protocol). No query instruction prefix."""
        if isinstance(input, str):
            input = [input]
        if input is None or len(input) == 0:
            return []
        self._lazy_load()
        embeddings = self._model.encode(
            input,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed queries with BGE instruction prefix (ChromaDB EF protocol)."""
        if not input:
            return []
        self._lazy_load()
        queries = [_BGE_QUERY_INSTRUCTION + t for t in input]
        embeddings = self._model.encode(
            queries,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed documents without instruction prefix (ChromaDB EF protocol)."""
        return self(input)


# ---------------------------------------------------------------------------
# Registry — integrates with mempalace.embedding.get_embedding_function()
# ---------------------------------------------------------------------------

_BGE_EF_CACHE: dict = {}
_BGE_CACHE_LOCK = threading.Lock()


def get_bge_embedding_function(
    model_key: str = "bge-small-zh",
    device: Optional[str] = None,
):
    """Return a cached BGE embedding function.

    Shared across calls so model load cost is paid once per process.
    """
    cache_key = (model_key, device or "auto")
    cached = _BGE_EF_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _BGE_CACHE_LOCK:
        cached = _BGE_EF_CACHE.get(cache_key)
        if cached is not None:
            return cached
        ef = BGEEmbeddingFunction(model_key=model_key, device=device)
        _BGE_EF_CACHE[cache_key] = ef
    return ef


def list_bge_models() -> dict:
    """Return available BGE models with metadata for UI display."""
    return dict(_BGE_MODELS)
