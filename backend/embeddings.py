"""Embedding generation via fastembed (ONNX runtime, no torch dependency).

We use fastembed instead of plain sentence-transformers because
sentence-transformers pulls in PyTorch, which alone is commonly 500MB-1GB+ of
RAM/disk -- risky on Render's free tier (512MB-1GB RAM). fastembed runs the
same underlying sentence-transformers models through ONNX Runtime with a much
smaller footprint (this model is ~0.22GB), which is a better fit for a
free-tier deployment.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-dim, multilingual incl. Russian). Unlike E5-family models, this one does
not require "query: "/"passage: " prefixes on the input text.

Render's free tier caps a service at 512MB RAM, which is tight for any ONNX
model. To stay under that: the ONNX Runtime session is limited to a single
thread (avoids per-thread buffer duplication) and embed() calls use a small
batch_size instead of fastembed's default of 256 (avoids holding a large
batch's tokenized/output tensors in memory at once for files with many
chunks).
"""
import logging

from fastembed import TextEmbedding

logger = logging.getLogger("embeddings")

_EMBED_BATCH_SIZE = 16

_model: TextEmbedding | None = None
_model_name: str | None = None


def _get_model(model_name: str) -> TextEmbedding:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        logger.info("Загрузка embedding-модели '%s'...", model_name)
        _model = TextEmbedding(model_name=model_name, threads=1)
        _model_name = model_name
        logger.info("Embedding-модель загружена")
    return _model


def embed_passages(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model(model_name)
    return [v.tolist() for v in model.embed(texts, batch_size=_EMBED_BATCH_SIZE)]


def embed_query(text: str, model_name: str) -> list[float]:
    model = _get_model(model_name)
    vectors = list(model.embed([text], batch_size=_EMBED_BATCH_SIZE))
    return vectors[0].tolist()


def get_embedding_dim(model_name: str) -> int:
    model = _get_model(model_name)
    return list(model.embed(["dimension probe"], batch_size=_EMBED_BATCH_SIZE))[0].shape[0]
