"""Embedding generation via the Hugging Face Inference API (hosted, remote).

A local ONNX model (fastembed) was tried first, but even the smallest
available multilingual model doesn't fit in Render's free-tier 512MB RAM
once combined with the rest of the app (FastAPI, Qdrant client, Groq client,
ONNX Runtime, and the loaded model itself) -- the service OOM'd loading the
model, before ever touching a document. Calling a hosted model instead means
no ML runtime/model lives in this process at all, which is what actually
keeps memory usage low enough for the free tier.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(384-dim, multilingual incl. Russian) -- same model as before, now hosted on
HF's infrastructure instead of loaded locally.

Trade-offs vs a local model:
- Adds network latency per request/reindex.
- HF's free serverless tier "cold starts": if the model hasn't been called
  recently, the first request can return 503 while HF loads it on their
  side -- handled here with a bounded retry/wait loop.
- Free tier is rate-limited (roughly a few hundred requests/hour) -- fine
  for a personal/small-team tool, not for heavy traffic.
"""
import logging
import time

import requests

logger = logging.getLogger("embeddings")

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"
EMBEDDING_DIM = 384  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
_BATCH_SIZE = 16
_MAX_COLD_START_WAIT_S = 90


class EmbeddingAPIError(RuntimeError):
    pass


def _post_with_cold_start_retry(url: str, headers: dict, payload: dict) -> requests.Response:
    deadline = time.monotonic() + _MAX_COLD_START_WAIT_S
    while True:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 503:
            wait_s = 5.0
            try:
                wait_s = min(float(resp.json().get("estimated_time", 5.0)), 20.0)
            except (ValueError, TypeError):
                pass
            if time.monotonic() + wait_s > deadline:
                return resp
            logger.info("HF-модель ещё прогревается (cold start), жду %.0fs...", wait_s)
            time.sleep(wait_s)
            continue
        return resp


def _pool_if_needed(item: list) -> list[float]:
    # sentence-transformers models normally return one already-pooled vector
    # per input (a flat list of floats). Some deployments instead return
    # token-level vectors (a list of per-token vectors) -- mean-pool those.
    if item and isinstance(item[0], list):
        dim = len(item[0])
        sums = [0.0] * dim
        for token_vec in item:
            for i, v in enumerate(token_vec):
                sums[i] += v
        return [s / len(item) for s in sums]
    return item


def _embed_batch(texts: list[str], model_name: str, hf_token: str) -> list[list[float]]:
    url = f"{HF_ROUTER_BASE}/{model_name}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}
    resp = _post_with_cold_start_retry(url, headers, {"inputs": texts})

    if resp.status_code != 200:
        raise EmbeddingAPIError(
            f"Hugging Face Inference API вернул ошибку {resp.status_code} для модели "
            f"'{model_name}': {resp.text[:500]}"
        )

    data = resp.json()
    return [_pool_if_needed(item) for item in data]


def embed_passages(texts: list[str], model_name: str, hf_token: str) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        vectors.extend(_embed_batch(batch, model_name, hf_token))
    return vectors


def embed_query(text: str, model_name: str, hf_token: str) -> list[float]:
    return _embed_batch([text], model_name, hf_token)[0]


def get_embedding_dim(model_name: str) -> int:
    return EMBEDDING_DIM
