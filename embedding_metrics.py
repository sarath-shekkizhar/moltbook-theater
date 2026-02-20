"""
Embedding-based semantic metrics for Moltbook analysis.

Uses OpenAI text-embedding-3-small (1536-dim) for:
  - Semantic similarity between posts and comments
  - Semantic specificity (actual vs random baseline)
  - Semantic saturation curves (novelty per comment position)

Embeddings are cached to parquet to avoid recomputation.
"""

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import os
import time
import hashlib

import tiktoken
from openai import OpenAI

CACHE = "cache"


def _cache_path(name: str) -> str:
    return os.path.join(CACHE, f"embeddings_{name}.npz")


def embed_texts_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2000,
    cache_name: str | None = None,
) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API.

    Args:
        texts: List of strings to embed.
        model: OpenAI embedding model name.
        batch_size: Number of texts per API call.
        cache_name: If provided, cache results to disk under this name.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    # Check cache
    if cache_name:
        path = _cache_path(cache_name)
        if os.path.exists(path):
            data = np.load(path)
            cached = data["embeddings"]
            if len(cached) == len(texts):
                print(f"  Loaded cached embeddings ({cache_name}): {cached.shape}")
                return cached

    client = OpenAI()
    enc = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 8191  # text-embedding-3-small limit

    # Clean and truncate texts using tiktoken for accurate token counting
    MAX_TOKENS_PER_REQUEST = 250_000  # API limit is 300K, leave headroom
    clean_texts = []
    token_counts = []
    n_truncated = 0
    for t in texts:
        s = str(t).strip() if t and str(t).strip() else " "
        tokens = enc.encode(s, disallowed_special=())
        if len(tokens) > MAX_TOKENS:
            s = enc.decode(tokens[:MAX_TOKENS])
            token_counts.append(MAX_TOKENS)
            n_truncated += 1
        else:
            token_counts.append(len(tokens))
        clean_texts.append(s)
    if n_truncated:
        print(f"  Truncated {n_truncated} texts to {MAX_TOKENS} tokens")

    # Build token-aware batches that stay under the per-request limit
    all_embeddings = []
    t0 = time.time()
    i = 0
    while i < len(clean_texts):
        batch = []
        batch_tokens = 0
        while i < len(clean_texts) and len(batch) < batch_size:
            if batch_tokens + token_counts[i] > MAX_TOKENS_PER_REQUEST and batch:
                break
            batch.append(clean_texts[i])
            batch_tokens += token_counts[i]
            i += 1

        response = client.embeddings.create(model=model, input=batch)
        batch_emb = [item.embedding for item in response.data]
        all_embeddings.extend(batch_emb)

        elapsed = time.time() - t0
        n_done = len(all_embeddings)
        if n_done % 10000 < len(batch) or n_done == len(clean_texts):
            rate = n_done / elapsed if elapsed > 0 else 0
            print(
                f"    Embedded {n_done:,}/{len(clean_texts):,} "
                f"({elapsed:.1f}s, {rate:.0f} texts/sec)"
            )

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Cache to disk
    if cache_name:
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(_cache_path(cache_name), embeddings=embeddings)
        print(f"  Cached embeddings to {_cache_path(cache_name)}")

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def cosine_similarity_batch(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a and each row of matrix B.

    Args:
        a: shape (d,)
        B: shape (n, d)

    Returns:
        np.ndarray of shape (n,) with cosine similarities.
    """
    norm_a = np.linalg.norm(a)
    norms_B = np.linalg.norm(B, axis=1)
    if norm_a == 0:
        return np.zeros(len(B))
    dots = B @ a
    denom = norm_a * norms_B
    denom = np.where(denom == 0, 1.0, denom)
    return dots / denom


def semantic_specificity(
    comment_emb: np.ndarray,
    post_emb: np.ndarray,
    random_embs: np.ndarray,
) -> float:
    """Semantic specificity: cosine sim to actual post minus mean cosine sim to random posts.

    Args:
        comment_emb: Embedding of the comment, shape (d,).
        post_emb: Embedding of the actual post, shape (d,).
        random_embs: Embeddings of random posts, shape (n, d).

    Returns:
        Specificity score. Positive = semantically closer to actual post than random.
    """
    sim_actual = cosine_similarity(comment_emb, post_emb)
    sims_random = cosine_similarity_batch(comment_emb, random_embs)
    sim_random_mean = float(np.mean(sims_random))
    return sim_actual - sim_random_mean


def semantic_saturation_curve(
    embeddings_ordered: np.ndarray,
) -> dict:
    """Compute semantic novelty at each comment position.

    For position k, novelty = 1 - max_cos_sim(emb_k, emb_0..k-1).
    This measures how semantically distinct each new comment is from all
    preceding comments.

    Also computes centroid-based novelty:
    novelty_centroid = 1 - cos_sim(emb_k, mean(emb_0..k-1))

    Args:
        embeddings_ordered: Shape (n_comments, d), ordered by timestamp.

    Returns:
        dict with:
            'max_novelty': list of (1 - max similarity to any prior comment)
            'centroid_novelty': list of (1 - similarity to centroid of prior)
    """
    n = len(embeddings_ordered)
    max_novelty = []
    centroid_novelty = []

    for k in range(n):
        if k == 0:
            # First comment is fully novel
            max_novelty.append(1.0)
            centroid_novelty.append(1.0)
            continue

        prior = embeddings_ordered[:k]
        current = embeddings_ordered[k]

        # Max similarity to any prior comment
        sims = cosine_similarity_batch(current, prior)
        max_sim = float(np.max(sims))
        max_novelty.append(1.0 - max_sim)

        # Similarity to centroid of prior comments
        centroid = prior.mean(axis=0)
        cent_sim = cosine_similarity(current, centroid)
        centroid_novelty.append(1.0 - cent_sim)

    return {
        "max_novelty": max_novelty,
        "centroid_novelty": centroid_novelty,
    }
