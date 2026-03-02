"""Three-stage hybrid retrieval over a single Wikipedia page's link set.

Pipeline
--------
1. BM25 recall        — fast keyword matching, keeps top ``bm25_k`` candidates
2. Bi-encoder         — ONNX-backed fastembed cosine similarity, keeps top ``bi_k``
   Both sets are fused with min-max normalisation and deduplicated into a pool.
3. Cross-encoder      — ONNX-backed flashrank reranker jointly scores (query, title)
   pairs for high-precision final ranking, returns top ``k`` results.

All three models are passed in at construction time so the caller can cache them
across many pages (only one load per agent lifetime).  The bi-encoder and
cross-encoder use ONNX Runtime under the hood (via fastembed / flashrank) and
have no dependency on ``transformers`` or PyTorch.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if math.isclose(float(hi - lo), 0.0):
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class WikiPageIndex:
    """In-memory retrieval index for the links on one Wikipedia page.

    Parameters
    ----------
    items:
        List of ``(title, section_heading)`` pairs representing every link on
        the page.
    bi_encoder:
        A loaded ``fastembed.TextEmbedding`` model used for semantic embeddings.
    cross_encoder:
        A loaded ``flashrank.Ranker`` model used for final reranking.
    bm25_k:
        Number of BM25 candidates to keep before fusion.
    bi_k:
        Number of bi-encoder candidates to keep before fusion.
    """

    def __init__(
        self,
        items: List[Tuple[str, str]],
        bi_encoder: TextEmbedding,
        cross_encoder: Ranker,
        bm25_k: int = 40,
        bi_k: int = 40,
    ) -> None:
        self._items = items
        self._titles = [t for t, _ in items]
        self._sections = [s for _, s in items]
        self._bi_encoder = bi_encoder
        self._cross_encoder = cross_encoder
        self._bm25_k = bm25_k
        self._bi_k = bi_k

        # Build BM25 index
        tokenized = [_tokenize(t) for t in self._titles]
        self._bm25 = BM25Okapi(tokenized)

        # Pre-compute bi-encoder embeddings for all titles (ONNX, fast)
        if self._titles:
            self._title_embeddings: np.ndarray = np.array(
                list(bi_encoder.embed(self._titles)), dtype=np.float32
            )
        else:
            self._title_embeddings = np.empty((0, 0), dtype=np.float32)

    def search(
        self, query: str, k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Run the full three-stage pipeline for a single query.

        Returns
        -------
        List of ``(title, section, score)`` sorted descending by cross-encoder
        score, at most ``k`` results.
        """
        if not self._titles:
            return []

        n = len(self._titles)

        # --- Stage 1: BM25 ---
        bm25_raw = np.array(self._bm25.get_scores(_tokenize(query)), dtype=np.float64)
        bm25_top_idx = set(
            np.argsort(bm25_raw)[::-1][: min(self._bm25_k, n)].tolist()
        )

        # --- Stage 2: Bi-encoder ---
        query_emb = np.array(
            list(self._bi_encoder.embed([query])), dtype=np.float32
        )[0]
        norms = np.linalg.norm(self._title_embeddings, axis=1) * np.linalg.norm(query_emb)
        cosine_raw = np.where(
            norms > 0,
            self._title_embeddings @ query_emb / np.where(norms > 0, norms, 1.0),
            0.0,
        )
        bi_top_idx = set(
            np.argsort(cosine_raw)[::-1][: min(self._bi_k, n)].tolist()
        )

        # --- Fusion: union of both candidate sets ---
        pool_idx = sorted(bm25_top_idx | bi_top_idx)
        if not pool_idx:
            return []

        bm25_norm = _minmax(bm25_raw[pool_idx])
        bi_norm = _minmax(cosine_raw[pool_idx])
        fused = 0.5 * bm25_norm + 0.5 * bi_norm

        # Sort pool by fused score (limits cross-encoder work to best candidates)
        pool_sorted = [pool_idx[i] for i in np.argsort(fused)[::-1]]

        # --- Stage 3: Cross-encoder rerank via flashrank ---
        passages = [
            {"id": i, "text": self._titles[idx]}
            for i, idx in enumerate(pool_sorted)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self._cross_encoder.rerank(rerank_req)

        # Map back to original indices; reranked is sorted by score desc
        results: List[Tuple[str, str, float]] = []
        for passage in reranked[:k]:
            orig_idx = pool_sorted[passage.get("id", 0)]
            score = float(passage.get("score", 0.0))
            results.append((self._titles[orig_idx], self._sections[orig_idx], score))

        return results


def load_models(
    bi_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ce_model_name: str = "ms-marco-TinyBERT-L-2-v2",
) -> Tuple[TextEmbedding, Ranker]:
    """Load and return both retrieval models.

    Call this once at agent initialisation and pass the results to every
    ``WikiPageIndex`` you create.

    Both models use ONNX Runtime — no PyTorch or ``transformers`` dependency.
    The cross-encoder default (TinyBERT, ~4 MB) balances speed and accuracy.
    For higher accuracy at the cost of ~66 MB, use ``ms-marco-MiniLM-L-12-v2``.
    """
    bi_encoder = TextEmbedding(bi_model_name)
    cross_encoder = Ranker(model_name=ce_model_name)
    return bi_encoder, cross_encoder


def build_page_index(
    page_structure: dict,
    bi_encoder: TextEmbedding,
    cross_encoder: Ranker,
) -> WikiPageIndex:
    """Convenience constructor: build a ``WikiPageIndex`` from the dict returned
    by ``WikipediaClient.get_page_with_structure()``.
    """
    items = [
        (title, page_structure["link_sections"].get(title, ""))
        for title in page_structure["links"]
    ]
    return WikiPageIndex(items=items, bi_encoder=bi_encoder, cross_encoder=cross_encoder)
