"""Utility module wrapping the BGE reranker model."""
from __future__ import annotations

import os
from typing import Iterable, List, Sequence, Tuple

# 优先使用镜像与更快的下载器（与其余脚本保持一致）
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

from FlagEmbedding import FlagReranker  # type: ignore


class BGEReranker:
    """Wrapper around BAAI/bge-reranker-v2-m3."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True) -> None:
        self._model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[float, dict]],
        topk: int,
    ) -> List[Tuple[float, dict]]:
        if not candidates:
            return []
        pairs: List[Tuple[str, str]] = [(query, chunk["text"]) for _, chunk in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        reranked = sorted(
            ((float(score), candidate[1]) for score, candidate in zip(scores, candidates)),
            key=lambda item: item[0],
            reverse=True,
        )
        return reranked[:topk]
