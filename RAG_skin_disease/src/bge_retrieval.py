"""BGE-based retrieval utilities for the skin dataset."""
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

# 环境变量与其余脚本保持一致
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from src.ingestion import VectorDBIngestorBGE
from src.bge_reranker import BGEReranker


@dataclass
class DocumentIndex:
    sha1_name: str
    index: faiss.Index
    chunks: Sequence[Dict[str, object]]
    category: str
    chunk_file: Optional[Path] = None

    def search(self, query_vector: np.ndarray, topk: int) -> List[Tuple[float, Dict[str, object]]]:
        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Embedding dimension mismatch: query=%.0f, index=%d." % (query_vector.shape[1], self.index.d)
            )
        scores, indices = self.index.search(query_vector, topk)
        results: List[Tuple[float, Dict[str, object]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            chunk_copy = dict(chunk)
            chunk_copy.setdefault("sha1_name", self.sha1_name)
            chunk_copy.setdefault("category", self.category)
            chunk_copy.setdefault("chunk_index", int(idx))
            if self.chunk_file is not None:
                chunk_copy.setdefault("chunk_file", str(self.chunk_file))
            results.append((float(score), chunk_copy))
        return results


class EmbeddingBackend:
    def __init__(self, batch_size: int = 12, max_length: int = 8192) -> None:
        self._backend = VectorDBIngestorBGE(model_name="BAAI/bge-m3", use_fp16=True)
        self._batch_size = batch_size
        self._max_length = max_length

    def encode(self, text: str) -> np.ndarray:
        vecs = self._backend._get_embeddings([text], batch_size=self._batch_size, max_length=self._max_length)
        arr = np.array(vecs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr


def _iter_category_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def _load_chunks(chunk_file: Path) -> List[Dict[str, object]]:
    with chunk_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("content", {}).get("chunks", [])
    return chunks if isinstance(chunks, list) else []


def _load_document_index(chunk_file: Path, index_file: Path, category: str) -> Optional[DocumentIndex]:
    if not index_file.exists():
        return None
    chunks = _load_chunks(chunk_file)
    if not chunks:
        return None
    index = faiss.read_index(str(index_file))
    return DocumentIndex(
        sha1_name=chunk_file.stem,
        index=index,
        chunks=chunks,
        category=category,
        chunk_file=chunk_file,
    )


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _build_category_tokens(name: str) -> List[str]:
    raw_tokens = re.split(r"[^A-Za-z0-9]+", name)
    sanitized = [_sanitize_token(tok) for tok in raw_tokens]
    ignored = {"", "photos", "pictures", "other", "and", "the", "diseases", "disease"}
    return [tok for tok in sanitized if tok not in ignored and len(tok) > 2]


def _match_category(question: str, token_map: Dict[str, List[str]]) -> Tuple[Optional[str], List[str]]:
    q = _sanitize_token(question)
    best_category: Optional[str] = None
    best_matches: List[str] = []
    best_score = 0
    for category, tokens in token_map.items():
        matches = [tok for tok in tokens if tok and tok in q]
        score = len(matches)
        if score > best_score:
            best_category = category
            best_score = score
            best_matches = matches
    return (best_category, best_matches) if best_score > 0 else (None, [])


class BGERetrieval:
    def __init__(
        self,
        root_path: Path,
        batch_size: int = 12,
        max_length: int = 8192,
        use_reranker: bool = False,
        rerank_candidates: int = 20,
    ) -> None:
        self._root_path = root_path
        self._encoder = EmbeddingBackend(batch_size=batch_size, max_length=max_length)
        self._use_reranker = use_reranker
        self._rerank_candidates = rerank_candidates
        self._reranker = BGEReranker() if use_reranker else None
        self._category_docs, self._token_map = self._load_indexes()

    def _load_indexes(self) -> Tuple[Dict[str, List[DocumentIndex]], Dict[str, List[str]]]:
        chunk_root = self._root_path / "chunked_reports"
        vector_root = self._root_path / "vector_dbs_bge"
        if not chunk_root.exists():
            raise FileNotFoundError(f"Chunked reports directory not found: {chunk_root}")
        if not vector_root.exists():
            raise FileNotFoundError(f"Vector database directory not found: {vector_root}")

        category_docs: Dict[str, List[DocumentIndex]] = {}
        token_map: Dict[str, List[str]] = {}
        for category_dir in _iter_category_dirs(chunk_root):
            category = category_dir.name
            vector_dir = vector_root / category
            if not vector_dir.exists():
                continue

            documents: List[DocumentIndex] = []
            for chunk_path in category_dir.glob("*.json"):
                index_path = vector_dir / f"{chunk_path.stem}.faiss"
                doc_index = _load_document_index(chunk_path, index_path, category)
                if doc_index:
                    documents.append(doc_index)

            if documents:
                category_docs[category] = documents
                token_map[category] = _build_category_tokens(category)

        if not category_docs:
            raise RuntimeError("No vector databases were loaded. Run the pipeline first.")

        return category_docs, token_map

    def retrieve(self, question: str, topk: int) -> List[Tuple[float, Dict[str, object]]]:
        matched_category: Optional[str] = None
        matched_tokens: List[str] = []

        matched_category, matched_tokens = _match_category(question, self._token_map)
        if matched_category:
            if matched_tokens:
                print(f"[router] Matched category '{matched_category}' via tokens: {', '.join(matched_tokens)}")
            else:
                print(f"[router] Matched category '{matched_category}'")
        else:
            print("[router] No category tokens matched; searching all categories")

        if matched_category and matched_category in self._category_docs:
            categories_to_search = [matched_category]
        else:
            categories_to_search = list(self._category_docs.keys())

        query_vec = self._encoder.encode(question)
        candidate_topk = max(topk, self._rerank_candidates if self._use_reranker else topk)

        aggregated: List[Tuple[float, Dict[str, object]]] = []
        dimension_error: Optional[ValueError] = None
        for cat in categories_to_search:
            for doc_index in self._category_docs[cat]:
                try:
                    aggregated.extend(doc_index.search(query_vec, candidate_topk))
                except ValueError as err:
                    dimension_error = err
                    break
            if dimension_error:
                break

        if dimension_error:
            raise dimension_error

        aggregated.sort(key=lambda item: item[0], reverse=True)

        if self._use_reranker and self._reranker is not None:
            annotated: List[Tuple[float, Dict[str, object]]] = []
            for rank, (score, chunk) in enumerate(aggregated[:candidate_topk], start=1):
                chunk_copy = dict(chunk)
                chunk_copy["retrieval_rank"] = rank
                chunk_copy["retrieval_score"] = float(score)
                annotated.append((score, chunk_copy))
            if not annotated:
                return []
            return self._reranker.rerank(question, annotated, topk)

        trimmed: List[Tuple[float, Dict[str, object]]] = []
        for score, chunk in aggregated[:topk]:
            chunk_copy = dict(chunk)
            trimmed.append((score, chunk_copy))
        return trimmed

    @staticmethod
    def to_payload(question: str, topk: int, results: List[Tuple[float, Dict[str, object]]]) -> Dict[str, object]:
        payload = {
            "question": question,
            "topk": topk,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "results": [],
        }
        for rank, (score, chunk) in enumerate(results, start=1):
            record = {
                "rank": rank,
                "score": float(score),
                "category": chunk.get("category"),
                "sha1_name": chunk.get("sha1_name"),
                "chunk_file": chunk.get("chunk_file"),
                "chunk_index": chunk.get("chunk_index"),
                "page": chunk.get("page"),
                "text": chunk.get("text", ""),
            }
            if "retrieval_rank" in chunk:
                record["retrieval_rank"] = chunk.get("retrieval_rank")
            if "retrieval_score" in chunk:
                record["retrieval_score"] = chunk.get("retrieval_score")
            payload["results"].append(record)
        return payload

    @staticmethod
    def format_result(score: float, chunk: Dict[str, object]) -> str:
        page = chunk.get("page")
        sha1 = chunk.get("sha1_name")
        category = chunk.get("category")
        snippet = chunk.get("text", "")
        if "retrieval_score" in chunk:
            header = (
                "[retrieval_rank={orank} retrieval_score={os:.4f}] category={cat} doc={doc} page={page}"
            ).format(
                orank=chunk.get("retrieval_rank"),
                os=float(chunk.get("retrieval_score", score)),
                cat=category,
                doc=sha1,
                page=page,
            )
        else:
            header = f"[score={score:.4f}] category={category} doc={sha1} page={page}"
        return f"{header}\n{snippet}\n"
