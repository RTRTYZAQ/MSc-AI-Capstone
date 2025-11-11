"""Ad-hoc script to simulate RAG lookup over the skin_set corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.bge_retrieval import BGERetrieval

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG simulation over the skin_set corpus.")
    parser.add_argument("--question", type=str, help="Question to ask. If omitted, use --interactive mode.")
    parser.add_argument("--root", type=Path, default="./data/skin_set", help="Root path of the skin dataset.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top chunks to display.")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive mode (reads questions from stdin).")
    parser.add_argument("--output", type=Path, default="./rag_results/results.json", help="Optional path to write results as JSON.")
    parser.add_argument("--rerank", action="store_true", help="Enable BGE reranker for the final top-k results.")
    parser.add_argument("--rerank-candidates", type=int, default=20, help="Number of retrieval candidates to feed into the reranker before trimming to top-k.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    retrieval = BGERetrieval(
        root_path=args.root,
        use_reranker=args.rerank,
        rerank_candidates=max(args.topk, args.rerank_candidates),
    )

    def run_query(question: str) -> None:
        effective_topk = args.topk if args.rerank else args.rerank_candidates
        results = retrieval.retrieve(question, effective_topk)

        if not results:
            print("No results found.")
            return

        if args.output:
            payload = retrieval.to_payload(question, effective_topk, results)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Wrote {len(results)} results to {args.output}")
        else:
            for score, chunk in results:
                print(retrieval.format_result(score, chunk))

    if args.interactive:
        while True:
            question = input("Question> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            run_query(question)
    else:
        if not args.question:
            print("Please provide --question or enable --interactive mode.")
            return 1
        run_query(args.question)

    return 0


if __name__ == "__main__":
    sys.exit(main())
