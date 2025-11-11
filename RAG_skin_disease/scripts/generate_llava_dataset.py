"""Generate LLaVA-style fine-tuning samples using DashScope's Qwen models."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple
from dashscope import MultiModalConversation
from src.bge_retrieval import BGERetrieval


class QwenClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int,
        retry_base_delay: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max(1, max_retries)
        self._retry_base_delay = max(0.5, retry_base_delay)

    def _parse_text(self, response: object) -> str:
        output = getattr(response, "output", None)
        if output is None:
            raise RuntimeError("dashscope response missing output field")
        choices = getattr(output, "choices", None)
        if not choices:
            raise RuntimeError("dashscope response contains no choices")
        message = choices[0].message
        if isinstance(message, dict):
            content = message.get("content", [])
        else:
            content = getattr(message, "content", [])
        texts: List[str] = []
        for item in content or []:
            text = item.get("text") if isinstance(item, dict) else None
            if text:
                texts.append(text)
        if not texts:
            raise RuntimeError("dashscope response does not include text content")
        return "\n".join(texts).strip()

    def generate(self, messages: List[Dict[str, object]]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = MultiModalConversation.call(
                    api_key=self._api_key,
                    model=self._model,
                    messages=messages,
                    stream=False,
                )
                return self._parse_text(response)
            except Exception as error:  # noqa: BLE001
                last_error = error
                wait_time = self._retry_base_delay * (2 ** (attempt - 1))
                time.sleep(wait_time)
        detail = f": {last_error}" if last_error else ""
        raise RuntimeError(f"Failed to obtain response from DashScope{detail}") from last_error


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLaVA fine-tuning data using DashScope Qwen outputs.")
    parser.add_argument("--image-root", type=Path, required=False, default=Path("DermNet/train"), help="Root directory whose subfolders contain disease images.")
    parser.add_argument("--relative-to", type=Path, default=Path(""), help="Base directory for emitting relative image paths (defaults to --image-root).")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png", help="Comma-separated list of image filename extensions to include.")
    parser.add_argument("--dataset-size", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--rag-root", type=Path, default=Path("data/skin_set"), help="Location of the skin-set vector stores and chunks.")
    parser.add_argument("--min-score", type=float, default=0.5, help="Minimum reranker score to keep a chunk.")
    parser.add_argument("--max-chunks", type=int, default=2, help="Maximum number of chunks to keep per question after filtering.")
    parser.add_argument("--retrieval-topk", type=int, default=6, help="Initial top-k to request from retrieval before filtering.")
    parser.add_argument("--max-retries", type=int, default=4, help="Maximum retries per API call.")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="Base delay for exponential backoff between retries (seconds).")
    parser.add_argument("--model", type=str, default="qwen3-vl-plus", help="DashScope model name to use.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output", type=Path, default=Path("data/llava_finetune_dataset.jsonl"), help="Path to the JSONL file to write (1 JSON record per line).")
    parser.add_argument("--dry-run", action="store_true", help="Skip DashScope calls and fill gpt outputs with empty strings (useful for debugging).")
    parser.add_argument("--json-output", type=Path, default=None, help="Path for the JSON array output (defaults to --output with .json suffix).")
    return parser.parse_args(argv)


def build_image_catalog(image_root: Path, extensions: Sequence[str]) -> Dict[str, List[Path]]:
    catalog: Dict[str, List[Path]] = {}
    extensions_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file():
            continue
        if extensions_set and image_path.suffix.lower() not in extensions_set:
            continue
        category = image_path.parent.name or image_root.name
        if category.lower().endswith(" photos"):
            category = category[: -len(" photos")].rstrip()
        catalog.setdefault(category, []).append(image_path)
    if not catalog:
        raise RuntimeError(f"No images discovered under {image_root}.")
    return catalog


def build_sampling_pools(
    catalog: Dict[str, List[Path]],
    rng: random.Random,
) -> Dict[str, deque[Path]]:
    pools: Dict[str, deque[Path]] = {}
    for category, paths in catalog.items():
        # Shuffle in-place to avoid creating a duplicate list in memory
        rng.shuffle(paths)
        pools[category] = deque(paths)
    return pools


def ensure_relative_path(path: Path, base: Path, prefix: Optional[str] = None) -> str:
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        rel = Path(path.name)
    if prefix:
        rel = Path(prefix) / rel
    return PurePosixPath(rel).as_posix()


def _load_seen_from_jsonl(path: Path) -> Tuple[set[str], int, Optional[str]]:
    """Load existing JSONL records, returning (seen_images, count, last_category).

    Robust to occasional malformed/partial last line: such lines are skipped with a warning.
    """
    seen: set[str] = set()
    count = 0
    last_category: Optional[str] = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] Skipping unparsable JSONL line {ln}: {e}")
                    continue
                if isinstance(obj, dict):
                    img = obj.get("image")
                    if isinstance(img, str):
                        seen.add(img)
                    # Read explicit category field only (newer files)
                    cat = obj.get("category")
                    if isinstance(cat, str):
                        last_category = cat
                    count += 1
    except FileNotFoundError:
        pass
    return seen, count, last_category


def convert_jsonl_to_json(jsonl_path: Path, json_path: Path) -> int:
    """Convert JSONL file into a pretty-printed JSON array file.

    Streaming approach: parse each line individually and write comma-delimited array to avoid high memory usage.
    Returns the number of records written.
    """
    count = 0
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("r", encoding="utf-8") as infh, json_path.open("w", encoding="utf-8") as outfh:
        outfh.write("[\n")
        pending_lines: Optional[list[str]] = None
        for ln, line in enumerate(infh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] Skipping unparsable JSONL line {ln}: {e}")
                continue
            if not isinstance(obj, dict):
                continue

            # Dump object with 2-space indent, then add 2 spaces for array-indent
            dumped_lines = json.dumps(obj, ensure_ascii=False, indent=2).splitlines()
            prefixed = ["  " + dl for dl in dumped_lines]

            if pending_lines is None:
                pending_lines = prefixed
            else:
                # Write previous object and append a trailing comma on its closing line
                for dl in pending_lines[:-1]:
                    outfh.write(dl + "\n")
                outfh.write(pending_lines[-1] + ",\n")
                pending_lines = prefixed
                count += 1

        # Flush last pending object without a trailing comma
        if pending_lines is not None:
            for dl in pending_lines:
                outfh.write(dl + "\n")
            count += 1
        outfh.write("]\n")
    return count


def gather_chunks(
    retrieval: BGERetrieval,
    question: str,
    min_score: float,
    max_chunks: int,
    retrieval_topk: int,
) -> List[Tuple[float, Dict[str, object]]]:
    results = retrieval.retrieve(question, retrieval_topk)
    if not results:
        return []
    filtered = [(score, chunk) for score, chunk in results if score >= min_score]
    if not filtered:
        filtered = results[:1]
    return filtered[:max_chunks]


def build_relevant_knowledge(
    retrieval: BGERetrieval,
    category: str,
    min_score: float,
    max_chunks: int,
    retrieval_topk: int,
) -> str:
    disease_question = f"What is {category}?"
    treatment_question = f"What are the recommended treatment options for {category}?"
    disease_chunks = gather_chunks(retrieval, disease_question, min_score, max_chunks, retrieval_topk)
    treatment_chunks = gather_chunks(retrieval, treatment_question, min_score, max_chunks, retrieval_topk)
    payload = {
        "disease": [
            {
                "question": disease_question,
                "confidence": round(float(score), 4),
                "text": " ".join(str(chunk.get("text", "")).split()),
            }
            for score, chunk in disease_chunks
        ],
        "treatment": [
            {
                "question": treatment_question,
                "confidence": round(float(score), 4),
                "text": " ".join(str(chunk.get("text", "")).split()),
            }
            for score, chunk in treatment_chunks
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def precompute_knowledge_for_categories(
    retrieval: BGERetrieval,
    categories: Sequence[str],
    min_score: float,
    max_chunks: int,
    retrieval_topk: int,
) -> Dict[str, str]:
    """Compute relevant knowledge once for each category.

    Returns a dict category(normalized) -> knowledge_json_string.
    """
    result: Dict[str, str] = {}
    for cat in categories:
        norm = cat.replace("_", " ")
        try:
            result[norm] = build_relevant_knowledge(
                retrieval=retrieval,
                category=norm,
                min_score=min_score,
                max_chunks=max_chunks,
                retrieval_topk=retrieval_topk,
            )
        except Exception as e:  # noqa: BLE001
            # Fallback to empty structure to keep pipeline running
            fallback = {"disease": [], "treatment": []}
            result[norm] = json.dumps(fallback, ensure_ascii=False)
            print(f"[warn] Failed to build knowledge for '{norm}': {e}")
    return result


def build_human_prompt(category: str, relevant_knowledge: str) -> str:
    return (
        "<image>\n"
        f"The image is from the dermatology category: {category}.\n"
        "Please generate a diagnostic report based on the skin images uploaded by the patient and the relevant knowledge about the skin condition provided. \n"
        f"Relevant knowledge: {relevant_knowledge}\n"
        "Please output strictly in JSON format, without any extra explanation:\n"
        "{{\n"
        '  "Disease Name": "<predicted disease name>",\n'
        '  "Symptom Description": "<simple description of patient\'s skin condition symptoms>",\n'
        '  "Treatment Plan Recommendation": "<given this patient\'s skin condition and the relevant knowledge about this dermatological disorder, please provide your treatment plan recommendation.>"\n'
        "}}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dataset_size <= 0:
        raise ValueError("--dataset-size must be positive")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")
    rng = random.Random(args.seed)
    extensions = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
    image_root = args.image_root.resolve()
    relative_base = (args.relative_to or args.image_root).resolve()
    relative_prefix: Optional[str] = None
    if args.relative_to is None:
        relative_prefix = args.image_root.resolve().name
    retrieval = BGERetrieval(
        root_path=args.rag_root.resolve(),
        use_reranker=True,
        rerank_candidates=max(args.retrieval_topk, args.max_chunks * 2),
    )
    catalog = build_image_catalog(image_root, extensions)
    total_images = sum(len(paths) for paths in catalog.values())
    if total_images == 0:
        raise RuntimeError("No images available for sampling")
    sample_size = min(args.dataset_size, total_images)

    # Default resume behavior: if output exists, resume; otherwise start fresh
    resume_mode = args.output.exists()

    # Resume support for JSONL: load seen images, existing count, and last category
    seen_images: set[str] = set()
    existing_count = 0
    last_category_from_file: Optional[str] = None
    if resume_mode:
        print(f"[resume] Output exists at {args.output}; resuming by default.")
        seen_images, existing_count, last_category_from_file = _load_seen_from_jsonl(args.output)

    # Determine remaining needed samples when resuming
    remaining_needed = max(0, sample_size - existing_count)
    if remaining_needed == 0:
        print(f"Nothing to do: existing {existing_count} entries >= requested {sample_size}.")
        # Always export JSON array by default
        json_path = args.json_output or args.output.with_suffix(".json")
        total = convert_jsonl_to_json(args.output, json_path)
        print(f"Exported {total} records to JSON array at {json_path}.")
        return 0
    # Prepare category list (keys from catalog) and precompute per-category knowledge once
    category_list = list(catalog.keys())
    knowledge_cache: Dict[str, str] = precompute_knowledge_for_categories(
        retrieval=retrieval,
        categories=category_list,
        min_score=args.min_score,
        max_chunks=args.max_chunks,
        retrieval_topk=args.retrieval_topk,
    )

    pools = build_sampling_pools(catalog, rng)
    # Free memory held by the original catalog (no longer needed after pools are built)
    catalog.clear()
    del catalog
    active_categories: deque[str] = deque(
        sorted(category for category, queue in pools.items() if queue)
    )
    # If resuming and we know the last used category, continue from the NEXT category
    if resume_mode and last_category_from_file and (last_category_from_file in active_categories):
        try:
            items = list(active_categories)
            idx = items.index(last_category_from_file)
            new_order = items[idx + 1 :] + items[: idx + 1]
            active_categories = deque(new_order)
            print(f"[resume] Continuing round-robin after category '{last_category_from_file}'.")
        except ValueError:
            pass
    client = None if args.dry_run else QwenClient(
        api_key=api_key,
        model=args.model,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
    )
    # Prepare output file handle: append if resuming, else overwrite
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if resume_mode else "w"
    processed = 0
    with args.output.open(file_mode, encoding="utf-8") as outfh:
        while active_categories and processed < remaining_needed:
            category = active_categories.popleft()
            queue = pools.get(category)
            if not queue:
                continue
            normalized_category = category.replace("_", " ")
            success = False
            attempts = 0
            while queue and not success:
                image_path = queue.popleft()
                attempts += 1
                # Build relative path first and skip if already generated
                rel_image = ensure_relative_path(image_path, relative_base, prefix=relative_prefix)
                if rel_image in seen_images:
                    continue
                # Use precomputed knowledge (always computed at startup)
                knowledge = knowledge_cache.get(normalized_category, json.dumps({"disease": [], "treatment": []}, ensure_ascii=False))
                human_value = build_human_prompt(normalized_category, knowledge)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"image": str(image_path.resolve())},
                            {"text": human_value},
                        ],
                    }
                ]
                try:
                    if args.dry_run:
                        gpt_value = ""
                    else:
                        gpt_value = client.generate(messages)
                except Exception as error:  # noqa: BLE001
                    print(
                        f"[skip] Failed to generate response for '{category}' (attempt {attempts}): {error}"
                    )
                    continue
                record = {
                    "id": str(uuid.uuid4()),
                    "category": normalized_category,
                    "image": rel_image,
                    "conversations": [
                        {"from": "human", "value": human_value},
                        {"from": "gpt", "value": gpt_value},
                    ],
                }
                # Append one JSON record per line and flush to reduce loss on interruption
                outfh.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfh.flush()
                seen_images.add(rel_image)
                processed += 1
                success = True
                print(f"[{processed}/{remaining_needed}] Prepared sample for category '{category}'.")
            # Re-queue policy (mutually exclusive):
            # - success and queue not empty: put category to the end to continue round-robin
            # - not success but queue not empty: also requeue to try remaining images later
            # - queue empty: drop category (and warn if never succeeded)
            if success:
                if queue:
                    active_categories.append(category)
            else:
                if queue:
                    active_categories.append(category)
                else:
                    print(f"[warn] Exhausted category '{category}' without a successful sample.")
    if processed < remaining_needed:
        print(
            f"Warning: only generated {processed} samples out of requested {remaining_needed}."
        )
    else:
        print(f"Saved {existing_count + processed} total samples to {args.output}.")

    # Always convert JSONL to JSON array by default
    json_path = args.json_output or args.output.with_suffix(".json")
    total = convert_jsonl_to_json(args.output, json_path)
    print(f"Exported {total} records to JSON array at {json_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
