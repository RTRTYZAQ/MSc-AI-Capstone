#!/usr/bin/env python3
"""Count occurrences per `category` in a JSON array or JSONL file.

Usage:
  python scripts\count_categories.py --input llava_5000.json
  python scripts\count_categories.py --input data/something.jsonl --csv out.csv

The script tries to parse the input as a JSON array first; if that fails, it will
attempt to parse the file as newline-delimited JSON (JSONL).
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_records(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8")
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
            return
    except Exception:
        # fall through to JSONL
        pass

    # Try JSONL (each line JSON object)
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception as e:
            raise ValueError(f"Failed to parse JSON on line {i}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Count records per category in JSON/JSONL dataset")
    ap.add_argument("--input", "-i", required=True, help="Path to input JSON array or JSONL file")
    ap.add_argument("--csv", help="Optional path to dump CSV of category,count")
    ap.add_argument("--top", type=int, help="Show only top N categories")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"ERROR: input not found: {p}")
        raise SystemExit(2)

    cnt = Counter()
    total = 0
    for rec in iter_records(p):
        total += 1
        cat = rec.get("category") if isinstance(rec, dict) else None
        if not cat:
            cat = "<MISSING>"
        cnt[str(cat)] += 1

    uniques = len(cnt)
    print(f"Input: {p}")
    print(f"Total records: {total}")
    print(f"Distinct categories: {uniques}")
    print()

    most = cnt.most_common(args.top) if args.top else cnt.most_common()
    max_len = max((len(k) for k, _ in most), default=8)
    print(f"{'Category'.ljust(max_len)}  Count")
    print(f"{'-'*max_len}  -----")
    for k, v in most:
        print(f"{k.ljust(max_len)}  {v}")

    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", encoding="utf-8") as f:
            f.write("category,count\n")
            for k, v in most:
                # quote categories with commas
                cat = '"' + k.replace('"', '""') + '"' if ("," in k or "\n" in k) else k
                f.write(f"{cat},{v}\n")
        print(f"\nWrote CSV: {outp}")


if __name__ == "__main__":
    main()
