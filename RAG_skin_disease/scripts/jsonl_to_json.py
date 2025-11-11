"""Convert a JSONL file (one JSON object per line) into a pretty-printed JSON array.

Formatting goals (to match LLaVA docs):
- Top-level is an array [ ... ]
- Each element (object) starts with two spaces of indentation inside the array
- Object internal indentation is 2 spaces as well

This script streams line-by-line to avoid high memory usage on large datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_jsonl_to_json(
    jsonl_path: Path,
    json_path: Path | None = None,
    array_indent: int = 2,
    object_indent: int = 2,
) -> int:
    """Convert JSONL -> JSON array with controlled indentation.

    Returns number of records written.
    """
    if json_path is None:
        json_path = jsonl_path.with_suffix(".json")

    count = 0
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("r", encoding="utf-8") as infh, json_path.open("w", encoding="utf-8") as outfh:
        outfh.write("[\n")
        pending_lines: list[str] | None = None
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
                # Only keep JSON objects for LLaVA dataset format
                continue

            dumped_lines = json.dumps(obj, ensure_ascii=False, indent=object_indent).splitlines()
            prefixed = [(" " * array_indent) + dl for dl in dumped_lines]

            if pending_lines is None:
                pending_lines = prefixed
            else:
                # Write previous object, adding comma to its closing line
                for dl in pending_lines[:-1]:
                    outfh.write(dl + "\n")
                outfh.write(pending_lines[-1] + ",\n")
                pending_lines = prefixed
                count += 1

        if pending_lines is not None:
            for dl in pending_lines:
                outfh.write(dl + "\n")
            count += 1
        outfh.write("]\n")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert JSONL (1 object per line) into a pretty JSON array.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source JSONL file.")
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON array (defaults to --input with .json suffix).")
    parser.add_argument("--array-indent", type=int, default=2, help="Indentation spaces for array elements (default: 2).")
    parser.add_argument("--object-indent", type=int, default=2, help="Indentation spaces within each object (default: 2).")
    args = parser.parse_args(argv)

    total = convert_jsonl_to_json(args.input, args.output, array_indent=args.array_indent, object_indent=args.object_indent)
    print(f"Exported {total} records to JSON array at {args.output or args.input.with_suffix('.json')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
