from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize PaDEL rerun status JSONL.")
    parser.add_argument("--status-jsonl", type=Path, default=Path("outputs/padel_cli_one_by_one/status.jsonl"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/padel_cli_one_by_one/summary.csv"))
    args = parser.parse_args()

    rows = []
    if args.status_jsonl.exists():
        with args.status_jsonl.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

    counts = Counter(row.get("classification", "unknown") for row in rows)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["classification", "count"])
        writer.writeheader()
        for key, value in sorted(counts.items()):
            writer.writerow({"classification": key, "count": value})

    print(f"Rows: {len(rows)}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
    print(f"Wrote: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
