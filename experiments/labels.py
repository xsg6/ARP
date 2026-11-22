"""Join human correctness decisions to measured candidate outcomes by candidate ID."""
import argparse
import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def merge(validation, decisions, destination):
    with decisions.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"id", "correct"}.issubset(reader.fieldnames or []):
            raise ValueError("labels CSV needs id,correct")
        labels = {}
        for item in reader:
            key = item["id"]
            if not key or key in labels or item["correct"] not in {"0", "1"}:
                raise ValueError(f"invalid or duplicate decision for {key!r}")
            labels[key] = int(item["correct"])

    seen = set()
    # A hard link publishes the fully written file without replacing an existing run.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as out:
        tmp = Path(out.name)
    try:
        with validation.open(encoding="utf-8") as stream, tmp.open("w", encoding="utf-8") as out:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row["id"]
                if not isinstance(key, str) or not key or key in seen or row["passed"] not in (0, 1):
                    raise ValueError(f"invalid candidate {key!r}")
                seen.add(key)
                decision = labels.pop(key, None)
                if row["passed"] and decision is None:
                    raise ValueError(f"missing adjudication for passing candidate {key}")
                if not row["passed"] and decision == 1:
                    raise ValueError(f"failing candidate marked correct: {key}")
                correct = int(decision or 0)
                if row.get("correct") not in (None, correct):
                    raise ValueError(f"existing label conflicts for {key}")
                row["correct"] = correct
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        if labels or not seen:
            raise ValueError("unknown IDs in decisions, or no validation records")
        os.link(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "decisions", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    merge(args.validation, args.decisions, args.destination)
