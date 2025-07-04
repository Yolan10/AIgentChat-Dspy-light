#!/usr/bin/env python3
"""Generate summary statistics from log files."""

import json
from collections import Counter
from pathlib import Path

LOG_DIR = Path("logs")
SYSTEM_LOG = LOG_DIR / "system.log"
TOKEN_LOG = LOG_DIR / "token_usage.json"


def load_system_log() -> Counter:
    """Return counts of messages in the structured system log."""
    counts: Counter[str] = Counter()
    if SYSTEM_LOG.exists():
        for line in SYSTEM_LOG.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = entry.get("message", "unknown")
            counts[msg] += 1
    return counts


def load_token_log() -> dict:
    """Return aggregated token usage across all runs."""
    totals = {"prompt": 0, "completion": 0}
    if TOKEN_LOG.exists():
        try:
            data = json.loads(TOKEN_LOG.read_text())
        except json.JSONDecodeError:
            data = {}
        for run in data.values():
            totals["prompt"] += int(run.get("prompt", 0))
            totals["completion"] += int(run.get("completion", 0))
    return totals


def main() -> None:
    counts = load_system_log()
    totals = load_token_log()

    print("System Log Event Counts:")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print("\nToken Usage Totals:")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
