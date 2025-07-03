import os
import json
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")
RUN_COUNTER_FILE = LOG_DIR / "run_counter.txt"


def ensure_logs_dir():
    LOG_DIR.mkdir(exist_ok=True)


def increment_run_number() -> int:
    ensure_logs_dir()
    if RUN_COUNTER_FILE.exists():
        run_no = int(RUN_COUNTER_FILE.read_text()) + 1
    else:
        run_no = 1
    RUN_COUNTER_FILE.write_text(str(run_no))
    return run_no


def extract_json_array(text: str):
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except Exception:
        return []

