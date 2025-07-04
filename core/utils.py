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


def get_usage_tokens(usage_meta: Any) -> tuple[int, int]:
    """Return prompt and completion token counts from usage metadata."""
    if not usage_meta:
        return 0, 0

    if isinstance(usage_meta, dict):
        prompt = (
            usage_meta.get("input_tokens")
            or usage_meta.get("prompt_tokens")
            or 0
        )
        completion = (
            usage_meta.get("output_tokens")
            or usage_meta.get("completion_tokens")
            or 0
        )
    else:
        prompt = (
            getattr(usage_meta, "input_tokens", None)
            or getattr(usage_meta, "prompt_tokens", 0)
        )
        completion = (
            getattr(usage_meta, "output_tokens", None)
            or getattr(usage_meta, "completion_tokens", 0)
        )

    return int(prompt or 0), int(completion or 0)

