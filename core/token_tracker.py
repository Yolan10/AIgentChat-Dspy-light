import json
from pathlib import Path
from typing import Dict

LOG_FILE = Path("logs/token_usage.json")

class TokenTracker:
    def __init__(self):
        self.data: Dict[str, Dict[str, int]] = {}

    def set_run(self, run_no: int):
        self.current_run = str(run_no)
        self.data[self.current_run] = {"prompt": 0, "completion": 0}

    def add_usage(self, prompt: int, completion: int):
        stats = self.data.setdefault(self.current_run, {"prompt": 0, "completion": 0})
        stats["prompt"] += prompt
        stats["completion"] += completion
        self.save()

    def save(self):
        LOG_FILE.parent.mkdir(exist_ok=True)
        LOG_FILE.write_text(json.dumps(self.data, indent=2))


# Global instance
tracker = TokenTracker()

