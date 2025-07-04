import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

LOG_FILE = Path("logs/system.log")

class StructuredLogger:
    def __init__(self):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.file = LOG_FILE.open("a")

    def log(self, message: str, **kwargs: Any):
        entry = {"time": datetime.utcnow().isoformat(), "message": message, **kwargs}
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

