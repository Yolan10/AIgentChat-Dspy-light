import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import config

LOG_FILE = Path("logs/system.log")

class StructuredLogger:
    def __init__(self):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.file = LOG_FILE.open("a")

        # Configure console logging once
        self.logger = logging.getLogger("AIgent")
        if not self.logger.handlers:
            level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
            self.logger.setLevel(level)
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(self, message: str, level: str = "info", **kwargs: Any):
        entry = {"time": datetime.utcnow().isoformat(), "message": message, **kwargs}
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()

        log_method = getattr(self.logger, level, self.logger.info)
        log_method(f"{message}: {kwargs}")

