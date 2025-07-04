import logging
import config

class ConsoleLogger:
    def __init__(self):
        self.logger = logging.getLogger("AIgentConsole")
        if not self.logger.handlers:
            level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.setLevel(level)
            self.logger.addHandler(handler)

    def log(self, message: str, level: str = "info"):
        log_method = getattr(self.logger, level, self.logger.info)
        log_method(message)

