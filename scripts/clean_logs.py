#!/usr/bin/env python3
"""Archive or clear log files from the logs directory."""

from pathlib import Path
import shutil

LOG_DIR = Path("logs")
ARCHIVE_DIR = LOG_DIR / "archive"


def archive_logs():
    """Move all log files into the archive directory."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in LOG_DIR.iterdir():
        if path.name in {".gitkeep", "archive"}:
            continue
        dest = ARCHIVE_DIR / path.name
        shutil.move(str(path), dest)
        print(f"Archived {path} -> {dest}")


if __name__ == "__main__":
    archive_logs()
