import os
import sys
from pathlib import Path


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    os.environ["PATH"] = str(path) + os.pathsep + current if current else str(path)


base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
ffmpeg_dir = base / "ffmpeg_bin"
if ffmpeg_dir.exists():
    _prepend_path(ffmpeg_dir)
