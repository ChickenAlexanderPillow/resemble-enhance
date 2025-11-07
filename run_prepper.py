import subprocess
import sys
from pathlib import Path


# Simple Python runner for Premiere Prepper on Windows
# Adjust these paths and options as needed.

INPUT_ROOT = Path.cwd() / "Shoot" / "2025_11_05"
OUTPUT_ROOT = Path.cwd() / "Shoot" / "PREPPED"
CAMS = ["camA", "camB"]
MICS = "audio"

# Defaults mirroring your desired settings
TIMEBASE = 25
DEVICE = "cuda"  # or "cpu"
MAX_OFFSET = 5.0
VAD_AGGR = 2
MIN_SPEECH = 0.25
MIN_SILENCE = 0.80
HEAD_PAD = 0.60
TAIL_PAD = 1.00

# Enable robust behavior for long single-mic, multi-clip shoots
KEEP_SCRATCH = True
WRITE_MP4 = False
WRITE_MOV_PCM = True
DEBUG_MUX = True
AUTO_MUX_CORRECT = True
TRIM_ON = True
FORCE_CLEAN = True
STRICT_CLEAN = True
GLOBAL_SEARCH = True
DRIFT_CORRECT = True


def main() -> int:
    py = sys.executable or "python"
    cmd = [
        py,
        str(Path(__file__).with_name("premiere_prepper.py")),
        "--input",
        str(INPUT_ROOT),
        "--out",
        str(OUTPUT_ROOT),
        "--cams",
        *CAMS,
        "--mics",
        MICS,
        "--timebase",
        str(TIMEBASE),
        "--device",
        DEVICE,
        "--vad-aggr",
        str(VAD_AGGR),
        "--min-speech",
        str(MIN_SPEECH),
        "--min-silence",
        str(MIN_SILENCE),
        "--head-pad",
        str(HEAD_PAD),
        "--tail-pad",
        str(TAIL_PAD),
        "--max-offset",
        str(MAX_OFFSET),
    ]
    if KEEP_SCRATCH:
        cmd.append("--keep-scratch")
    if WRITE_MP4:
        cmd.append("--write-mp4")
    if WRITE_MOV_PCM:
        cmd.append("--write-mov-pcm")
    if DEBUG_MUX:
        cmd.append("--debug-mux")
    if AUTO_MUX_CORRECT:
        cmd.append("--auto-mux-correct")
    if TRIM_ON:
        cmd.append("--trim-on")
    if FORCE_CLEAN:
        cmd.append("--force-clean")
    if STRICT_CLEAN:
        cmd.append("--strict-clean")
    if GLOBAL_SEARCH:
        cmd.append("--global-search")
    if DRIFT_CORRECT:
        cmd.append("--drift-correct")

    print("Running Premiere Prepper with:")
    print(" ".join(cmd))

    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
