import subprocess
import sys
from pathlib import Path
from datetime import datetime


"""
Simple runner for Resemble Enhance.

Defaults:
- Input:  ./input_audio
- Output: ./output_audio
- Args:   --denoise_only --device cuda

Extra CLI args are forwarded as-is, so you can do for example:
  python run_enhancer.py --profile camera_sync
  python run_enhancer.py --target_sr 48000
"""


def _get_arg_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _compute_wav_duration_sec(path: Path) -> float | None:
    try:
        import wave

        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


def _safe_delete(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except Exception:
        return False


def _prune_empty_dirs(root: Path) -> None:
    # Remove empty directories bottom-up
    for p in sorted([d for d in root.rglob("*") if d.is_dir()], key=lambda x: len(x.parts), reverse=True):
        try:
            next(p.iterdir())
        except StopIteration:
            try:
                p.rmdir()
            except Exception:
                pass


def main() -> int:
    py = sys.executable or "python"
    in_dir = Path.cwd() / "input_audio"
    base_out_dir = Path.cwd() / "output_audio"
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out_dir / run_stamp

    cmd = [
        py,
        "-m",
        "resemble_enhance.enhancer",
        str(in_dir),
        str(out_dir),
        "--denoise_only",
        "--device",
        "cuda",
        *sys.argv[1:],  # pass-through additional flags
    ]

    print("Running Resemble Enhance with:")
    print(" ".join(cmd))
    print(f"Run output folder: {out_dir}")

    proc = subprocess.run(cmd)

    # Keep input files untouched. Optionally, sanity-check outputs exist.
    if proc.returncode == 0:
        suffix = _get_arg_value(sys.argv, "--suffix") or ".wav"
        candidates = [p for p in in_dir.rglob(f"*{suffix}") if p.is_file()]
        checked = 0
        missing = 0
        for src in candidates:
            rel = src.relative_to(in_dir)
            dst = out_dir / rel
            checked += 1
            if not dst.exists() or dst.stat().st_size <= 44:
                missing += 1
        print(f"Post-run: checked {checked} inputs; {missing} missing/invalid outputs.")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
