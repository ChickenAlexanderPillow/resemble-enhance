import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torchaudio


def _extract_audio_streams(src: Path, work: Path) -> list[Path]:
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    # Try extract all audio streams as mono wavs
    out_paths: list[Path] = []
    try:
        # Probe number of audio streams
        if not ffprobe:
            # Assume one audio stream
            idxs = [0]
        else:
            p = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                capture_output=True,
                text=True,
                check=False,
            )
            idxs = [int(s) for s in p.stdout.strip().splitlines() if s.strip().isdigit()]
            if not idxs:
                idxs = [0]
        for i in idxs:
            out = work / f"stream{i}.wav"
            cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-map", f"0:a:{i}", "-ac", "1", str(out)]
            subprocess.run(cmd, check=True)
            out_paths.append(out)
    except Exception as e:
        raise RuntimeError(f"ffmpeg extraction failed: {e}")
    return out_paths


def _analyze_file(wav: Path) -> list[tuple[float, float]]:
    # Returns list of (time_seconds, zscore) for top spikes
    wav_t, sr = torchaudio.load(str(wav))
    if wav_t.dim() == 2 and wav_t.size(0) > 1:
        wav_t = wav_t.mean(0, keepdim=True)
    elif wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)
    x = wav_t[0]
    if x.numel() < 1000:
        return []
    d = torch.abs(x[1:] - x[:-1])
    mu = float(torch.mean(d))
    sd = float(torch.std(d)) + 1e-12
    z = (d - mu) / sd
    # Find candidates with z > 8
    mask = z > 8.0
    idxs = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    # De-duplicate within 0.5 s windows
    min_sep = int(sr * 0.5)
    keep = []
    last = -10**9
    for i in idxs:
        if i - last >= min_sep:
            keep.append(i)
            last = i
        if len(keep) >= 20:
            break
    out = []
    for i in keep:
        out.append((i / float(sr), float(z[i])))
    return out


def main():
    ap = argparse.ArgumentParser(description="Detect likely seam jumps in audio (spike in first-difference).")
    ap.add_argument("input", type=str, help="Input WAV/AIFF/MOV/MP4")
    args = ap.parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")
    work = Path(tempfile.mkdtemp(prefix="seamcheck_"))
    try:
        paths: list[Path]
        if inp.suffix.lower() in {".mov", ".mp4", ".mxf"}:
            import shutil
            paths = _extract_audio_streams(inp, work)
        else:
            paths = [inp]
        for p in paths:
            res = _analyze_file(p)
            print(f"Stream {p.name}:")
            if not res:
                print("  No strong spikes found.")
                continue
            for t, z in res:
                mm = int(t // 60)
                ss = int(t % 60)
                ms = int((t - int(t)) * 1000)
                print(f"  ~ {mm:02d}:{ss:02d}.{ms:03d}  z={z:.1f}")
    finally:
        pass  # keep temp for inspection if needed


if __name__ == "__main__":
    main()

