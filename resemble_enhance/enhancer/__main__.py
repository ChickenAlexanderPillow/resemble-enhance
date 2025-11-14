import argparse
import os
import random
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from .inference import denoise, enhance


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", type=Path, help="Path to input audio folder")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to the enhancer run folder, if None, use the default model",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".wav",
        help="Audio file suffix",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation, recommended to use CUDA",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=None,
        help="Optional output sample rate (e.g., 48000). If omitted, preserves the input file's sample rate",
    )
    parser.add_argument(
        "--denoise_only",
        action="store_true",
        help="Only apply denoising without enhancement",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["camera_sync"],
        help="Preset for common workflows (e.g., camera_sync)",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=31.0,
        help="Chunk length in seconds",
    )
    parser.add_argument(
        "--overlap_seconds",
        type=float,
        default=1.0,
        help="Overlap between chunks in seconds",
    )
    parser.add_argument(
        "--align_max_shift_ratio",
        type=float,
        default=0.25,
        help="Maximum fraction of overlap allowed for alignment shift (0-1)",
    )
    parser.add_argument(
        "--align_disable",
        action="store_true",
        help="Disable cross-chunk alignment correction",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1.0,
        help="Denoise strength for enhancement (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="CFM prior temperature (0.0 to 1.0)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        choices=["midpoint", "rk4", "euler"],
        help="Numerical solver to use",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=64,
        help="Number of function evaluations",
    )
    parser.add_argument(
        "--parallel_mode",
        action="store_true",
        help="Shuffle the audio paths and skip the existing ones, enabling multiple jobs to run in parallel",
    )

    args = parser.parse_args()

    # Apply profile presets if requested
    if args.profile == "camera_sync":
        # Favor sync safety and seam robustness
        if args.target_sr is None:
            args.target_sr = 48000
        args.chunk_seconds = 31.0
        args.overlap_seconds = 1.0
        args.align_max_shift_ratio = 0.25
        args.align_disable = True

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available but --device is set to cuda, using CPU instead")
        device = "cpu"

    start_time = time.perf_counter()

    run_dir = args.run_dir

    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))

    if args.parallel_mode:
        random.shuffle(paths)

    if len(paths) == 0:
        print(f"No {args.suffix} files found in the following path: {args.in_dir}")
        return

    pbar = tqdm(paths)

    for path in pbar:
        out_path = args.out_dir / path.relative_to(args.in_dir)
        if args.parallel_mode and out_path.exists():
            continue
        pbar.set_description(f"Processing {out_path}")
        # Expose file path for progress reporting in inference
        os.environ["RESEMBLE_FILE"] = str(path)
        dwav, sr = torchaudio.load(path)
        orig_sr = sr
        orig_len = dwav.shape[-1]
        dwav = dwav.mean(0)
        # Per-file overrides: allow forcing single-chunk via env (diagnostics)
        _chunk_seconds = args.chunk_seconds
        _overlap_seconds = args.overlap_seconds
        _align_disable = args.align_disable
        try:
            import os as _os
            if _os.environ.get("RESEMBLE_FORCE_SINGLE_CHUNK", "0") == "1":
                dur_s = float(orig_len) / float(orig_sr if orig_sr else 1)
                _chunk_seconds = max(1.0, dur_s + 1.0)
                _overlap_seconds = 0.0
                _align_disable = True
        except Exception:
            pass

        # OOM-resilient wrapper: shrink chunk size and fallback to CPU if needed
        def _run_with_fallback(run_fn):
            import torch as _t
            cur_device = device
            cs = float(_chunk_seconds)
            ov = float(_overlap_seconds)
            max_retries = 4
            attempt = 0
            while True:
                try:
                    return run_fn(cur_device, cs, ov)
                except Exception as e:
                    if "CUDA out of memory" in str(e) or getattr(type(e), "__name__", "").lower().startswith("outofmemory"):
                        attempt += 1
                        cs = max(7.0, cs / 2.0)
                        ov = min(ov, cs / 4.0)
                        try:
                            if _t.cuda.is_available():
                                _t.cuda.empty_cache()
                        except Exception:
                            pass
                        if attempt >= max_retries and cur_device == "cuda":
                            cur_device = "cpu"
                            attempt = 0
                        if attempt > max_retries and cur_device == "cpu":
                            raise
                        continue
                    else:
                        raise

        if args.denoise_only:
            def _do(device_, cs_, ov_):
                return denoise(
                    dwav=dwav,
                    sr=sr,
                    device=device_,
                    run_dir=args.run_dir,
                    chunk_seconds=cs_,
                    overlap_seconds=ov_,
                    align_max_shift_ratio=args.align_max_shift_ratio,
                    align_disable=_align_disable,
                )

            hwav, sr = _run_with_fallback(_do)
        else:
            def _do(device_, cs_, ov_):
                # Use a fast configuration when requested or on CPU to avoid apparent hangs
                nfe = args.nfe
                solver = args.solver
                try:
                    fast_env = os.environ.get('RESEMBLE_FAST_ENHANCE','0') == '1'
                except Exception:
                    fast_env = False
                if fast_env:
                    nfe = min(int(getattr(args, 'nfe', 64) or 64), 16)
                    solver = "midpoint"
                if str(device_).lower() == "cpu":
                    nfe = min(int(getattr(args, 'nfe', 64) or 64), 8)
                    solver = "euler"
                return enhance(
                    dwav=dwav,
                    sr=sr,
                    device=device_,
                    nfe=nfe,
                    solver=solver,
                    lambd=args.lambd,
                    tau=args.tau,
                    run_dir=run_dir,
                    chunk_seconds=cs_,
                    overlap_seconds=ov_,
                    align_max_shift_ratio=args.align_max_shift_ratio,
                    align_disable=_align_disable,
                )

            hwav, sr = _run_with_fallback(_do)
        # Choose destination sample rate: fixed target or original
        dest_sr = args.target_sr if args.target_sr is not None else orig_sr

        # Resample back to the requested output rate to preserve timeline sync
        if sr != dest_sr:
            hwav = torchaudio.functional.resample(hwav, orig_freq=sr, new_freq=dest_sr)
            sr = dest_sr

        # Enforce exact sample count to avoid off-by-one drift
        if args.target_sr is not None:
            expected_len = round(orig_len * dest_sr / orig_sr)
        else:
            expected_len = orig_len

        cur_len = hwav.shape[-1]
        if cur_len > expected_len:
            hwav = hwav[:expected_len]
        elif cur_len < expected_len:
            hwav = torch.nn.functional.pad(hwav, (0, expected_len - cur_len))

        # Optional transient-safe blend: mix in a little original on sharp, loud mismatches
        try:
            import torch as _t
            base = dwav  # original mono at orig_sr
            if dest_sr != orig_sr:
                base = torchaudio.functional.resample(base, orig_freq=orig_sr, new_freq=dest_sr)
            if base.shape[-1] > expected_len:
                base = base[:expected_len]
            elif base.shape[-1] < expected_len:
                base = _t.nn.functional.pad(base, (0, expected_len - base.shape[-1]))

            def _adaptive_transient_blend(proc, orig, sr: int, strength: float = 0.5):
                try:
                    if proc.dim() != 1:
                        proc = proc.view(-1)
                    if orig.dim() != 1:
                        orig = orig.view(-1)
                    n = min(proc.numel(), orig.numel())
                    x = proc[:n]
                    y = orig[:n]
                    k = max(8, int(sr * 0.005))
                    pad = k // 2
                    d = (x - y).abs().unsqueeze(0).unsqueeze(0)
                    w = _t.ones(1, 1, k, dtype=d.dtype, device=d.device) / float(k)
                    d_s = _t.nn.functional.conv1d(d, w, padding=pad).squeeze()
                    med = _t.quantile(d_s, _t.tensor(0.5, device=d_s.device))
                    thr = med * 6.0
                    lvl = _t.maximum(x.abs(), y.abs())
                    lvl_thr = 10 ** (-12.0 / 20.0)
                    m1 = _t.clamp((d_s - thr) / (thr + 1e-8), 0.0, 1.0)
                    m2 = (lvl > lvl_thr).to(m1.dtype)
                    mask = (m1 * m2)
                    k2 = max(16, int(sr * 0.02))
                    pad2 = k2 // 2
                    w2 = _t.ones(1, 1, k2, dtype=d.dtype, device=d.device) / float(k2)
                    mask_s = _t.nn.functional.conv1d(mask.unsqueeze(0).unsqueeze(0), w2, padding=pad2).squeeze()
                    mask_s = _t.clamp(mask_s, 0.0, 1.0) * float(max(0.0, min(1.0, strength)))
                    out = x * (1.0 - mask_s) + y * mask_s
                    if out.numel() < proc.numel():
                        out = _t.cat([out, proc[out.numel():]], dim=0)
                    return out
                except Exception:
                    return proc

            if os.environ.get('RESEMBLE_DISABLE_TRANSIENT_BLEND', '0') != '1':
                hwav = _adaptive_transient_blend(hwav, base, sr, strength=0.9)

            # Optional explicit bypass windows via env var (RESEMBLE_BYPASS="40.0:0.3,12.5:0.2")
            def _parse_bypass_env():
                raw = os.environ.get("RESEMBLE_BYPASS", "").strip()
                out = []
                if raw:
                    for part in raw.split(','):
                        part = part.strip()
                        if not part:
                            continue
                        if ':' in part:
                            a, b = part.split(':', 1)
                        else:
                            a, b = part, '0.2'
                        try:
                            start = float(a); dur = float(b)
                            if start >= 0 and dur > 0:
                                out.append((start, dur))
                        except Exception:
                            pass
                return out

            def _bypass_time_windows(proc, orig, sr: int, windows):
                try:
                    if not windows:
                        return proc
                    x = proc.clone(); y = orig
                    n = min(x.numel(), y.numel()); x = x[:n]; y = y[:n]
                    for (start_s, dur_s) in windows:
                        a = int(max(0, start_s * sr)); b = int(min(n, (start_s + dur_s) * sr))
                        if b <= a + 8: continue
                        ease = max(16, int(sr * 0.01))
                        a0 = max(0, a - ease); b0 = min(n, b + ease)
                        mlen = b0 - a0
                        w = _t.linspace(0, 1, steps=mlen, dtype=x.dtype, device=x.device)
                        mask = w
                        x[a0:b0] = x[a0:b0] * (1 - mask) + y[a0:b0] * mask
                    return x
                except Exception:
                    return proc

            wins = _parse_bypass_env()
            if wins:
                hwav = _bypass_time_windows(hwav, base, sr, wins)
        except Exception:
            pass

        # Apply a small peak ceiling before saving to avoid transient clipping
        # during integer PCM encoding or container conversions (e.g., MOV).
        try:
            ceiling_db = -1.0  # dBFS
            ceiling = 10 ** (ceiling_db / 20.0)
            peak = float(hwav.abs().max().item()) if hasattr(hwav, "abs") else 0.0
            if peak > ceiling and peak > 0:
                hwav = hwav * (ceiling / peak)
        except Exception:
            # On any failure, proceed without ceiling rather than failing the run
            pass

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, hwav[None], sr)

    # Print completion message (cross-platform safe)
    elapsed_time = time.perf_counter() - start_time
    print(f"Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
