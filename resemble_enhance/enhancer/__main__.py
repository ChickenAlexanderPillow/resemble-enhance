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
        if args.denoise_only:
            hwav, sr = denoise(
                dwav=dwav,
                sr=sr,
                device=device,
                run_dir=args.run_dir,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
                align_max_shift_ratio=args.align_max_shift_ratio,
                align_disable=args.align_disable,
            )
        else:
            hwav, sr = enhance(
                dwav=dwav,
                sr=sr,
                device=device,
                nfe=args.nfe,
                solver=args.solver,
                lambd=args.lambd,
                tau=args.tau,
                run_dir=run_dir,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
                align_max_shift_ratio=args.align_max_shift_ratio,
                align_disable=args.align_disable,
            )
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

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path, hwav[None], sr)

    # Print completion message (cross-platform safe)
    elapsed_time = time.perf_counter() - start_time
    print(f"Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
