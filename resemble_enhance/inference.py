import logging
import time
import os
import math

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations
from torchaudio.functional import resample
from torchaudio.transforms import MelSpectrogram
from tqdm import trange

from .hparams import HParams

logger = logging.getLogger(__name__)


@torch.inference_mode()
def inference_chunk(model, dwav, sr, device, npad=441):
    assert model.hp.wav_rate == sr, f"Expected {model.hp.wav_rate} Hz, got {sr} Hz"
    del sr

    length = dwav.shape[-1]
    abs_max = dwav.abs().max().clamp(min=1e-7)

    assert dwav.dim() == 1, f"Expected 1D waveform, got {dwav.dim()}D"
    dwav = dwav.to(device)
    dwav = dwav / abs_max  # Normalize
    dwav = F.pad(dwav, (0, npad))
    hwav = model(dwav[None])[0].cpu()  # (T,)
    hwav = hwav[:length]  # Trim padding
    hwav = hwav * abs_max  # Unnormalize

    return hwav


def compute_corr(x, y):
    return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()


def compute_offset(chunk1, chunk2, sr=44100):
    """
    Args:
        chunk1: (T,)
        chunk2: (T,)
    Returns:
        offset: int, offset in samples such that chunk1 ~= chunk2.roll(-offset)
    """
    hop_length = sr // 200  # 5 ms resolution
    win_length = hop_length * 4
    n_fft = 2 ** (win_length - 1).bit_length()

    mel_fn = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
        f_min=0.0,
        f_max=sr // 2,
    )

    spec1 = mel_fn(chunk1).log1p()
    spec2 = mel_fn(chunk2).log1p()

    corr = compute_corr(spec1, spec2)  # (F, T)
    corr = corr.mean(dim=0)  # (T,)

    argmax = corr.argmax().item()

    if argmax > len(corr) // 2:
        argmax -= len(corr)

    offset = -argmax * hop_length

    return offset


def merge_chunks(
    chunks,
    chunk_length,
    hop_length,
    sr=44100,
    length=None,
    max_shift_ratio: float = 0.25,
    disable_align: bool = False,
):
    signal_length = (len(chunks) - 1) * hop_length + chunk_length
    overlap_length = chunk_length - hop_length
    signal = torch.zeros(signal_length, device=chunks[0].device)

    # Equal-power crossfade to reduce perceptual loops at boundaries
    t = torch.linspace(0, torch.pi / 2, overlap_length, device=chunks[0].device)
    # Head of a chunk should fade in (0->1) over the overlap, and tail should fade out (1->0)
    # Previous implementation accidentally applied the reverse at the head of the first chunk,
    # causing an unintended fade-down and audible choppiness when the first loud transient appears.
    fadein_overlap = torch.sin(t) ** 2   # 0 -> 1 over overlap
    fadeout_overlap = torch.cos(t) ** 2  # 1 -> 0 over overlap
    # Correct layout over the full chunk_length = hop_length + overlap_length:
    # - fadein: apply head fade-in over the first overlap, then ones
    # - fadeout: ones over the first hop region, then tail fade-out over the last overlap
    fadein = torch.cat([fadein_overlap, torch.ones(hop_length, device=chunks[0].device)])
    fadeout = torch.cat([torch.ones(hop_length, device=chunks[0].device), fadeout_overlap])

    for i, chunk in enumerate(chunks):
        start = i * hop_length
        end = start + chunk_length

        if len(chunk) < chunk_length:
            chunk = F.pad(chunk, (0, chunk_length - len(chunk)))

        if i > 0:
            if disable_align:
                offset = 0
            else:
                pre_region = chunks[i - 1][-overlap_length:]
                cur_region = chunk[:overlap_length]
                # Transient-safe alignment: avoid shifting when a sharp transient is detected
                def _has_sharp_transient(x):
                    try:
                        if x.numel() < 1024:
                            return False
                        d = (x[1:] - x[:-1]).abs()
                        mu = d.mean()
                        sd = d.std()
                        thr = mu + sd * 8.0
                        return bool((d > thr).any().item())
                    except Exception:
                        return False

                if _has_sharp_transient(pre_region) or _has_sharp_transient(cur_region):
                    offset = 0
                else:
                    offset = compute_offset(pre_region, cur_region, sr=sr)

                # Clamp offset to a configurable fraction of the overlap
                max_shift = int(overlap_length * max_shift_ratio)
                if offset > max_shift:
                    offset = max_shift
                elif offset < -max_shift:
                    offset = -max_shift

            # Apply alignment offset; keep sign consistent with compute_offset
            start -= offset
            end -= offset

        if i == 0:
            # First chunk: no head fade (already at start), tail should fade out
            chunk = chunk * fadeout
        elif i == len(chunks) - 1:
            # Last chunk: head should fade in from previous, no tail fade needed
            chunk = chunk * fadein
        else:
            # Middle chunks: head fade-in from previous and tail fade-out into next
            chunk = chunk * fadein * fadeout

        # Safely add chunk into the signal buffer with clamped indices
        write_start = max(start, 0)
        write_end = min(end, signal_length)

        if write_end > write_start:
            chunk_start = write_start - start
            chunk_end = chunk_start + (write_end - write_start)
            signal[write_start:write_end] += chunk[chunk_start:chunk_end]

    signal = signal[:length]

    return signal


def remove_weight_norm_recursively(module):
    for _, module in module.named_modules():
        try:
            remove_parametrizations(module, "weight")
        except Exception:
            pass


def inference(
    model,
    dwav,
    sr,
    device,
    chunk_seconds: float = 31.0,
    overlap_seconds: float = 1.0,
    align_max_shift_ratio: float = 0.25,
    disable_align: bool = False,
    progress_cb=None,
):
    remove_weight_norm_recursively(model)

    hp: HParams = model.hp

    dwav = resample(
        dwav,
        orig_freq=sr,
        new_freq=hp.wav_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

    del sr  # Everything is in hp.wav_rate now

    sr = hp.wav_rate

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    chunks = []
    # Optional progress reporting for external UIs
    report = os.environ.get("RESEMBLE_PROGRESS", "0") == "1"
    cur_file = os.environ.get("RESEMBLE_FILE", "")
    n_chunks = max(1, math.ceil(dwav.shape[-1] / hop_length))
    if report:
        print(f"PROGRESS START file={cur_file} n={n_chunks}")
    if progress_cb is not None:
        try:
            progress_cb("start", cur_file, 0, n_chunks)
        except Exception:
            pass

    for i, start in enumerate(trange(0, dwav.shape[-1], hop_length), start=1):
        chunks.append(inference_chunk(model, dwav[start : start + chunk_length], sr, device))
        # Proactive GPU cache trim to reduce fragmentation on long files
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
        if report:
            print(f"PROGRESS CHUNK file={cur_file} i={i} n={n_chunks}")
        if progress_cb is not None:
            try:
                progress_cb("chunk", cur_file, i, n_chunks)
            except Exception:
                pass

    hwav = merge_chunks(
        chunks,
        chunk_length,
        hop_length,
        sr=sr,
        length=dwav.shape[-1],
        max_shift_ratio=align_max_shift_ratio,
        disable_align=disable_align,
    )

    # Optional strict sanitizer to avoid NaN/Inf and extreme spikes
    try:
        import os as _os
        import torch as _t
        if not _t.isfinite(hwav).all():
            mask = ~_t.isfinite(hwav)
            hwav = hwav.clone()
            hwav[mask] = 0.0
        if _os.environ.get("RESEMBLE_STRICT_SANITIZE", "0") == "1":
            # Clamp extreme outliers relative to local RMS (~50 ms)
            k = max(16, int(sr * 0.05))
            pad = k // 2
            x = hwav
            xx = x.unsqueeze(0).unsqueeze(0)
            w = _t.ones(1, 1, k, dtype=xx.dtype, device=xx.device) / float(k)
            rms = _t.sqrt(_t.nn.functional.conv1d(xx * xx, w, padding=pad).squeeze() + 1e-12)
            thr = 8.0 * rms
            over = x.abs() > thr
            if over.any():
                hwav = x.sign() * thr.where(over, x.abs())
    except Exception:
        pass

    # Optional transient bypass: replace small windows with original to avoid choppy artifacts
    try:
        import os as _os
        import torch as _t

        def _parse_bypass_env():
            raw = _os.environ.get("RESEMBLE_BYPASS", "").strip()
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

        def _bypass_windows(proc: _t.Tensor, orig: _t.Tensor, sr: int, wins):
            try:
                if not wins:
                    return proc
                x = proc.clone(); y = orig
                n = min(x.numel(), y.numel())
                x = x[:n]; y = y[:n]
                ease = max(16, int(sr * 0.01))
                for (start_s, dur_s) in wins:
                    a = int(max(0, start_s * sr)); b = int(min(n, (start_s + dur_s) * sr))
                    if b <= a + 8:
                        continue
                    a0 = max(0, a - ease); b0 = min(n, b + ease)
                    mlen = b0 - a0
                    w = _t.linspace(0, 1, steps=mlen, dtype=x.dtype, device=x.device)
                    x[a0:b0] = x[a0:b0] * (1 - w) + y[a0:b0] * w
                return x
            except Exception:
                return proc

        def _auto_detect_transients(x: _t.Tensor, sr: int):
            try:
                d = (x[1:] - x[:-1]).abs()
                if d.numel() < 1000:
                    return []
                mu = _t.mean(d)
                sd = _t.std(d)
                thr = mu + sd * 8.0
                idxs = (d > thr).nonzero(as_tuple=False).flatten().tolist()
                # Merge nearby indices into windows ~200 ms each
                win_s = 0.2
                min_gap = int(sr * 0.4)
                wins = []
                last = -10**9
                for i in idxs:
                    if i - last >= min_gap:
                        t0 = max(0, i - int(sr * (win_s / 2))) / sr
                        wins.append((t0, win_s))
                        last = i
                    if len(wins) >= 6:
                        break
                return wins
            except Exception:
                return []

        # Manual bypass via env
        wins = _parse_bypass_env()
        if wins:
            hwav = _bypass_windows(hwav, dwav, sr, wins)

        # Auto bypass if requested
        if _os.environ.get("RESEMBLE_AUTO_BYPASS", "0") == "1":
            auto_wins = _auto_detect_transients(hwav, sr)
            if auto_wins:
                hwav = _bypass_windows(hwav, dwav, sr, auto_wins)
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.3f} s, {hwav.shape[-1] / elapsed_time / 1000:.3f} kHz")
    if report:
        print(f"PROGRESS END file={cur_file}")
    if progress_cb is not None:
        try:
            progress_cb("end", cur_file, n_chunks, n_chunks)
        except Exception:
            pass

    return hwav, sr
