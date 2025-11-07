import argparse
import json
import logging
import math
import os
import time
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Using CLI invocation for Resemble Enhance (per user request)


# == SECTION: DATA STRUCTURES ==

@dataclass
class VideoClip:
    path: Path
    camera: str
    order_index: int


@dataclass
class MicClip:
    path: Path
    name: str


@dataclass
class AlignedTrack:
    path: Path
    start_offset_s: float  # placement offset on timeline (includes camera A/V offset and trim head)
    duration_s: float      # duration of this audio segment (seconds)


@dataclass
class SequenceSpec:
    name: str
    video: Path
    timebase: int
    duration_frames: int
    aligned_wavs: List[AlignedTrack] = field(default_factory=list)
    scratch_wav: Optional[Path] = None
    audio_offset_s: float = 0.0


# == SECTION: LOGGING ==

def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "premiere_prepper.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)


# == SECTION: DISCOVERY ==

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".mts", ".m2ts"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wma"}


def discover_clips(root: Path, cam_dirs: List[str]) -> List[VideoClip]:
    clips: List[VideoClip] = []
    for cam in cam_dirs:
        cam_path = root / cam
        if not cam_path.exists():
            logging.warning(f"Camera dir missing: {cam_path}")
            continue
        files = [p for p in cam_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        def sort_key(p: Path):
            try:
                c = p.stat().st_ctime
            except Exception:
                c = 0
            try:
                m = p.stat().st_mtime
            except Exception:
                m = 0
            return (c if c else m, m)
        files.sort(key=sort_key)
        for idx, f in enumerate(files):
            clips.append(VideoClip(path=f, camera=cam, order_index=idx))
    return clips


def discover_mics(root: Path, mic_dir: str) -> List[MicClip]:
    mroot = root / mic_dir
    if not mroot.exists():
        logging.warning(f"Mic dir missing: {mroot}")
        return []
    files = [p for p in mroot.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    files.sort()
    return [MicClip(path=p, name=p.stem) for p in files]


# == SECTION: AUDIO UTILS (PLACEHOLDER) ==

def extract_camera_audio_ffmpeg(video_path: Path, out_wav: Path, sr: int = 48000) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    logging.info(f"Extracting scratch audio: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        logging.error(f"ffmpeg error extracting audio from {video_path}: {proc.stderr.decode(errors='ignore')}")
        raise RuntimeError("Failed to extract camera audio with ffmpeg")


def load_mono_48k(path: Path) -> Tuple[np.ndarray, int]:
    try:
        x, sr = _read_wav_mono_int16(path)
        if sr != 48000:
            x = resample_if_needed(x, sr, 48000)
            sr = 48000
        return x, sr
    except Exception:
        # Fallback: transcode via ffmpeg to 48k mono s16le then read
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp48k.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(tmp)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if proc.returncode != 0 or not tmp.exists():
                raise
            x, sr = _read_wav_mono_int16(tmp)
            return x, sr


def resample_if_needed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    if len(x) == 0:
        return x
    ratio = sr_out / sr_in
    n_out = int(round(len(x) * ratio))
    t_in = np.linspace(0, 1, num=len(x), endpoint=False)
    t_out = np.linspace(0, 1, num=n_out, endpoint=False)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y


def bandpass_voice(x: np.ndarray, sr: int) -> np.ndarray:
    if len(x) == 0:
        return x
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mask = (freqs >= 100.0) & (freqs <= 3000.0)
    X *= mask.astype(X.dtype)
    y = np.fft.irfft(X, n=len(x)).astype(np.float32)
    return y


def rms_normalize(x: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    if len(x) == 0:
        return x
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    target = 10 ** (target_db / 20.0)
    if rms < 1e-6:
        return x
    gain = target / rms
    return (x * gain).astype(np.float32)


def clean_with_resemble(wav_in: Path, wav_out: Path, device: str = "cpu") -> None:
    """Call Resemble Enhance via its CLI module (directory in/out).

    Stages single file into a temp input directory and runs:
      python -m resemble_enhance.enhancer <input_dir> <output_dir> --denoise_only --device <device>
    Then copies the corresponding output WAV to wav_out.
    """
    import tempfile, shutil
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    py = sys.executable or "python"
    base = wav_in.name
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"; out_dir = Path(td) / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged = in_dir / base
        shutil.copy2(str(wav_in), str(staged))

        def _run(dev: str):
            cmd = [py, "-m", "resemble_enhance.enhancer", str(in_dir), str(out_dir), "--denoise_only", "--device", dev]
            logging.info(f"Cleaning mic (CLI): {' '.join(cmd)}")
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)

        proc = _run(device)
        if proc.returncode != 0 and device.lower() == "cuda":
            logging.warning("Resemble CLI failed on CUDA; retrying on CPU...\n" + proc.stderr.decode(errors='ignore'))
            proc = _run("cpu")
        if proc.returncode != 0:
            stderr = proc.stderr.decode(errors='ignore')
            logging.error(f"Resemble CLI failed: {stderr}")
            raise RuntimeError("Resemble Enhance CLI failed")

        # Output file should have same basename; otherwise pick first WAV
        out_file = out_dir / base
        if not out_file.exists():
            outs = list(out_dir.rglob("*.wav"))
            if not outs:
                raise RuntimeError("Resemble Enhance CLI produced no wav output")
            out_file = outs[0]
        shutil.copy2(str(out_file), str(wav_out))

def _read_wav_mono_int16(path: Path) -> Tuple[np.ndarray, int]:
    import wave
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        frames = w.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV supported by internal reader")
    data = np.frombuffer(frames, dtype=np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    x = (data.astype(np.float32) / 32768.0).copy()
    return x, framerate

def _write_wav_mono_int16(path: Path, data: np.ndarray, sr: int) -> None:
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(data, -1.0, 1.0)
    s16 = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(s16.tobytes())

def ffprobe_streams(path: Path) -> list:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-print_format", "json",
        str(path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        return []
    try:
        import json as _json
        obj = _json.loads(proc.stdout.decode())
        return obj.get("streams", [])
    except Exception:
        return []

def _read_wav_mono_any(path: Path) -> Tuple[np.ndarray, int]:
    """Robust reader: tries soundfile for arbitrary WAV, else transcodes to 16-bit PCM via ffmpeg.
    Returns float32 mono -1..1 and sample rate.
    """
    try:
        import soundfile as sf  # type: ignore
        y, sr = sf.read(str(path), always_2d=True)
        y = y.astype(np.float32)
        if y.shape[1] > 1:
            y = y.mean(axis=1)
        else:
            y = y[:, 0]
        return y, int(sr)
    except Exception:
        # Fallback: ffmpeg transcode to s16le mono then read
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "tmp_s16.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(path),
                "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(tmp)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if proc.returncode != 0 or not tmp.exists():
                raise
            return _read_wav_mono_int16(tmp)


def probe_av_offset_seconds(video_path: Path) -> float:
    """Return audio_start_time - video_start_time (seconds) using ffprobe.
    Positive means audio starts later than video and should be delayed by this amount.
    """
    def _probe_start_time(select: str) -> Optional[float]:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", select,
            "-show_entries", "stream=start_time",
            "-of", "default=nw=1:nk=1",
            str(video_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        if proc.returncode != 0:
            return None
        txt = proc.stdout.decode().strip()
        try:
            return float(txt)
        except Exception:
            return None

    a_start = _probe_start_time("a:0")
    v_start = _probe_start_time("v:0")
    if a_start is None or v_start is None:
        return 0.0
    return a_start - v_start


def gcc_phat_offset(x: np.ndarray, y: np.ndarray, sr: int, max_offset_s: float) -> Tuple[float, float]:
    n = min(len(x), len(y))
    if n <= 0:
        return 0.0, 0.0
    x = x[:n]
    y = y[:n]
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    R = X * np.conj(Y)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n)
    # shift
    cc = np.concatenate((cc[-(n // 2):], cc[: (n - n // 2)]))
    max_lag = int(round(max_offset_s * sr))
    center = len(cc) // 2
    lo = max(center - max_lag, 0)
    hi = min(center + max_lag + 1, len(cc))
    window = cc[lo:hi]
    peak_idx = np.argmax(window) + lo
    lag = peak_idx - center
    conf = float(np.max(window) / (np.mean(np.abs(window)) + 1e-12))
    offset_s = -lag / sr
    return offset_s, conf


def estimate_drift(x: np.ndarray, y: np.ndarray, sr: int, max_offset_s: float = 1.0) -> float:
    if len(x) < sr * 5 or len(y) < sr * 5:
        return 0.0
    win = int(sr * 10.0)
    x0 = x[:win]
    y0 = y[:win]
    o0, _ = gcc_phat_offset(x0, y0, sr, max_offset_s)
    x1 = x[-win:]
    y1 = y[-win:]
    o1, _ = gcc_phat_offset(x1, y1, sr, max_offset_s)
    t0 = 5.0
    t1 = (len(x) / sr) - 5.0
    if t1 <= t0:
        return 0.0
    slope = (o1 - o0) / (t1 - t0)
    return slope


def apply_offset_and_drift(mic: np.ndarray, sr: int, target_len: int, offset_s: float, drift_model: Optional[float] = None) -> np.ndarray:
    n_out = target_len
    t = np.arange(n_out) / sr
    slope = drift_model or 0.0
    src_t = t - offset_s - slope * t
    src_idx = src_t * sr
    y = np.interp(src_idx, np.arange(len(mic)), mic, left=0.0, right=0.0).astype(np.float32)
    return y


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def coarse_offset_via_xcorr(scratch: np.ndarray, mic: np.ndarray, sr: int, down_sr: int = 8000) -> Tuple[int, float]:
    """Find best starting index of scratch within a long mic via FFT cross-correlation.
    Returns (start_idx_samples_at_48k, confidence).
    """
    if len(scratch) == 0 or len(mic) == 0:
        return 0, 0.0
    # Downsample for speed
    scr_ds = resample_if_needed(scratch, sr, down_sr)
    mic_ds = resample_if_needed(mic, sr, down_sr)
    # Band-limit and normalize
    scr_ds = bandpass_voice(scr_ds, down_sr)
    mic_ds = bandpass_voice(mic_ds, down_sr)
    scr_ds = rms_normalize(scr_ds)
    mic_ds = rms_normalize(mic_ds)
    # Remove DC
    scr_ds = scr_ds - float(np.mean(scr_ds))
    mic_ds = mic_ds - float(np.mean(mic_ds))

    n = len(mic_ds) + len(scr_ds) - 1
    nfft = _next_pow2(n)
    # Cross-correlation by convolution with time-reversed scratch
    MIC = np.fft.rfft(mic_ds, nfft)
    SCR = np.fft.rfft(scr_ds[::-1], nfft)
    cc = np.fft.irfft(MIC * SCR, nfft)[:n]
    # Valid lags where scratch fully overlaps mic start positions [0 .. len(mic)-len(scr)]
    valid_start_lo = 0
    valid_start_hi = max(0, len(mic_ds) - len(scr_ds))
    # The index in cc corresponding to a given start k is k + len(scr)-1
    lo = valid_start_lo + len(scr_ds) - 1
    hi = valid_start_hi + len(scr_ds) - 1
    if hi <= lo:
        peak_pos = int(np.argmax(cc))
    else:
        window = cc[lo:hi+1]
        peak_pos = int(np.argmax(window)) + lo
    start_k_ds = peak_pos - (len(scr_ds) - 1)
    # Confidence: peak relative to mean abs in window
    denom = float(np.mean(np.abs(cc[max(0, lo-1000):min(len(cc), hi+1000)])) + 1e-12)
    conf = float(np.max(cc[lo:hi+1]) / denom) if hi > lo else float(np.max(cc) / (np.mean(np.abs(cc)) + 1e-12))
    # Map to 48k sample index
    start_k_48k = int(round(start_k_ds * (sr / down_sr)))
    start_k_48k = max(0, min(start_k_48k, max(0, len(mic) - len(scratch))))
    return start_k_48k, conf


def coarse_offset_via_envelope(scratch: np.ndarray, mic: np.ndarray, sr: int, env_sr: int = 100) -> Tuple[int, float]:
    """Coarse locate by correlating smoothed energy envelopes to be robust against timbre differences.
    Returns (start_idx_samples_at_48k, confidence).
    """
    if len(scratch) == 0 or len(mic) == 0:
        return 0, 0.0
    # Absolute value and low-pass by moving average ~100 ms
    def env(sig: np.ndarray, sr_local: int) -> np.ndarray:
        x = np.abs(sig).astype(np.float32)
        # Downsample to env_sr using block averaging
        hop = max(1, int(sr_local / env_sr))
        if hop <= 1:
            ds = x
        else:
            n = (len(x) // hop) * hop
            if n <= 0:
                return np.zeros(1, dtype=np.float32)
            ds = x[:n].reshape(-1, hop).mean(axis=1)
        # Smooth with 0.5 s window
        win = max(3, int(0.5 * env_sr))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        sm = np.convolve(ds, kernel, mode="same")
        # Normalize
        sm = sm - float(np.mean(sm))
        sd = float(np.std(sm) + 1e-6)
        sm = sm / sd
        return sm.astype(np.float32)

    e_scr = env(scratch, sr)
    e_mic = env(mic, sr)
    n = len(e_mic) + len(e_scr) - 1
    nfft = _next_pow2(n)
    MIC = np.fft.rfft(e_mic, nfft)
    SCR = np.fft.rfft(e_scr[::-1], nfft)
    cc = np.fft.irfft(MIC * SCR, nfft)[:n]
    # Valid starts in env domain
    valid_hi = max(0, len(e_mic) - len(e_scr))
    lo = len(e_scr) - 1
    hi = valid_hi + len(e_scr) - 1
    if hi <= lo:
        peak = int(np.argmax(cc))
    else:
        window = cc[lo:hi+1]
        peak = int(np.argmax(window)) + lo
    start_k_env = peak - (len(e_scr) - 1)
    # Confidence: peak against global abs mean
    denom = float(np.mean(np.abs(cc)) + 1e-6)
    conf = float(np.max(cc[lo:hi+1]) / denom) if hi > lo else float(np.max(cc) / denom)
    # Map env index to 48k samples approximately
    # Each env step ~ hop samples; use ratio len(mic)/len(e_mic)
    ratio = len(mic) / max(1, len(e_mic))
    start_k_48k = int(round(start_k_env * ratio))
    start_k_48k = max(0, min(start_k_48k, max(0, len(mic) - len(scratch))))
    return start_k_48k, conf


def refine_start_via_ncc(scratch_bp: np.ndarray, mic_bp: np.ndarray, start_guess: int, sr: int, radius_s: float = 1.0) -> Tuple[int, float]:
    """Refine start index using normalized cross-correlation within ±radius_s around start_guess.
    Returns (best_start_idx, score).
    """
    n = len(scratch_bp)
    if n == 0:
        return max(0, min(start_guess, len(mic_bp))), 0.0
    radius = int(max(1, round(radius_s * sr)))
    best_idx = max(0, min(start_guess, len(mic_bp)))
    best_score = -1.0
    xs = scratch_bp.astype(np.float32)
    xs = xs - float(np.mean(xs))
    denom_x = float(np.linalg.norm(xs) + 1e-8)
    lo = max(0, start_guess - radius)
    hi = min(len(mic_bp) - 1, start_guess + radius)
    for s in range(lo, hi + 1):
        e = s + n
        if e > len(mic_bp):
            break
        ys = mic_bp[s:e].astype(np.float32)
        ys = ys - float(np.mean(ys))
        denom_y = float(np.linalg.norm(ys) + 1e-8)
        if denom_y < 1e-6:
            continue
        score = float(np.dot(xs, ys) / (denom_x * denom_y))
        if score > best_score:
            best_score = score
            best_idx = s
    return best_idx, best_score


def _env_energy(x: np.ndarray, sr_local: int, env_sr: int = 100) -> np.ndarray:
    hop = max(1, int(sr_local / env_sr))
    n = (len(x) // hop) * hop
    if n <= 0:
        return np.zeros(1, dtype=np.float32)
    xabs = np.abs(x[:n])
    ds = xabs.reshape(-1, hop).mean(axis=1).astype(np.float32)
    win = max(3, int(0.5 * env_sr))
    ker = np.ones(win, dtype=np.float32) / float(win)
    sm = np.convolve(ds, ker, mode="same")
    return sm


def refine_start_via_ncc_ds(
    scratch_bp: np.ndarray,
    mic_bp: np.ndarray,
    sr: int,
    start_guess: int,
    seg_s: float = 8.0,
    radius_s: float = 2.0,
    down_sr: int = 8000,
) -> Tuple[int, float]:
    """Refine start using downsampled NCC on a short, high-energy template.
    Returns (best_start_idx_48k, score).
    """
    if len(scratch_bp) == 0:
        return max(0, min(start_guess, len(mic_bp))), 0.0
    # Downsample both
    scr_ds = resample_if_needed(scratch_bp, sr, down_sr)
    mic_ds = resample_if_needed(mic_bp, sr, down_sr)
    # Choose high-energy window in scratch
    env = _env_energy(scr_ds, down_sr, env_sr=100)
    seg_n = int(max(1, round(seg_s * down_sr)))
    if len(scr_ds) <= seg_n:
        tpl = scr_ds
        tpl_lo_ds = 0
    else:
        # sliding window on env to pick max energy region
        max_sum = -1.0
        tpl_lo_ds = 0
        acc = np.convolve(env, np.ones(int(seg_s*100), dtype=np.float32), mode="valid") if len(env) >= int(seg_s*100) else env
        # Use downsampled env step (~10ms per step) since env_sr=100
        if len(env) >= int(seg_s * 100):
            idx = int(np.argmax(acc))
            tpl_lo_ds = int(idx * (down_sr / 100))
        tpl_lo_ds = max(0, min(tpl_lo_ds, len(scr_ds) - seg_n))
        tpl = scr_ds[tpl_lo_ds:tpl_lo_ds + seg_n]

    # Build mic window around start_guess
    start_guess_ds = int(round(start_guess * (down_sr / sr)))
    rad_ds = int(max(1, round(radius_s * down_sr)))
    win_lo = max(0, start_guess_ds - rad_ds)
    win_hi = min(len(mic_ds), start_guess_ds + rad_ds + len(tpl))
    mic_win = mic_ds[win_lo:win_hi]
    if len(mic_win) < len(tpl) + 1:
        # pad to allow valid correlation
        pad = np.zeros(len(tpl) + 1 - len(mic_win), dtype=np.float32)
        mic_win = np.concatenate([mic_win, pad])

    # NCC via FFT convolution
    n = len(mic_win) + len(tpl) - 1
    nfft = _next_pow2(n)
    MIC = np.fft.rfft(mic_win, nfft)
    TPL = np.fft.rfft(tpl[::-1], nfft)
    cc = np.fft.irfft(MIC * TPL, nfft)[:n]
    # valid positions
    lo = len(tpl) - 1
    hi = len(mic_win) - 1
    corr_seg = cc[lo:hi+1]
    # Normalize scores roughly by local energy (use envelope scaling)
    # For simplicity, take plain peak
    peak_off = int(np.argmax(corr_seg))
    best_start_ds = win_lo + peak_off
    best_start_48k = int(round(best_start_ds * (sr / down_sr)))
    best_start_48k = max(0, min(best_start_48k, max(0, len(mic_bp) - len(scratch_bp))))
    peak_val = float(np.max(corr_seg))
    mean_abs = float(np.mean(np.abs(corr_seg)) + 1e-9)
    score = peak_val / mean_abs
    return best_start_48k, score


def _speech_envelope(x: np.ndarray, sr: int, env_sr: int = 50) -> np.ndarray:
    hop = max(1, int(sr / env_sr))
    n = (len(x) // hop) * hop
    if n <= 0:
        return np.zeros(1, dtype=np.float32)
    x = np.abs(x[:n]).astype(np.float32)
    env = x.reshape(-1, hop).mean(axis=1)
    # Smooth with ~0.4 s window
    win = max(3, int(0.4 * env_sr))
    ker = np.ones(win, dtype=np.float32) / float(win)
    sm = np.convolve(env, ker, mode="same")
    return sm


def pick_speech_templates(scratch_bp: np.ndarray, sr: int, tpl_s: float = 8.0, max_tpl: int = 3) -> List[int]:
    """Pick up to max_tpl template start indices (48k samples) from voiced regions across the clip."""
    env = _speech_envelope(scratch_bp, sr, env_sr=50)
    if len(env) < 3:
        return [0]
    thr = float(np.percentile(env, 70))
    voiced = (env >= thr).astype(np.int8)
    # Map envelope index to sample index
    step = int(sr / 50)
    tpl_len = int(tpl_s * sr)
    candidates: List[int] = []
    # Aim for start, middle, end thirds
    thirds = [0.1, 0.5, 0.8]
    for frac in thirds:
        idx = int(frac * len(env))
        # search locally for a voiced point
        win = 50
        lo = max(0, idx - win)
        hi = min(len(env) - 1, idx + win)
        seg = voiced[lo:hi]
        if seg.any():
            j = int(np.argmax(seg)) + lo
            start = max(0, j * step - tpl_len // 4)
            if start + tpl_len <= len(scratch_bp):
                candidates.append(start)
    # Dedup/limit
    out: List[int] = []
    for s in candidates:
        if not out or all(abs(s - t) > int(2.0 * sr) for t in out):
            out.append(s)
        if len(out) >= max_tpl:
            break
    if not out:
        out = [0]
    return out


def global_align_multi_template(scratch_bp: np.ndarray, mic_bp: np.ndarray, sr: int) -> Tuple[int, float, dict]:
    """Find best alignment start index of mic for scratch by using multiple templates.
    Returns (best_start_idx_48k, confidence, debug_info).
    """
    tpl_starts = pick_speech_templates(scratch_bp, sr, tpl_s=8.0, max_tpl=3)
    candidates: List[int] = []
    scores: List[float] = []
    for s0 in tpl_starts:
        tpl = scratch_bp[s0 : min(len(scratch_bp), s0 + int(8.0 * sr))]
        if len(tpl) < int(2.0 * sr):
            continue
        # correlate over entire mic
        start_ds, score = refine_start_via_ncc_ds(tpl, mic_bp, sr, start_guess=0, seg_s=8.0, radius_s=max(1.0, len(mic_bp) / sr), down_sr=8000)
        # convert template-local to global mic start for whole clip
        start_idx = max(0, start_ds - s0)
        candidates.append(start_idx)
        scores.append(score)
    if not candidates:
        return 0, 0.0, {"candidates": [], "scores": []}
    # Robust combine: median and inlier filtering
    med = int(np.median(np.array(candidates)))
    tol = int(1.0 * sr)
    inliers = [i for i, c in enumerate(candidates) if abs(c - med) <= tol]
    if inliers:
        best = int(np.mean([candidates[i] for i in inliers]))
        conf = float(np.mean([scores[i] for i in inliers])) + 0.5 * len(inliers)
    else:
        best = med
        conf = float(np.mean(scores))
    return best, conf, {"tpl_starts": tpl_starts, "candidates": candidates, "scores": scores, "median": med}


def _ncc_search_full(tpl: np.ndarray, mic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized cross-correlation of tpl over mic. Returns (ncc, valid_indices_start)."""
    n_tpl = len(tpl)
    n_mic = len(mic)
    if n_tpl <= 1 or n_mic <= n_tpl:
        return np.zeros(1, dtype=np.float32), np.arange(1, dtype=np.int64)
    # zero-mean template
    tpl_zm = tpl.astype(np.float32) - float(np.mean(tpl))
    den_tpl = float(np.sqrt(np.sum(tpl_zm * tpl_zm)) + 1e-8)
    nfft = _next_pow2(n_mic + n_tpl - 1)
    MIC = np.fft.rfft(mic, nfft)
    TPL = np.fft.rfft(tpl_zm[::-1], nfft)
    cc = np.fft.irfft(MIC * TPL, nfft)[: n_mic + n_tpl - 1]
    # sliding energy of mic
    mic2 = mic * mic
    win = np.zeros(n_tpl, dtype=np.float32) + 1.0
    WIN = np.fft.rfft(win, nfft)
    conv = np.fft.irfft(np.fft.rfft(mic2, nfft) * WIN, nfft)[: n_mic + n_tpl - 1]
    lo = n_tpl - 1
    hi = n_mic - 1
    valid = np.arange(lo, hi + 1, dtype=np.int64)
    denom = np.sqrt(conv[valid]) * den_tpl + 1e-8
    ncc = (cc[valid] / denom).astype(np.float32)
    return ncc, valid - (n_tpl - 1)


def best_start_ncc_multi(scratch_bp: np.ndarray, mic_bp: np.ndarray, sr: int, down_sr: int = 4000) -> Tuple[int, float, dict]:
    """Robustly find best start of mic for the scratch by NCC over entire mic.
    Downsamples to down_sr for speed and energy-normalizes windows for reliability.
    Returns (best_start_48k, confidence, debug).
    """
    # choose template segment length
    tpl_len_s = 12.0 if len(scratch_bp) / sr > 20.0 else max(6.0, 0.6 * (len(scratch_bp) / sr))
    # high-energy template starts
    tpl_starts = pick_speech_templates(scratch_bp, sr, tpl_s=tpl_len_s, max_tpl=3)
    scr_ds = resample_if_needed(scratch_bp, sr, down_sr)
    mic_ds = resample_if_needed(mic_bp, sr, down_sr)
    starts_ds: List[int] = []
    scores: List[float] = []
    for s0_48 in tpl_starts:
        s0_ds = int(round(s0_48 * (down_sr / sr)))
        tpl = scr_ds[s0_ds : s0_ds + int(tpl_len_s * down_sr)]
        if len(tpl) < int(2.0 * down_sr):
            continue
        # zero-mean mic
        mic_ds_zm = mic_ds.astype(np.float32) - float(np.mean(mic_ds))
        ncc, idx = _ncc_search_full(tpl, mic_ds_zm)
        if len(ncc) == 0:
            continue
        k = int(np.argmax(ncc))
        pos_ds = int(idx[k])
        conf = float(ncc[k]) / (float(np.mean(ncc) + 1e-6))
        starts_ds.append(pos_ds)
        scores.append(conf)
    if not starts_ds:
        return 0, 0.0, {"tpl_starts": tpl_starts, "starts_ds": [], "scores": []}
    # Robust combine using median and inliers within ±1 s at down_sr scale
    med_ds = int(np.median(np.array(starts_ds)))
    tol = int(1.0 * down_sr)
    inliers = [i for i, sds in enumerate(starts_ds) if abs(sds - med_ds) <= tol]
    if inliers:
        best_ds = int(np.mean([starts_ds[i] for i in inliers]))
        conf = float(np.mean([scores[i] for i in inliers])) + 0.5 * len(inliers)
    else:
        best_ds = med_ds
        conf = float(np.mean(scores))
    best_48k = int(round(best_ds * (sr / down_sr)))
    return best_48k, conf, {"tpl_starts": tpl_starts, "starts_ds": starts_ds, "scores": scores, "median_ds": med_ds, "best_ds": best_ds}


def residual_offset_correct(aligned_bp: np.ndarray, scratch_bp: np.ndarray, sr: int) -> Tuple[int, float]:
    """Measure residual offset via GCC-PHAT on a mid window; return (shift_samples, seconds). Positive means aligned needs to be delayed."""
    n = min(len(aligned_bp), len(scratch_bp))
    if n < int(5 * sr):
        return 0, 0.0
    lo = int(n * 0.25)
    hi = int(n * 0.25 + min(n * 0.5, 20 * sr))
    x = scratch_bp[lo:hi]
    y = aligned_bp[lo:hi]
    off_s, _ = gcc_phat_offset(x, y, sr, max_offset_s=2.0)
    shift = int(round(off_s * sr))
    return shift, off_s


# ------------------------------ Audalign Backend (Preferred) ------------------------------

def align_with_audalign_files(scratch_wav: Path, mic_wav: Path, out_len: int, mode: str = "correlation_spectrogram") -> np.ndarray:
    """Align mic_wav to scratch_wav using audalign's high-level file API.
    Returns aligned mono float32 array of exact out_len.
    """
    try:
        import audalign as ad  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "audalign is not installed. Please install with:\n"
            "  python -m pip install audalign==1.3.1\n"
            f"Import error: {e}"
        )
    # Pick recognizer
    rec = None
    try:
        if mode == "fingerprint" and hasattr(ad, "FingerprintRecognizer"):
            rec = ad.FingerprintRecognizer()
            if hasattr(rec, "config") and hasattr(rec.config, "set_accuracy"):
                rec.config.set_accuracy(3)
        elif mode == "correlation" and hasattr(ad, "CorrelationRecognizer"):
            rec = ad.CorrelationRecognizer()
        elif mode == "correlation_spectrogram" and hasattr(ad, "CorrelationSpectrogramRecognizer"):
            rec = ad.CorrelationSpectrogramRecognizer()
        elif mode == "visual" and hasattr(ad, "VisualRecognizer"):
            rec = ad.VisualRecognizer()
        elif hasattr(ad, "CorrelationRecognizer"):
            rec = ad.CorrelationRecognizer()
    except Exception:
        rec = None
    if rec is None:
        raise RuntimeError("audalign recognizer classes not found in this installation")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "aligned"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            logging.info(
                f"audalign version={getattr(ad, '__version__', 'unknown')}; recognizer={rec.__class__.__name__ if rec else 'None'}; dest={dest}"
            )
            results = ad.align_files(str(scratch_wav), str(mic_wav), destination_path=str(dest), recognizer=rec)
            logging.info("audalign.align_files completed")
        except Exception as e:
            raise RuntimeError(f"audalign.align_files failed: {e}")
        # Find aligned mic file in dest with same stem
        mic_stem = mic_wav.stem.lower()
        outs = [p for p in dest.glob("**/*") if p.is_file() and p.suffix.lower() == ".wav"]
        logging.info(f"audalign outputs: {[str(p.name) for p in outs]}")
        pick = None
        for p in outs:
            if p.stem.lower() == mic_stem:
                pick = p
                break
        if pick is None and outs:
            pick = max(outs, key=lambda p: p.stat().st_size)
        if pick is None or not pick.exists():
            raise RuntimeError("audalign did not produce an aligned wav in destination")
        logging.info(f"Using aligned file from audalign: {pick}")
        y, sr_out = _read_wav_mono_any(pick)
        y = y.astype(np.float32)
        # Crop/pad to out_len
        if len(y) < out_len:
            y = np.concatenate([y, np.zeros(out_len - len(y), dtype=np.float32)])
        elif len(y) > out_len:
            y = y[:out_len]
        return y


# ------------------------------ Coarse-to-Fine DTW (legacy fallback) ------------------------------

def _rfft_band_energies(x: np.ndarray, sr: int, win: int, hop: int, fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    if len(x) < win:
        pad = np.zeros(win - len(x), dtype=np.float32)
        x = np.concatenate([x, pad])
    n = len(x)
    frames = 1 + (n - win) // hop if n >= win else 1
    feat = np.zeros((frames, n_bands), dtype=np.float32)
    # Precompute band bin indices
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    valid = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(valid) < n_bands:
        # fallback: stretch bands across available bins
        bands = np.array_split(valid, max(1, min(len(valid), n_bands)))
    else:
        # linear bands between fmin..fmax
        edges = np.linspace(fmin, fmax, num=n_bands + 1)
        bands = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0] for i in range(n_bands)]
    window = np.hanning(win).astype(np.float32)
    for i in range(frames):
        s = i * hop
        e = s + win
        if e > n:
            frame = np.zeros(win, dtype=np.float32)
            chunk = x[s:]
            frame[: len(chunk)] = chunk
        else:
            frame = x[s:e]
        X = np.fft.rfft(frame * window)
        mag2 = (np.abs(X) ** 2).astype(np.float32)
        for b, idxs in enumerate(bands):
            if len(idxs) == 0:
                feat[i, b] = 0.0
            else:
                feat[i, b] = float(np.mean(mag2[idxs]))
    # log and normalize per-frame
    feat = np.log(feat + 1e-8)
    # z-normalize per band
    mu = np.mean(feat, axis=0, keepdims=True)
    sd = np.std(feat, axis=0, keepdims=True) + 1e-6
    feat = (feat - mu) / sd
    return feat


def compute_logspec_features(x: np.ndarray, sr: int, hop_s: float, win_s: float, n_bands: int = 48, fmin: float = 100.0, fmax: float = 4000.0) -> Tuple[np.ndarray, int, int]:
    win = max(128, int(round(win_s * sr)))
    hop = max(32, int(round(hop_s * sr)))
    feat = _rfft_band_energies(x, sr, win, hop, fmin, fmax, n_bands)
    return feat, win, hop


def sliding_coarse_candidates(s_env: np.ndarray, m_env: np.ndarray, tpl_len_frames: int, top_k: int = 3) -> List[int]:
    # Cross-correlate 1D envelopes to get candidate starts
    if len(s_env) > tpl_len_frames:
        tpl = s_env[:tpl_len_frames]
    else:
        tpl = s_env
    tpl = tpl - float(np.mean(tpl))
    m = m_env - float(np.mean(m_env))
    nfft = _next_pow2(len(m) + len(tpl) - 1)
    M = np.fft.rfft(m, nfft)
    T = np.fft.rfft(tpl[::-1], nfft)
    cc = np.fft.irfft(M * T, nfft)[: len(m) + len(tpl) - 1]
    lo = len(tpl) - 1
    hi = len(m) - 1
    if hi <= lo:
        peaks = [int(np.argmax(cc)) - (len(tpl) - 1)]
    else:
        seg = cc[lo:hi + 1]
        # Pick top_k peaks with non-maximum suppression
        peaks = []
        seg_copy = seg.copy()
        for _ in range(top_k):
            k = int(np.argmax(seg_copy))
            if seg_copy[k] <= 0:
                break
            peaks.append(k)
            # suppress neighborhood
            rad = max(1, len(tpl) // 4)
            a = max(0, k - rad)
            b = min(len(seg_copy), k + rad)
            seg_copy[a:b] = -1.0
        peaks = [int(p) for p in peaks]
    starts = [p for p in peaks]
    return starts


def subsequence_dtw(Fs: np.ndarray, Fm: np.ndarray, j_lo: int, j_hi: int) -> Tuple[int, float, List[Tuple[int, int]]]:
    """Subsequence DTW aligning Fs (Ns x K) to Fm window [j_lo:j_hi) (Nm_w x K).
    Returns (best_end_j, avg_cost, path)
    """
    Ns, K = Fs.shape
    Mwin = max(1, j_hi - j_lo)
    # Precompute distances
    # normalize rows
    def _znorm(A):
        mu = np.mean(A, axis=1, keepdims=True)
        sd = np.std(A, axis=1, keepdims=True) + 1e-6
        return (A - mu) / sd
    A = _znorm(Fs).astype(np.float32)
    B = _znorm(Fm[j_lo:j_hi]).astype(np.float32)
    # Cosine distance ~ 1 - dot/|| ||, since rows normalized already
    Dcost = 1.0 - (A @ B.T)
    # DP
    INF = 1e9
    Acc = np.full((Ns + 1, Mwin + 1), INF, dtype=np.float32)
    Ptr = np.zeros((Ns + 1, Mwin + 1), dtype=np.int8)  # 1=diag,2=up,3=left
    Acc[0, :] = 0.0  # subsequence DTW initialization
    for i in range(1, Ns + 1):
        # constrain j to [i-2 .. Mwin] to avoid huge band; allow some slack
        jstart = 1
        jend = Mwin
        for j in range(jstart, jend + 1):
            c = Dcost[i - 1, j - 1]
            # min of insertion (i-1,j), match (i-1,j-1), deletion (i,j-1)
            m1 = Acc[i - 1, j]
            m2 = Acc[i - 1, j - 1]
            m3 = Acc[i, j - 1]
            if m2 <= m1 and m2 <= m3:
                Acc[i, j] = c + m2
                Ptr[i, j] = 1
            elif m1 <= m3:
                Acc[i, j] = c + m1
                Ptr[i, j] = 2
            else:
                Acc[i, j] = c + m3
                Ptr[i, j] = 3
    # Best end at i=Ns, any j
    j_end = int(np.argmin(Acc[Ns, 1:])) + 1
    best_cost = float(Acc[Ns, j_end] / max(1, Ns))
    # backtrack
    path: List[Tuple[int, int]] = []
    i = Ns
    j = j_end
    while i > 0 and j > 0:
        path.append((i - 1, j_lo + j - 1))
        p = Ptr[i, j]
        if p == 1:
            i -= 1; j -= 1
        elif p == 2:
            i -= 1
        elif p == 3:
            j -= 1
        else:
            break
    path.reverse()
    return j_lo + j_end - 1, best_cost, path


def build_time_map_from_path(path: List[Tuple[int, int]], hop_s: float) -> Tuple[np.ndarray, np.ndarray]:
    if not path:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    i_frames = np.array([p[0] for p in path], dtype=np.int64)
    j_frames = np.array([p[1] for p in path], dtype=np.int64)
    # ensure monotonically increasing and unique i_frames
    uniq, idx = np.unique(i_frames, return_index=True)
    i_frames = i_frames[idx]
    j_frames = j_frames[idx]
    t_out = i_frames.astype(np.float32) * hop_s
    t_src = j_frames.astype(np.float32) * hop_s
    # pad edges
    if t_out[0] > 0.0:
        t_out = np.concatenate([[0.0], t_out])
        t_src = np.concatenate([[t_src[0]], t_src])
    return t_out, t_src


def apply_time_map_resample(mic: np.ndarray, sr: int, t_out: np.ndarray, t_src: np.ndarray, target_len: int) -> np.ndarray:
    # Build target time vector for each sample of output
    tout = np.linspace(0.0, t_out[-1], num=target_len, endpoint=False).astype(np.float32)
    # Interpolate source times for each output time
    tsrc = np.interp(tout, t_out, t_src)
    xs = np.arange(len(mic), dtype=np.float32) / float(sr)
    y = np.interp(tsrc, xs, mic, left=0.0, right=0.0).astype(np.float32)
    return y


def linear_params_from_path(path: List[Tuple[int, int]], hop_s: float) -> Tuple[float, float, float]:
    """Fit linear mapping j ≈ a + b*i from DTW path pairs (i,j) in frames.
    Returns (a_frames, b_slope, r2)
    """
    if not path:
        return 0.0, 1.0, 0.0
    i = np.array([p[0] for p in path], dtype=np.float64)
    j = np.array([p[1] for p in path], dtype=np.float64)
    # Least squares fit j = a + b*i
    A = np.vstack([np.ones_like(i), i]).T
    x, _, _, _ = np.linalg.lstsq(A, j, rcond=None)
    a, b = float(x[0]), float(x[1])
    # r^2
    j_pred = a + b * i
    ss_res = float(np.sum((j - j_pred) ** 2))
    ss_tot = float(np.sum((j - np.mean(j)) ** 2) + 1e-9)
    r2 = 1.0 - ss_res / ss_tot
    return a, b, r2


def trim_silence_safely(
    wav: np.ndarray,
    sr: int,
    pad_head_s: float = 0.60,
    pad_tail_s: float = 1.00,
    vad_aggr: int = 2,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.80,
) -> Tuple[np.ndarray, int, int]:
    n = len(wav)
    if n == 0:
        return wav, 0, 0
    # Internal helpers
    def _try_import_webrtcvad():
        try:
            import webrtcvad  # type: ignore
            return webrtcvad
        except Exception:
            return None
    def _rms_db(x: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        return 20.0 * math.log10(max(rms, 1e-12))

    sr_vad = 16000
    x16 = resample_if_needed(wav, sr, sr_vad)
    frame_ms = 30
    frame_len = int(sr_vad * frame_ms / 1000.0)
    if frame_len <= 0:
        return wav, 0, n

    vad_lib = _try_import_webrtcvad()
    if vad_lib is None:
        logging.info("webrtcvad not available; using ffmpeg silenceremove fallback")
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                tmp_in = Path(td) / "aligned.wav"
                tmp_out = Path(td) / "trimmed.wav"
                _write_wav_mono_int16(tmp_in, wav, sr)
                cmd = [
                    "ffmpeg","-y","-i",str(tmp_in),
                    "-af","silenceremove=start_periods=1:start_silence=1:start_threshold=-35dB:stop_periods=1:stop_silence=1:stop_threshold=-35dB",
                    str(tmp_out)
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                if proc.returncode != 0 or not tmp_out.exists():
                    logging.warning("ffmpeg silenceremove failed; skipping trim")
                    return wav, 0, n
                trimmed, _ = _read_wav_mono_int16(tmp_out)
                # Manual safety pads
                pad_head = int(round(pad_head_s * sr))
                pad_tail = int(round(pad_tail_s * sr))
                trimmed = np.concatenate((np.zeros(pad_head, dtype=np.float32), trimmed, np.zeros(pad_tail, dtype=np.float32)))
                # Guardrails relative to original
                if len(trimmed) < int(3.0 * sr):
                    return wav, 0, n
                if (n - len(trimmed)) > 0.25 * n:
                    return wav, 0, n
                return trimmed[:n].copy(), 0, min(n, len(trimmed))
        except Exception as e:
            logging.warning(f"ffmpeg fallback exception: {e}; skipping trim")
            return wav, 0, n

    # Primary VAD path
    vad = vad_lib.Vad(int(max(0, min(3, vad_aggr))))
    n_frames = len(x16) // frame_len
    flags = []
    for i in range(n_frames):
        frame = x16[i * frame_len:(i + 1) * frame_len]
        pcm = np.clip(frame, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
        try:
            is_speech = vad.is_speech(pcm16, sr_vad)
        except Exception:
            is_speech = False
        flags.append(bool(is_speech))
    if not any(flags):
        return wav, 0, n

    min_speech_frames = max(1, int(round(min_speech_s * 1000.0 / frame_ms)))
    min_silence_frames = max(1, int(round(min_silence_s * 1000.0 / frame_ms)))

    # Find start: first run of voiced >= min_speech_frames
    start_frame = 0
    run = 0
    found_start = False
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= min_speech_frames:
            start_frame = i - run + 1
            found_start = True
            break
    if not found_start:
        return wav, 0, n

    # Find end: find last voiced; prefer a trailing silence of >= min_silence_frames
    last_voiced = max(i for i, f in enumerate(flags) if f)
    end_frame = last_voiced + 1
    silence_run = 0
    for i in range(len(flags) - 1, -1, -1):
        if not flags[i]:
            silence_run += 1
            if silence_run >= min_silence_frames:
                j = i - silence_run + 1
                prev_voiced = None
                for k in range(j - 1, -1, -1):
                    if flags[k]:
                        prev_voiced = k
                        break
                if prev_voiced is not None:
                    end_frame = prev_voiced + 1
                break
        else:
            silence_run = 0

    start_s_rough = (start_frame * frame_len) / sr_vad
    end_s_rough = (end_frame * frame_len) / sr_vad

    start_s = max(0.0, start_s_rough - pad_head_s)
    end_s = min(len(wav) / sr, end_s_rough + pad_tail_s)

    def cross_check(bound_s: float, interior: bool) -> float:
        win_s = 0.2
        backoff_s = 0.5
        if interior:
            inside_start = int(bound_s * sr)
            inside_end = int(min(len(wav), inside_start + win_s * sr))
            outside_end = int(max(0, inside_start - 1))
            outside_start = int(max(0, outside_end - win_s * sr))
        else:
            inside_end = int(bound_s * sr)
            inside_start = int(max(0, inside_end - win_s * sr))
            outside_start = int(min(len(wav), inside_end + 1))
            outside_end = int(min(len(wav), outside_start + win_s * sr))
        if inside_end <= inside_start or outside_end <= outside_start:
            return bound_s
        in_db = _rms_db(wav[inside_start:inside_end])
        out_db = _rms_db(wav[outside_start:outside_end])
        if in_db < out_db + 12.0:
            return max(0.0, bound_s - backoff_s) if interior else min(len(wav) / sr, bound_s + backoff_s)
        return bound_s

    start_s = cross_check(start_s, interior=True)
    end_s = cross_check(end_s, interior=False)

    start_idx = int(round(start_s * sr))
    end_idx = int(round(end_s * sr))
    start_idx = max(0, min(start_idx, len(wav)))
    end_idx = max(0, min(end_idx, len(wav)))
    if end_idx <= start_idx:
        return wav, 0, n

    speech_len = end_idx - start_idx
    if speech_len < int(3.0 * sr):
        return wav, 0, n
    removed = start_idx + (len(wav) - end_idx)
    if removed > 0.25 * len(wav):
        return wav, 0, n

    return wav[start_idx:end_idx].copy(), start_idx, end_idx


def write_aligned_wav(out_path: Path, data: np.ndarray, sr: int, stereo: bool = False) -> None:
    _write_wav_mono_int16(out_path, data.astype(np.float32), sr)


def _file_url(p: Path) -> str:
    ap = str(p.resolve()).replace("\\", "/")
    if not ap.startswith("/"):
        ap = "/" + ap
    return f"file://{ap}"


def write_fcp7_xml(sequences: List['SequenceSpec'], timebase: int, out_path: Path, project_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ntsc = "FALSE"
    xml_parts = []
    xml_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_parts.append('<!DOCTYPE xmeml>')
    xml_parts.append('<xmeml version="5">')
    for seq in sequences:
        xml_parts.append("  <sequence>")
        xml_parts.append(f"    <name>{project_name} - {seq.name}</name>")
        xml_parts.append("    <rate>")
        xml_parts.append(f"      <timebase>{timebase}</timebase>")
        xml_parts.append(f"      <ntsc>{ntsc}</ntsc>")
        xml_parts.append("    </rate>")
        xml_parts.append("    <timecode>")
        xml_parts.append("      <rate>")
        xml_parts.append(f"        <timebase>{timebase}</timebase>")
        xml_parts.append(f"        <ntsc>{ntsc}</ntsc>")
        xml_parts.append("      </rate>")
        xml_parts.append("      <string>00:00:00:00</string>")
        xml_parts.append("      <frame>0</frame>")
        xml_parts.append("      <displayformat>NDF</displayformat>")
        xml_parts.append("    </timecode>")
        xml_parts.append(f"    <duration>{seq.duration_frames}</duration>")
        xml_parts.append("    <media>")
        xml_parts.append("      <video>")
        xml_parts.append("        <track>")
        xml_parts.append("          <clipitem id=\"v1\">")
        xml_parts.append(f"            <name>{seq.video.stem}</name>")
        xml_parts.append("            <enabled>TRUE</enabled>")
        xml_parts.append("            <start>0</start>")
        xml_parts.append(f"            <end>{seq.duration_frames}</end>")
        xml_parts.append("            <in>0</in>")
        xml_parts.append(f"            <out>{seq.duration_frames}</out>")
        xml_parts.append("            <file>")
        xml_parts.append(f"              <name>{seq.video.name}</name>")
        xml_parts.append("              <pathurl>" + _file_url(seq.video) + "</pathurl>")
        xml_parts.append("              <rate>")
        xml_parts.append(f"                <timebase>{timebase}</timebase>")
        xml_parts.append(f"                <ntsc>{ntsc}</ntsc>")
        xml_parts.append("              </rate>")
        xml_parts.append("            </file>")
        xml_parts.append("          </clipitem>")
        xml_parts.append("        </track>")
        xml_parts.append("      </video>")
        xml_parts.append("      <audio>")
        track_index = 1
        if seq.scratch_wav is not None:
            xml_parts.append("        <track>")
            xml_parts.append(f"          <clipitem id=\"a{track_index}\">")
            xml_parts.append(f"            <name>scratch_{seq.video.stem}</name>")
            xml_parts.append("            <enabled>FALSE</enabled>")
            xml_parts.append("            <start>0</start>")
            xml_parts.append(f"            <end>{seq.duration_frames}</end>")
            xml_parts.append("            <in>0</in>")
            xml_parts.append(f"            <out>{seq.duration_frames}</out>")
            xml_parts.append("            <file>")
            xml_parts.append(f"              <name>{seq.scratch_wav.name}</name>")
            xml_parts.append("              <pathurl>" + _file_url(seq.scratch_wav) + "</pathurl>")
            xml_parts.append("              <rate>")
            xml_parts.append(f"                <timebase>{timebase}</timebase>")
            xml_parts.append(f"                <ntsc>{ntsc}</ntsc>")
            xml_parts.append("              </rate>")
            xml_parts.append("              <media>")
            xml_parts.append("                <audio>")
            xml_parts.append("                  <samplecharacteristics>")
            xml_parts.append("                    <samplerate>48000</samplerate>")
            xml_parts.append("                    <channels>1</channels>")
            xml_parts.append("                  </samplecharacteristics>")
            xml_parts.append("                </audio>")
            xml_parts.append("              </media>")
            xml_parts.append("            </file>")
            xml_parts.append("          </clipitem>")
            xml_parts.append("        </track>")
            track_index += 1
        for tr in seq.aligned_wavs:
            xml_parts.append("        <track>")
            xml_parts.append(f"          <clipitem id=\"a{track_index}\">")
            xml_parts.append(f"            <name>{tr.path.stem}</name>")
            xml_parts.append("            <enabled>TRUE</enabled>")
            start_f = int(round(tr.start_offset_s * timebase))
            # Place for remaining sequence duration
            end_f = seq.duration_frames
            xml_parts.append(f"            <start>{start_f}</start>")
            xml_parts.append(f"            <end>{end_f}</end>")
            xml_parts.append("            <in>0</in>")
            xml_parts.append(f"            <out>{end_f - start_f}</out>")
            xml_parts.append("            <file>")
            xml_parts.append(f"              <name>{tr.path.name}</name>")
            xml_parts.append("              <pathurl>" + _file_url(tr.path) + "</pathurl>")
            xml_parts.append("              <rate>")
            xml_parts.append(f"                <timebase>{timebase}</timebase>")
            xml_parts.append(f"                <ntsc>{ntsc}</ntsc>")
            xml_parts.append("              </rate>")
            xml_parts.append("              <media>")
            xml_parts.append("                <audio>")
            xml_parts.append("                  <samplecharacteristics>")
            xml_parts.append("                    <samplerate>48000</samplerate>")
            xml_parts.append("                    <channels>1</channels>")
            xml_parts.append("                  </samplecharacteristics>")
            xml_parts.append("                </audio>")
            xml_parts.append("              </media>")
            xml_parts.append("            </file>")
            xml_parts.append("          </clipitem>")
            xml_parts.append("        </track>")
            track_index += 1
        xml_parts.append("      </audio>")
        xml_parts.append("    </media>")
        xml_parts.append("  </sequence>")
    xml_parts.append("</xmeml>")
    out_path.write_text("\n".join(xml_parts), encoding="utf-8")


def write_prepped_mp4(video_in: Path, aligned_wavs: List[AlignedTrack], out_mp4: Path, include_scratch: Optional[Path], audio_offset_s: float = 0.0) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_in)]
    for tr in aligned_wavs:
        if tr.start_offset_s > 0:
            cmd += ["-itsoffset", f"{tr.start_offset_s:.6f}"]
        cmd += ["-i", str(tr.path)]
    if include_scratch is not None:
        cmd += ["-i", str(include_scratch)]
    cmd += ["-map", "0:v:0", "-c:v", "copy"]
    num_audio = len(aligned_wavs) + (1 if include_scratch is not None else 0)
    for i in range(num_audio):
        cmd += ["-map", f"{i+1}:a:0"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_mp4)]
    logging.info(f"Writing MP4 with aligned audio: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        logging.error(f"ffmpeg error muxing MP4: {proc.stderr.decode(errors='ignore')}")
        raise RuntimeError("Failed to write MP4 with ffmpeg")


def _compute_residual_ms(scratch_wav: Path, aligned_wav: Path, max_offset_s: float = 2.0) -> int:
    try:
        s, sr_s = _read_wav_mono_any(scratch_wav)
        a, sr_a = _read_wav_mono_any(aligned_wav)
        if sr_s != 48000:
            s = resample_if_needed(s, sr_s, 48000)
            sr_s = 48000
        if sr_a != 48000:
            a = resample_if_needed(a, sr_a, 48000)
            sr_a = 48000
        # Use middle window up to 30s
        n = min(len(s), len(a))
        if n <= 0:
            return 0
        sb = bandpass_voice(s[:n], 48000)
        ab = bandpass_voice(a[:n], 48000)
        sb = rms_normalize(sb)
        ab = rms_normalize(ab)
        off_s, _ = gcc_phat_offset(sb, ab, 48000, max_offset_s=max_offset_s)
        return int(round(off_s * 1000.0))
    except Exception:
        return 0


def write_prepped_mov_pcm(video_in: Path, aligned_wavs: List[AlignedTrack], out_mov: Path, include_scratch: Optional[Path], scratch_wav: Optional[Path] = None, auto_correct: bool = False, max_residual_s: float = 2.0) -> None:
    """Mux video with aligned audio using filter_complex and adelay to apply per-track offsets.
    Uses PCM s16le for audio to avoid codec delay and preserves video with stream copy.
    """
    out_mov.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(video_in)]
    # Add audio inputs
    for tr in aligned_wavs:
        cmd += ["-i", str(tr.path)]
    if include_scratch is not None:
        cmd += ["-i", str(include_scratch)]

    # Build filter_complex for per-track delay and formatting
    filter_parts = []
    out_labels = []
    # Audio inputs start at index 1
    ai = 1
    residuals_ms: List[int] = []
    applied_ms: List[int] = []
    for idx, tr in enumerate(aligned_wavs, start=1):
        delay_ms = max(0, int(round(tr.start_offset_s * 1000.0)))
        res_ms = 0
        if auto_correct and scratch_wav is not None:
            res_ms = _compute_residual_ms(scratch_wav, tr.path, max_offset_s=max_residual_s)
            # Only apply small, sane correction (|res| <= max_residual_s)
            res_ms = int(max(-max_residual_s*1000.0, min(max_residual_s*1000.0, res_ms)))
            delay_ms = max(0, delay_ms + res_ms)
        residuals_ms.append(res_ms)
        applied_ms.append(delay_ms)
        in_label = f"[{ai}:a]"
        out_label = f"[a{idx}]"
        # aformat to mono@48k -> adelay head silence -> reset pts -> apad tail to avoid -shortest truncation
        if delay_ms > 0:
            chain = (
                f"{in_label}aformat=sample_rates=48000:channel_layouts=mono,"
                f"adelay={delay_ms}:all=1,asetpts=PTS-STARTPTS,apad{out_label}"
            )
        else:
            chain = f"{in_label}aformat=sample_rates=48000:channel_layouts=mono,asetpts=PTS-STARTPTS,apad{out_label}"
        filter_parts.append(chain)
        out_labels.append(out_label)
        ai += 1
    # Optional scratch as last track (no delay)
    if include_scratch is not None:
        out_label = f"[a{len(out_labels)+1}]"
        filter_parts.append(f"[{ai}:a]aformat=sample_rates=48000:channel_layouts=mono,apad{out_label}")
        out_labels.append(out_label)

    if filter_parts:
        cmd += ["-filter_complex", ";".join(filter_parts)]

    # Mapping
    cmd += ["-map", "0:v:0", "-c:v", "copy"]
    for ol in out_labels:
        cmd += ["-map", ol]
    cmd += ["-c:a", "pcm_s16le", "-shortest", str(out_mov)]

    logging.info(f"Writing MOV(PCM) with aligned audio (filters): {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        logging.error(f"ffmpeg error muxing MOV: {proc.stderr.decode(errors='ignore')}")
        raise RuntimeError("Failed to write MOV with ffmpeg")
    # Attach residual/applied info to logger for debug
    logging.info(f"Residual corrections per track (ms): {residuals_ms}; applied delays (ms): {applied_ms}")


# == SECTION: MAIN ==

def main():
    parser = argparse.ArgumentParser(description="Premiere Prepper: sync, clean, trim and export FCP7 XML/MP4")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cams", nargs="+", required=True)
    parser.add_argument("--mics", required=True)
    parser.add_argument("--timebase", type=int, default=25)
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--write-mp4", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--strict-clean", action="store_true")
    parser.add_argument("--max-offset", type=float, default=1.5)
    parser.add_argument("--drift-correct", action="store_true")
    parser.add_argument("--trim-on", action="store_true")
    parser.add_argument("--vad-aggr", type=int, default=2)
    parser.add_argument("--min-speech", type=float, default=0.25)
    parser.add_argument("--min-silence", type=float, default=0.80)
    parser.add_argument("--head-pad", type=float, default=0.60)
    parser.add_argument("--tail-pad", type=float, default=1.00)
    parser.add_argument("--global-search", action="store_true", help="Enable global long-range mic search per clip")
    parser.add_argument("--write-mov-pcm", action="store_true", help="Also write MOV with PCM audio (for QC, no encoder delay)")
    parser.add_argument("--debug-mux", action="store_true", help="Export sync diagnostics and ffprobe reports for MOV muxing")
    parser.add_argument("--auto-mux-correct", action="store_true", help="Auto-correct residual sync in mux by measuring scratch vs aligned wav")
    args = parser.parse_args()

    in_root: Path = args.input
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(out_root / "Logs")
    logging.info("Premiere Prepper starting...")

    logging.info("Discovering camera clips and mic files...")
    video_clips = discover_clips(in_root, args.cams)
    mic_clips = discover_mics(in_root, args.mics)
    logging.info(f"Found {len(video_clips)} camera clips across {len(args.cams)} cams")
    logging.info(f"Found {len(mic_clips)} mic files in '{args.mics}'")

    # Prepare directories
    scratch_dir = out_root / "ScratchWavs"
    prepared_dir = out_root / "PreparedMics"
    aligned_dir = out_root / "AlignedWavs"
    xml_dir = out_root / "XML"
    mp4_dir = out_root / "MP4"
    manifest_path = out_root / "manifest.json"
    diag_dir = out_root / "Diagnostics"
    for d in [scratch_dir, prepared_dir, aligned_dir, xml_dir, mp4_dir, diag_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Prepare mic WAVs @48k mono (no cleaning). Align first, then clean aligned segment.
    prepared_map = {}
    for mic in mic_clips:
        out_prep = prepared_dir / f"{mic.name}_prep48k.wav"
        need_prep = True
        if out_prep.exists() and not args.force_clean:
            try:
                need_prep = mic.path.stat().st_mtime > out_prep.stat().st_mtime
            except Exception:
                need_prep = False
        if need_prep:
            src = mic.path
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out_prep)
            ]
            logging.info(f"Preparing mic (48k mono): {' '.join(cmd)}")
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if proc.returncode != 0:
                logging.error(f"ffmpeg error preparing mic {src}: {proc.stderr.decode(errors='ignore')}")
                continue
        else:
            logging.info(f"Using existing prepared mic: {out_prep}")
        prepared_map[mic.name] = out_prep

    sequences: List[SequenceSpec] = []

    # For each camera clip, extract scratch, align each cleaned mic
    for vc in video_clips:
        logging.info(f"Processing camera clip: {vc.path}")
        # Probe camera A/V start time offset (audio relative to video)
        av_offset_s = 0.0
        try:
            av_offset_s = probe_av_offset_seconds(vc.path)
            logging.info(f"Camera A/V offset (audio-video) for {vc.path.name}: {av_offset_s:.4f}s")
        except Exception as e:
            logging.warning(f"ffprobe failed to get A/V offset for {vc.path.name}: {e}")
        scratch_wav = scratch_dir / f"{vc.camera}_{vc.order_index:03d}_scratch.wav"
        if not scratch_wav.exists():
            try:
                extract_camera_audio_ffmpeg(vc.path, scratch_wav, sr=48000)
            except Exception as e:
                logging.error(f"Skipping clip (failed scratch extract): {e}")
                continue
        try:
            scratch, sr = load_mono_48k(scratch_wav)
        except Exception as e:
            logging.error(f"Failed to load scratch wav {scratch_wav}: {e}")
            continue

        scratch_bp = bandpass_voice(scratch, sr)
        scratch_bp = rms_normalize(scratch_bp)

        aligned_tracks: List[AlignedTrack] = []
        for mic_name, prepared_path in prepared_map.items():
            try:
                mic_raw, sr_m = load_mono_48k(prepared_path)
            except Exception as e:
                logging.warning(f"Failed to load prepared mic {prepared_path}: {e}")
                continue
            mic_bp = bandpass_voice(mic_raw, sr_m)
            mic_bp = rms_normalize(mic_bp)

            # Preferred: audalign backend
            try:
                # Align raw mic to scratch
                aligned_raw = align_with_audalign_files(scratch_wav, prepared_path, out_len=len(scratch), mode="correlation_spectrogram")
                logging.info("Aligned (raw) via audalign backend")
            except Exception as e:
                logging.error(f"audalign backend failed: {e}")
                raise

            # Clean the aligned segment with Resemble Enhance
            try:
                import tempfile as _tf
                with _tf.TemporaryDirectory() as td2:
                    seg_in = Path(td2) / "seg_raw.wav"
                    seg_out = Path(td2) / "seg_clean.wav"
                    _write_wav_mono_int16(seg_in, aligned_raw, sr)
                    clean_with_resemble(seg_in, seg_out, device=args.device)
                    aligned, _ = _read_wav_mono_any(seg_out)
                    aligned = aligned.astype(np.float32)
                    if len(aligned) != len(scratch):
                        if len(aligned) < len(scratch):
                            aligned = np.concatenate([aligned, np.zeros(len(scratch)-len(aligned), dtype=np.float32)])
                        else:
                            aligned = aligned[:len(scratch)]
                logging.info("Cleaned aligned segment with Resemble Enhance")
            except Exception as e:
                logging.error(f"Cleaning aligned segment failed, using raw aligned: {e}")
                aligned = aligned_raw
            drift_s_per_s = 0.0
            if args.drift_correct and len(aligned) > sr * 10:
                try:
                    # Re-estimate drift on the aligned window
                    aligned_bp = bandpass_voice(aligned, sr)
                    aligned_bp = rms_normalize(aligned_bp)
                    drift_s_per_s = estimate_drift(scratch_bp, aligned_bp, sr, max_offset_s=args.max_offset)
                except Exception as e:
                    logging.warning(f"Drift estimation failed: {e}")
                    drift_s_per_s = 0.0
            track_start_s = 0.0

            if args.trim_on:
                try:
                    trimmed, sidx, eidx = trim_silence_safely(
                        aligned,
                        sr,
                        pad_head_s=args.head_pad,
                        pad_tail_s=args.tail_pad,
                        vad_aggr=args.vad_aggr,
                        min_speech_s=args.min_speech,
                        min_silence_s=args.min_silence,
                    )
                    if trimmed is not aligned:
                        logging.info(f"Trim applied to {mic_name} on {vc.camera}[{vc.order_index}]: start {sidx/sr:.2f}s, end {eidx/sr:.2f}s")
                        aligned = trimmed
                        track_start_s = sidx / sr
                except Exception as e:
                    logging.warning(f"Trimming failed (skipped) for {mic_name}: {e}")

            out_aligned = aligned_dir / f"{vc.camera}_{vc.order_index:03d}_{mic_name}_aligned.wav"
            if out_aligned.exists():
                try:
                    out_aligned.unlink()
                except Exception:
                    pass
            write_aligned_wav(out_aligned, aligned, sr, stereo=False)
            # Compute track placement offset: only trimmed head (safe)
            track_offset_s = track_start_s
            aligned_tracks.append(AlignedTrack(path=out_aligned, start_offset_s=track_offset_s, duration_s=len(aligned)/sr))

            # Diagnostics per clip: write 10s reference windows and a JSON summary
            try:
                import json as _json
                ref_len = int(10 * sr)
                start_ref = max(0, int(track_start_s * sr))
                mic_seg = mic_clean[start_ref:start_ref + ref_len]
                scr_seg = scratch[start_ref:start_ref + ref_len]
                _write_wav_mono_int16(diag_dir / f"{vc.camera}_{vc.order_index:03d}_{mic_name}_ref_mic.wav", mic_seg, sr)
                _write_wav_mono_int16(diag_dir / f"{vc.camera}_{vc.order_index:03d}_{mic_name}_ref_scratch.wav", scr_seg, sr)
                diag = {
                    "clip": f"{vc.camera}_{vc.order_index:03d}",
                    "mic": mic_name,
                    "coarse_search": True if (len(mic_clips) == 1 and len(video_clips) > 1) or bool(args.global_search) else False,
                    "start_offset_s": track_start_s,
                    "refined_start_sample": int(best_start * sr * 0.02),
                    "dtw_avg_cost": float(best_cost),
                    "linear_slope": float(b_slope),
                    "linear_r2": float(r2),
                    "start_index_samples": int(start_idx),
                    "sequence_duration_s": len(scratch) / sr,
                }
                (diag_dir / f"{vc.camera}_{vc.order_index:03d}_{mic_name}_diag.json").write_text(_json.dumps(diag, indent=2), encoding="utf-8")
            except Exception:
                pass

        dur_seconds = len(scratch) / sr
        dur_frames = int(round(dur_seconds * args.timebase))
        seq_name = f"{vc.camera}_{vc.order_index:03d}"
        seq = SequenceSpec(
            name=seq_name,
            video=vc.path,
            timebase=args.timebase,
            duration_frames=dur_frames,
            aligned_wavs=aligned_tracks,
            scratch_wav=scratch_wav if args.keep_scratch else None,
            audio_offset_s=0.0,
        )
        sequences.append(seq)

        if args.write_mp4:
            out_mp4 = mp4_dir / f"{seq_name}.mp4"
            try:
                include_scratch = None if len(aligned_tracks) > 0 else (scratch_wav if args.keep_scratch else None)
                write_prepped_mp4(vc.path, aligned_tracks, out_mp4, include_scratch, audio_offset_s=0.0)
            except Exception as e:
                logging.error(f"Failed to write MP4 for {seq_name}: {e}")
        if args.write_mov_pcm:
            out_mov = mp4_dir / f"{seq_name}.mov"
            try:
                include_scratch = None if len(aligned_tracks) > 0 else (scratch_wav if args.keep_scratch else None)
                write_prepped_mov_pcm(
                    vc.path,
                    aligned_tracks,
                    out_mov,
                    include_scratch,
                    scratch_wav=scratch_wav,
                    auto_correct=bool(args.auto_mux_correct),
                    max_residual_s=max(0.5, min(5.0, args.max_offset))
                )
            except Exception as e:
                logging.error(f"Failed to write MOV(PCM) for {seq_name}: {e}")
            # Debug mux diagnostics
            if args.debug_mux:
                try:
                    import json as _json
                    in_streams = ffprobe_streams(vc.path)
                    out_streams = ffprobe_streams(out_mov)
                    report = {
                        "video_in": str(vc.path),
                        "mov_out": str(out_mov),
                        "in_streams": in_streams,
                        "out_streams": out_streams,
                        "audio_offsets_ms": [int(round(t.start_offset_s * 1000.0)) for t in aligned_tracks],
                        "num_aligned_tracks": len(aligned_tracks),
                        "include_scratch": bool(include_scratch is not None),
                    }
                    (diag_dir / f"{seq_name}_mux_report.json").write_text(_json.dumps(report, indent=2), encoding="utf-8")
                    # Also write a stereo debug WAV: L=scratch, R=first aligned track with delay applied
                    if aligned_tracks:
                        at = aligned_tracks[0]
                        # load aligned written wav
                        aligned_data, _ = _read_wav_mono_any(at.path)
                        # apply head delay
                        delay_samp = max(0, int(round(at.start_offset_s * sr)))
                        r = np.concatenate([np.zeros(delay_samp, dtype=np.float32), aligned_data])
                        r = r[: len(scratch)] if len(r) >= len(scratch) else np.pad(r, (0, len(scratch)-len(r)))
                        # interleave stereo
                        stereo = np.stack([scratch.astype(np.float32), r.astype(np.float32)], axis=1).reshape(-1)
                        # write as 16-bit stereo via ffmpeg to avoid re-implementing multichannel WAV here
                        tmp_stereo = diag_dir / f"{seq_name}_stereo_debug.wav"
                        # Use numpy to write two-channel via a quick PCM16 buffer
                        # Build interleaved int16
                        l = np.clip(stereo, -1.0, 1.0)
                        s16 = (l * 32767.0).astype(np.int16)
                        # naive writer for stereo
                        import wave as _wv
                        with _wv.open(str(tmp_stereo), 'wb') as wv:
                            wv.setnchannels(2)
                            wv.setsampwidth(2)
                            wv.setframerate(sr)
                            wv.writeframes(s16.tobytes())
                except Exception as e:
                    logging.warning(f"Failed to write mux diagnostics: {e}")

    # XML
    xml_path = xml_dir / "premiere_prepper_fcp7.xml"
    try:
        write_fcp7_xml(sequences, args.timebase, xml_path, project_name="PremierePrepper")
        logging.info(f"FCP7 XML written: {xml_path}")
    except Exception as e:
        logging.error(f"Failed to write FCP7 XML: {e}")

    # Manifest
    manifest = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "timebase": args.timebase,
        "keep_scratch": bool(args.keep_scratch),
        "write_mp4": bool(args.write_mp4),
        "trim_on": bool(args.trim_on),
        "sequences": [
            {
                "name": s.name,
                "video": str(s.video),
                "aligned_wavs": [str(p) for p in s.aligned_wavs],
                "scratch_wav": str(s.scratch_wav) if s.scratch_wav else None,
            }
            for s in sequences
        ],
        "xml": str(xml_path),
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logging.info(f"Manifest written: {manifest_path}")
    except Exception as e:
        logging.error(f"Failed to write manifest: {e}")


if __name__ == "__main__":
    main()
