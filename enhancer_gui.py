import os
import hashlib
import importlib
import json
import multiprocessing
import re
import runpy
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
# Optional modern theming via ttkbootstrap
try:
    import ttkbootstrap as _ttkb  # type: ignore[import-not-found]
    _TTKBOOT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TTKBOOT_AVAILABLE = False
import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import io as _io
import sys as _sys

# Optional drag-and-drop support via tkinterdnd2 (if available)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore[import-not-found]
    DND_AVAILABLE = True
except Exception:  # noqa: BLE001
    DND_AVAILABLE = False


BASE = Path.cwd()
APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
ICON_ROOT = APP_ROOT / "assets" / "icons"
INPUT_TMP_ROOT = BASE / ".enhancer_runs_gui"
OUTPUT_ROOT = BASE / "output_audio"
MIN_AUDIO_SAMPLES = 2048
OUTPUT_MEDIA_CLEAN_ENV = "RESEMBLE_OUTPUT_MEDIA_CLEAN"
MEDIA_ROOT_FOLDER = "01_MEDIA"
MEDIA_CLEAN_FOLDER = "030_AUDIO_CLEAN"
MEDIA_AUDIO_RAW_FOLDER = "040_AUDIO_RAW"
MEDIA_VIDEO_RAW_FOLDER = "020_VIDEO_RAW"
VIDEO_FILE_EXTENSIONS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".mts", ".m2ts"}
AI_SYNTHESIS_DEFAULT_WET = 0.35
AI_SYNTHESIS_DEFAULT_LAMBD = 0.35
AI_SYNTHESIS_DEFAULT_TAU = 0.25
AI_SYNTHESIS_DEFAULT_NFE = 32
LAST_MODEL_SR: int | None = None
LAST_MODEL_SR_PATH: str | None = None
_AUDIO_TRACKS_CACHE: dict[tuple[str, int, int, int], list] = {}
_FASTER_WHISPER_MODEL_CACHE: dict[tuple[str, str], object] = {}
_WHISPER_WINDOW_CACHE: dict[tuple[str, int, int, str, str, int, int], list[dict]] = {}
_AUDALIGN_OFFSETS_CACHE: dict[tuple, dict[str, float]] = {}
_LIVE_SUBPROCS: set[subprocess.Popen] = set()
_LIVE_SUBPROCS_LOCK = threading.Lock()
_SINGLE_INSTANCE_MUTEX_NAME = r"Global\ResembleEnhanceGUI_SingleInstance"


def _register_live_subprocess(proc: subprocess.Popen) -> None:
    try:
        with _LIVE_SUBPROCS_LOCK:
            _LIVE_SUBPROCS.add(proc)
    except Exception:
        pass


def _unregister_live_subprocess(proc: subprocess.Popen) -> None:
    try:
        with _LIVE_SUBPROCS_LOCK:
            _LIVE_SUBPROCS.discard(proc)
    except Exception:
        pass


def _terminate_live_subprocesses(timeout_s: float = 2.0) -> None:
    try:
        with _LIVE_SUBPROCS_LOCK:
            procs = list(_LIVE_SUBPROCS)
    except Exception:
        procs = []
    for proc in procs:
        try:
            if proc.poll() is not None:
                _unregister_live_subprocess(proc)
                continue
            proc.terminate()
            try:
                proc.wait(timeout=max(0.1, float(timeout_s)))
            except Exception:
                proc.kill()
        except Exception:
            pass
        finally:
            _unregister_live_subprocess(proc)


def _acquire_single_instance_mutex(name: str) -> int | None:
    if os.name != "nt":
        return 1
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.GetLastError.argtypes = []
        kernel32.GetLastError.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateMutexW(None, False, str(name))
        if not handle:
            return None
        err = int(kernel32.GetLastError() or 0)
        if err == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return None
        return int(handle)
    except Exception:
        return None


def _release_single_instance_mutex(handle: int | None) -> None:
    if os.name != "nt" or not handle:
        return
    try:
        ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle))
    except Exception:
        pass


def _show_single_instance_notice() -> None:
    msg = "Resemble Enhance is already running.\nOnly one instance can be opened at a time."
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            user32.MessageBoxW(None, msg, "Resemble Enhance", 0x00000030)  # MB_ICONWARNING
            return
        except Exception:
            pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _dispatch_frozen_module_invocation() -> None:
    """Support subprocess calls like `<frozen exe> -m module ...`.

    PyInstaller one-dir builds use the app executable as sys.executable. The GUI
    launches the enhancer CLI with `sys.executable -m resemble_enhance.enhancer`,
    so route that form before enforcing the single-instance GUI mutex.
    """
    try:
        if len(sys.argv) >= 3 and sys.argv[1] == "-m":
            module_name = str(sys.argv[2]).strip()
            if not module_name:
                return
            sys.argv = [module_name, *sys.argv[3:]]
            if module_name == "resemble_enhance.enhancer":
                from resemble_enhance.enhancer.__main__ import main as enhancer_main
                enhancer_main()
                sys.exit(0)
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
            sys.exit(0)
    except SystemExit:
        raise
    except Exception as exc:
        try:
            log_path = APP_ROOT / "module_dispatch_error.log"
            log_path.write_text(
                f"argv={sys.argv!r}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )
        except Exception:
            pass
        try:
            print(f"Module dispatch failed: {exc}", flush=True)
        except Exception:
            pass
        sys.exit(1)


def _stable_offsets_cache_key(source_paths: list[str]) -> str:
    parts: list[str] = []
    for p in source_paths:
        try:
            pp = Path(p)
            st = pp.stat()
            parts.append(f"{str(pp.resolve(strict=False))}|{int(st.st_mtime)}|{int(st.st_size)}")
        except Exception:
            parts.append(str(p))
    parts.sort()
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def _load_persistent_audalign_offsets(cache_key: str) -> dict[str, float] | None:
    try:
        f = INPUT_TMP_ROOT / "cache" / "otio_audalign_offsets.json"
        if not f.exists():
            return None
        data = json.loads(f.read_text(encoding="utf-8"))
        row = (data.get("entries", {}) or {}).get(cache_key)
        if not isinstance(row, dict):
            return None
        offs = row.get("offsets", {})
        if not isinstance(offs, dict):
            return None
        return {str(k): float(v) for k, v in offs.items()}
    except Exception:
        return None


def _save_persistent_audalign_offsets(cache_key: str, offsets: dict[str, float]) -> None:
    try:
        d = INPUT_TMP_ROOT / "cache"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "otio_audalign_offsets.json"
        data = {"entries": {}}
        if f.exists():
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {"entries": {}}
            except Exception:
                data = {"entries": {}}
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        entries[cache_key] = {"offsets": {str(k): float(v) for k, v in offsets.items()}, "ts": int(time.time())}
        # Keep file bounded.
        if len(entries) > 256:
            items = sorted(entries.items(), key=lambda kv: int((kv[1] or {}).get("ts", 0)), reverse=True)[:256]
            entries = {k: v for k, v in items}
        data["entries"] = entries
        f.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

def _format_seconds(seconds: float) -> str:
    try:
        s = float(seconds)
    except Exception:
        return "n/a"
    if s < 1.0:
        return f"{s * 1000.0:.0f} ms"
    return f"{s:.2f} s"

def _find_media_clean_dir(src_path: Path) -> Path | None:
    try:
        # First, look for an ancestor named 01_MEDIA
        for parent in src_path.parents:
            if parent.name.lower() == MEDIA_ROOT_FOLDER.lower():
                preferred = parent / MEDIA_CLEAN_FOLDER
                if preferred.exists() or preferred.parent.exists():
                    return preferred
                # Fallback: any folder containing "audio_clean" under 01_MEDIA
                try:
                    for child in parent.iterdir():
                        if child.is_dir() and "audio_clean" in child.name.lower():
                            return child
                except Exception:
                    return preferred
                return preferred
        # If no 01_MEDIA ancestor, try sibling 01_MEDIA at each parent level
        for parent in src_path.parents:
            sib = parent / MEDIA_ROOT_FOLDER
            if sib.exists() and sib.is_dir():
                preferred = sib / MEDIA_CLEAN_FOLDER
                if preferred.exists() or preferred.parent.exists():
                    return preferred
                try:
                    for child in sib.iterdir():
                        if child.is_dir() and "audio_clean" in child.name.lower():
                            return child
                except Exception:
                    return preferred
                return preferred
    except Exception:
        return None
    return None

def _build_output_dest_dir(src_path: Path, output_base: Path | None, stamp: str) -> Path:
    if output_base is not None:
        return output_base
    try:
        if os.environ.get(OUTPUT_MEDIA_CLEAN_ENV, "0") == "1":
            media_dir = _find_media_clean_dir(src_path)
            if media_dir is not None:
                return media_dir
    except Exception:
        pass
    return src_path.parent / f"Enhanced_{stamp}"


def _is_video_file(path: str | Path) -> bool:
    try:
        return Path(path).suffix.lower() in VIDEO_FILE_EXTENSIONS
    except Exception:
        return False

def _has_clean_output_in_media(src_path: Path) -> bool:
    try:
        # If source is already a CLEAN file or lives in the clean folder, skip
        if src_path.name.upper().startswith("CLEAN_"):
            return True
        try:
            if src_path.parent.name.lower() == MEDIA_CLEAN_FOLDER.lower():
                return True
            if "audio_clean" in src_path.parent.name.lower():
                return True
        except Exception:
            pass
        media_dir = _find_media_clean_dir(src_path)
        if media_dir is None or not media_dir.exists():
            return False
        # If a synced multichannel clean already exists for this folder, skip all
        try:
            for cand in media_dir.glob("CLEAN_Synced_Multichannel*"):
                if cand.is_file():
                    return True
        except Exception:
            pass
        stem = src_path.stem
        pat = f"CLEAN_{stem}*"
        for cand in media_dir.glob(pat):
            try:
                if cand.is_file():
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False

def _find_existing_clean_for_source(src_path: Path) -> Path | None:
    """Find an existing CLEAN file that corresponds to a source file."""
    try:
        if src_path.name.upper().startswith("CLEAN_") and src_path.exists():
            return src_path
        stem = src_path.stem
        matches: list[Path] = []
        # Probe multiple plausible locations where CLEAN files may exist.
        cand_dirs: list[Path] = []
        media_dir = _find_media_clean_dir(src_path)
        if media_dir is not None:
            cand_dirs.append(media_dir)
        cand_dirs.append(src_path.parent)
        try:
            # If source is in 040_AUDIO_RAW, also probe sibling 030_AUDIO_CLEAN.
            if src_path.parent.name.lower() == MEDIA_AUDIO_RAW_FOLDER.lower():
                cand_dirs.append(src_path.parent.parent / MEDIA_CLEAN_FOLDER)
        except Exception:
            pass
        # Deduplicate and search.
        seen_dirs: set[str] = set()
        uniq_dirs: list[Path] = []
        for d in cand_dirs:
            try:
                k = str(d.resolve()).lower()
            except Exception:
                k = str(d).lower()
            if k in seen_dirs:
                continue
            seen_dirs.add(k)
            uniq_dirs.append(d)

        # First-pass strict: CLEAN_<exact-stem>* in candidate dirs.
        strict_pat = f"CLEAN_{stem}*"
        for d in uniq_dirs:
            try:
                if not d.exists() or not d.is_dir():
                    continue
                for cand in d.glob(strict_pat):
                    try:
                        if cand.is_file():
                            matches.append(cand)
                    except Exception:
                        continue
            except Exception:
                continue

        # Fallback fuzzy: CLEAN files containing source stem token.
        if not matches:
            stem_l = stem.lower()
            for d in uniq_dirs:
                try:
                    if not d.exists() or not d.is_dir():
                        continue
                    for cand in d.glob("CLEAN_*"):
                        try:
                            if not cand.is_file():
                                continue
                            if stem_l in cand.stem.lower():
                                matches.append(cand)
                        except Exception:
                            continue
                except Exception:
                    continue

        # Final fallback: if there is exactly one CLEAN*.mov in candidate dirs,
        # use it as a reusable synced source (common dual-mono workflow).
        if not matches:
            movs: list[Path] = []
            for d in uniq_dirs:
                try:
                    if not d.exists() or not d.is_dir():
                        continue
                    for cand in d.glob("CLEAN*.mov"):
                        try:
                            if cand.is_file():
                                movs.append(cand)
                        except Exception:
                            continue
                except Exception:
                    continue
            if len(movs) == 1:
                matches = movs

        if not matches:
            return None
        # Prefer MOV first (synced dual-/multi-mono workflows), then newest.
        def _rank(p: Path) -> tuple[int, float]:
            try:
                mtime = p.stat().st_mtime if p.exists() else 0.0
            except Exception:
                mtime = 0.0
            is_mov = 1 if p.suffix.lower() == ".mov" else 0
            return (is_mov, mtime)
        matches.sort(key=_rank, reverse=True)
        return matches[0]
    except Exception:
        return None


def _find_reusable_synced_clean_for_group(group_files: list[str]) -> Path | None:
    """Find a reusable synced CLEAN MOV for a group of source files."""
    try:
        cand_dirs: list[Path] = []
        for fp in group_files:
            src_path = Path(fp)
            media_dir = _find_media_clean_dir(src_path)
            if media_dir is not None:
                cand_dirs.append(media_dir)
            cand_dirs.append(src_path.parent)
            try:
                if src_path.parent.name.lower() == MEDIA_AUDIO_RAW_FOLDER.lower():
                    cand_dirs.append(src_path.parent.parent / MEDIA_CLEAN_FOLDER)
            except Exception:
                pass

        seen_dirs: set[str] = set()
        uniq_dirs: list[Path] = []
        for d in cand_dirs:
            try:
                k = str(d.resolve()).lower()
            except Exception:
                k = str(d).lower()
            if k in seen_dirs:
                continue
            seen_dirs.add(k)
            uniq_dirs.append(d)

        movs: list[Path] = []
        for d in uniq_dirs:
            try:
                if not d.exists() or not d.is_dir():
                    continue
                for cand in d.glob("CLEAN_Synced_Multichannel*.mov"):
                    try:
                        if cand.is_file():
                            movs.append(cand)
                    except Exception:
                        continue
            except Exception:
                continue
        if not movs:
            for d in uniq_dirs:
                try:
                    if not d.exists() or not d.is_dir():
                        continue
                    for cand in d.glob("CLEAN*.mov"):
                        try:
                            if cand.is_file():
                                movs.append(cand)
                        except Exception:
                            continue
                except Exception:
                    continue
        if not movs:
            return None

        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except Exception:
                return 0.0

        movs.sort(key=_mtime, reverse=True)
        return movs[0]
    except Exception:
        return None


def _find_synced_clean_for_group(group_files: list[str]) -> Path | None:
    """Find newest synced CLEAN multichannel source for a group (MOV preferred, then WAV)."""
    try:
        cand_dirs: list[Path] = []
        for fp in group_files:
            src_path = Path(fp)
            media_dir = _find_media_clean_dir(src_path)
            if media_dir is not None:
                cand_dirs.append(media_dir)
            cand_dirs.append(src_path.parent)
            try:
                if src_path.parent.name.lower() == MEDIA_AUDIO_RAW_FOLDER.lower():
                    cand_dirs.append(src_path.parent.parent / MEDIA_CLEAN_FOLDER)
            except Exception:
                pass
        seen_dirs: set[str] = set()
        uniq_dirs: list[Path] = []
        for d in cand_dirs:
            try:
                k = str(d.resolve()).lower()
            except Exception:
                k = str(d).lower()
            if k in seen_dirs:
                continue
            seen_dirs.add(k)
            uniq_dirs.append(d)

        movs: list[Path] = []
        wavs: list[Path] = []
        for d in uniq_dirs:
            try:
                if not d.exists() or not d.is_dir():
                    continue
                for cand in d.glob("CLEAN_Synced_Multichannel*.mov"):
                    if cand.is_file():
                        movs.append(cand)
                for cand in d.glob("CLEAN_Synced_Multichannel*.wav"):
                    if cand.is_file():
                        wavs.append(cand)
            except Exception:
                continue

        def _mtime(p: Path) -> float:
            try:
                return float(p.stat().st_mtime)
            except Exception:
                return 0.0

        if movs:
            movs.sort(key=_mtime, reverse=True)
            return movs[0]
        if wavs:
            wavs.sort(key=_mtime, reverse=True)
            return wavs[0]
        return None
    except Exception:
        return None


def _build_output_name(src_path: Path, stamp: str) -> str:
    stem = src_path.stem
    suf = src_path.suffix
    return f"CLEAN_{stem}_{stamp}{suf}"

def _pad_audio_tensor_min_samples(wav, min_samples: int) -> tuple:
    try:
        import torch
        if not isinstance(wav, torch.Tensor):
            return wav, 0
        n = int(wav.numel())
        if n >= int(min_samples):
            return wav, n
        pad = int(min_samples) - n
        wav = torch.nn.functional.pad(wav, (0, pad))
        return wav, n
    except Exception:
        return wav, 0

def _pad_wav_on_disk(path: Path, min_samples: int) -> tuple[int, int] | None:
    try:
        import torchaudio
        info = torchaudio.info(str(path))
        orig_len = int(getattr(info, "num_frames", 0) or 0)
        sr = int(getattr(info, "sample_rate", 0) or 0)
        if orig_len <= 0 or sr <= 0:
            return None
        if orig_len >= int(min_samples):
            return (orig_len, sr)
        wav, sr2 = torchaudio.load(str(path))
        if sr2:
            sr = int(sr2)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        pad = int(min_samples) - int(wav.size(-1))
        if pad > 0:
            import torch as _t
            wav = _t.nn.functional.pad(wav, (0, pad))
            torchaudio.save(str(path), wav, sr)
        return (orig_len, sr)
    except Exception:
        return None

def _prune_staging_dirs(max_age_hours: float = 24.0) -> int:
    """Delete old temp staging subfolders under .enhancer_runs_gui.

    Returns number of folders removed.
    """
    try:
        root = INPUT_TMP_ROOT
        if not root.exists():
            return 0
        now = time.time()
        removed = 0
        for p in list(root.iterdir()):
            try:
                if not p.is_dir():
                    continue
                age_h = (now - p.stat().st_mtime) / 3600.0
                if age_h >= float(max_age_hours):
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
            except Exception:
                # ignore errors on per-entry basis
                pass
        return removed
    except Exception:
        return 0

def _cleanup_run_artifacts(remove_logs: bool = True) -> None:
    """Remove temp audio/log artifacts under .enhancer_runs_gui."""
    try:
        root = INPUT_TMP_ROOT
        if not root.exists():
            return
        # Known temp subfolders
        for name in ("staging", "tmp_sync", "preview_out"):
            try:
                shutil.rmtree(root / name, ignore_errors=True)
            except Exception:
                pass
        if remove_logs:
            try:
                shutil.rmtree(root / "run_logs", ignore_errors=True)
            except Exception:
                pass
        # Remove any leftover run_* folders or stray run ids
        try:
            for p in list(root.iterdir()):
                if not p.is_dir():
                    continue
                if p.name in {"staging", "tmp_sync", "preview_out", "run_logs"}:
                    continue
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    except Exception:
        pass

def _find_free_port(host: str = "127.0.0.1") -> int:
    import socket as _sock
    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
    s.bind((host, 0))
    addr, port = s.getsockname()
    try:
        s.close()
    except Exception:
        pass
    return int(port)


def _load_env_files() -> None:
    """Load dotenv-style variables from 'env' or '.env' at repo root.
    Only sets keys not already present in os.environ.
    """
    paths = [BASE / "env", BASE / ".env"]
    for p in paths:
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k and (k not in os.environ):
                    os.environ[k] = v
        except Exception:
            pass


# Load env overrides at import time so subprocess inherits them too
_load_env_files()


def _apply_peak_ceiling(wav_t, ceiling_db: float = -1.0):
    """Return a version of wav_t scaled so its absolute peak <= ceiling.
    Accepts a 1D or 2D torch tensor; returns original on failure.
    """
    try:
        import torch  # local import to avoid global dependency when unused
        if not isinstance(wav_t, torch.Tensor):
            return wav_t
        ceiling = 10 ** (ceiling_db / 20.0)
        peak = float(wav_t.abs().max().item())
        if peak > ceiling and peak > 0:
            return wav_t * (ceiling / peak)
    except Exception:
        pass
    return wav_t


def _adaptive_transient_blend(proc, orig, sr: int, strength: float = 0.5):
    """Mix some original back during sharp, loud mismatches to avoid choppy artifacts.
    - proc, orig: 1D torch tensors at same sample rate and length
    - strength: 0..1 fraction of how much of the computed mask to apply
    Returns tensor same shape as inputs.
    """
    try:
        import torch
        if not (isinstance(proc, torch.Tensor) and isinstance(orig, torch.Tensor)):
            return proc
        if proc.dim() != 1:
            proc = proc.view(-1)
        if orig.dim() != 1:
            orig = orig.view(-1)
        n = min(proc.numel(), orig.numel())
        x = proc[:n]
        y = orig[:n]
        # Mismatch measure smoothed over ~5 ms
        k = max(8, int(sr * 0.005))
        pad = k // 2
        d = (x - y).abs().unsqueeze(0).unsqueeze(0)
        w = torch.ones(1, 1, k, dtype=d.dtype, device=d.device) / float(k)
        d_s = torch.nn.functional.conv1d(d, w, padding=pad).squeeze()
        # Threshold relative to robust median
        med = torch.quantile(d_s, 0.5)
        thr = med * 6.0
        # Soft mask where mismatch is above threshold and level is loud
        lvl = torch.maximum(x.abs(), y.abs())
        lvl_thr = 10 ** (-12.0 / 20.0)  # -12 dBFS
        m1 = torch.clamp((d_s - thr) / (thr + 1e-8), 0.0, 1.0)
        m2 = (lvl > lvl_thr).to(m1.dtype)
        mask = (m1 * m2)  # 0..1
        # Smooth mask over ~20 ms to avoid flutter
        k2 = max(16, int(sr * 0.02))
        pad2 = k2 // 2
        w2 = torch.ones(1, 1, k2, dtype=d.dtype, device=d.device) / float(k2)
        mask_s = torch.nn.functional.conv1d(mask.unsqueeze(0).unsqueeze(0), w2, padding=pad2).squeeze()
        mask_s = torch.clamp(mask_s, 0.0, 1.0) * float(max(0.0, min(1.0, strength)))
        out = x * (1.0 - mask_s) + y * mask_s
        if out.numel() < proc.numel():
            # pad tail unchanged if needed
            tail = proc[out.numel():]
            out = torch.cat([out, tail], dim=0)
        return out
    except Exception:
        return proc


def _compute_rms_envelope(mono, sr: int, win_ms: float = 25.0, hop_ms: float = 10.0):
    """Return RMS envelope sampled every hop_ms milliseconds."""
    import torch
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if mono.numel() < win:
        return torch.sqrt(torch.mean(mono * mono) + 1e-12).unsqueeze(0)
    kernel = torch.ones(1, 1, win, dtype=mono.dtype, device=mono.device) / float(win)
    x = mono.unsqueeze(0).unsqueeze(0)  # [1,1,T]
    rms = torch.nn.functional.conv1d(x * x, kernel, stride=hop)
    return torch.sqrt(rms.squeeze(0).squeeze(0) + 1e-12)


def _smooth_gain_envelope(env, attack_ms: float, release_ms: float, sr: int, hop_ms: float):
    """Apply separate attack/release smoothing to a per-frame envelope."""
    import math
    import torch

    hop = max(1, int(sr * hop_ms / 1000.0))
    attack_tc = max(1e-3, attack_ms / 1000.0)
    release_tc = max(1e-3, release_ms / 1000.0)
    # Convert time constants to smoothing coefficients
    alpha_a = math.exp(-hop / (sr * attack_tc))
    alpha_r = math.exp(-hop / (sr * release_tc))
    out = torch.empty_like(env)
    prev = float(env[0]) if env.numel() else 0.0
    for i in range(env.numel()):
        v = float(env[i])
        coeff = alpha_a if v > prev else alpha_r
        prev = coeff * prev + (1.0 - coeff) * v
        out[i] = prev
    return out


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


@dataclass
class _BleedGateConfig:
    win_ms: float = 25.0
    hop_ms: float = 10.0
    silence_db: float = -62.0
    activity_db: float = 7.0
    start_db: float = 3.0
    full_db: float = 9.0
    max_att_db: float = 24.0
    min_att_db: float = 16.0
    min_att_conf: float = 0.35
    hold_ms: float = 120.0
    switch_db: float = 2.5
    overlap_margin_db: float = 2.0
    attack_ms: float = 18.0
    release_ms: float = 260.0
    upsample_smooth_ms: float = 10.0
    contender_frames: int = 3
    score_smooth_frames: int = 3
    lookahead_ms: float = 30.0
    spectral_sim_low: float = 0.55
    spectral_sim_high: float = 0.86
    spectral_bleed_sim: float = 0.90
    hard_isolation: bool = True
    hard_att_db: float = 42.0
    hard_doubletalk_sim_max: float = 0.72
    hard_mute: bool = True
    hard_mute_attack_ms: float = 8.0
    hard_mute_release_ms: float = 45.0
    hard_open_pad_ms: float = 12.0
    hard_open_hold_ms: float = 45.0

    @classmethod
    def from_env(cls) -> "_BleedGateConfig":
        cfg = cls()
        cfg.start_db = max(0.1, _env_float("RESEMBLE_BLEED_START_DB", cfg.start_db))
        cfg.full_db = max(cfg.start_db + 0.25, _env_float("RESEMBLE_BLEED_FULL_DB", cfg.full_db))
        cfg.max_att_db = max(1.0, _env_float("RESEMBLE_BLEED_MAX_ATT_DB", cfg.max_att_db))
        cfg.min_att_db = max(0.0, _env_float("RESEMBLE_BLEED_MIN_ATT_DB", cfg.min_att_db))
        cfg.min_att_conf = min(1.0, max(0.0, _env_float("RESEMBLE_BLEED_MIN_ATT_CONF", cfg.min_att_conf)))
        cfg.hold_ms = max(0.0, _env_float("RESEMBLE_BLEED_HOLD_MS", cfg.hold_ms))
        cfg.switch_db = max(0.0, _env_float("RESEMBLE_BLEED_SWITCH_DB", cfg.switch_db))
        cfg.overlap_margin_db = max(0.0, _env_float("RESEMBLE_BLEED_OVERLAP_MARGIN_DB", cfg.overlap_margin_db))
        cfg.attack_ms = max(2.0, _env_float("RESEMBLE_BLEED_ATTACK_MS", cfg.attack_ms))
        cfg.release_ms = max(10.0, _env_float("RESEMBLE_BLEED_RELEASE_MS", cfg.release_ms))
        cfg.contender_frames = max(1, int(round(_env_float("RESEMBLE_BLEED_CONTENDER_FRAMES", float(cfg.contender_frames)))))
        cfg.score_smooth_frames = max(1, int(round(_env_float("RESEMBLE_BLEED_SCORE_SMOOTH_FRAMES", float(cfg.score_smooth_frames)))))
        cfg.lookahead_ms = max(0.0, _env_float("RESEMBLE_BLEED_LOOKAHEAD_MS", cfg.lookahead_ms))
        cfg.spectral_sim_low = min(0.99, max(0.0, _env_float("RESEMBLE_BLEED_SPECTRAL_SIM_LOW", cfg.spectral_sim_low)))
        cfg.spectral_sim_high = min(0.999, max(cfg.spectral_sim_low + 1e-3, _env_float("RESEMBLE_BLEED_SPECTRAL_SIM_HIGH", cfg.spectral_sim_high)))
        cfg.spectral_bleed_sim = min(0.999, max(cfg.spectral_sim_low, _env_float("RESEMBLE_BLEED_SPECTRAL_BLEED_SIM", cfg.spectral_bleed_sim)))
        cfg.hard_isolation = _env_bool("RESEMBLE_BLEED_HARD_ISOLATION", cfg.hard_isolation)
        cfg.hard_att_db = max(6.0, _env_float("RESEMBLE_BLEED_HARD_ATT_DB", cfg.hard_att_db))
        cfg.hard_doubletalk_sim_max = min(0.99, max(0.0, _env_float("RESEMBLE_BLEED_HARD_DOUBLETALK_SIM_MAX", cfg.hard_doubletalk_sim_max)))
        cfg.hard_mute = _env_bool("RESEMBLE_BLEED_HARD_MUTE", cfg.hard_mute)
        cfg.hard_mute_attack_ms = max(0.5, _env_float("RESEMBLE_BLEED_HARD_MUTE_ATTACK_MS", cfg.hard_mute_attack_ms))
        cfg.hard_mute_release_ms = max(2.0, _env_float("RESEMBLE_BLEED_HARD_MUTE_RELEASE_MS", cfg.hard_mute_release_ms))
        cfg.hard_open_pad_ms = max(0.0, _env_float("RESEMBLE_BLEED_HARD_OPEN_PAD_MS", cfg.hard_open_pad_ms))
        cfg.hard_open_hold_ms = max(0.0, _env_float("RESEMBLE_BLEED_HARD_OPEN_HOLD_MS", cfg.hard_open_hold_ms))
        return cfg


def _analyze_speaker_activity(monos: list, sr: int, cfg: _BleedGateConfig | None = None):
    """Shared confidence/winner analysis used by bleed-gate and OTIO cut logic."""
    try:
        import torch
    except Exception:
        return None
    try:
        from torchaudio.functional import highpass_biquad, lowpass_biquad
    except Exception:
        highpass_biquad = None
        lowpass_biquad = None
    if len(monos) < 2:
        return None
    lengths = [int(m.size(-1)) for m in monos if isinstance(m, torch.Tensor)]
    if not lengths:
        return None
    T = max(lengths)
    if T <= 0:
        return None
    cfg = cfg or _BleedGateConfig.from_env()
    win_ms = cfg.win_ms
    hop_ms = cfg.hop_ms

    padded: list[torch.Tensor] = []
    for mono in monos:
        if not isinstance(mono, torch.Tensor):
            return None
        if mono.size(-1) < T:
            mono = torch.nn.functional.pad(mono, (0, T - mono.size(-1)))
        padded.append(mono)

    def _voice_env(m: torch.Tensor) -> torch.Tensor:
        if highpass_biquad is None or lowpass_biquad is None:
            return _compute_rms_envelope(m, sr, win_ms=win_ms, hop_ms=hop_ms)
        try:
            sig = m.unsqueeze(0)
            sig = highpass_biquad(sig, sr, cutoff_freq=180.0, Q=0.707)
            sig = lowpass_biquad(sig, sr, cutoff_freq=4200.0, Q=0.707)
            sig = sig.squeeze(0)
            return _compute_rms_envelope(sig, sr, win_ms=win_ms, hop_ms=hop_ms)
        except Exception:
            return _compute_rms_envelope(m, sr, win_ms=win_ms, hop_ms=hop_ms)

    def _forward_mean_1d(x: torch.Tensor, look: int) -> torch.Tensor:
        if look <= 0 or x.numel() <= 1:
            return x
        n = int(x.numel())
        csum = torch.cumsum(torch.cat([torch.zeros(1, dtype=x.dtype, device=x.device), x], dim=0), dim=0)
        idx = torch.arange(n, device=x.device, dtype=torch.long)
        end = torch.clamp(idx + look + 1, max=n)
        den = (end - idx).to(x.dtype)
        return (csum[end] - csum[idx]) / torch.clamp(den, min=1.0)

    def _forward_mean_2d(x: torch.Tensor, look: int) -> torch.Tensor:
        if look <= 0 or x.size(-1) <= 1:
            return x
        n = int(x.size(-1))
        z = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        csum = torch.cumsum(torch.cat([z, x], dim=1), dim=1)
        idx = torch.arange(n, device=x.device, dtype=torch.long)
        end = torch.clamp(idx + look + 1, max=n)
        den = (end - idx).to(x.dtype).unsqueeze(0)
        return (csum[:, end] - csum[:, idx]) / torch.clamp(den, min=1.0)

    def _spectral_vectors(chans: list[torch.Tensor]) -> torch.Tensor | None:
        if highpass_biquad is None or lowpass_biquad is None:
            return None
        bands = [(120.0, 420.0), (420.0, 1800.0), (1800.0, 4200.0)]
        all_ch: list[torch.Tensor] = []
        for m in chans:
            per_band: list[torch.Tensor] = []
            try:
                sig = m.unsqueeze(0)
                for lo, hi in bands:
                    b = highpass_biquad(sig, sr, cutoff_freq=float(lo), Q=0.707)
                    b = lowpass_biquad(b, sr, cutoff_freq=float(hi), Q=0.707)
                    per_band.append(_compute_rms_envelope(b.squeeze(0), sr, win_ms=win_ms, hop_ms=hop_ms))
            except Exception:
                return None
            all_ch.append(torch.stack(per_band, dim=0))
        spec = torch.stack(all_ch, dim=0) + 1e-9
        denom = torch.clamp(spec.sum(dim=1, keepdim=True), min=1e-9)
        return spec / denom

    wide_envs = [_compute_rms_envelope(m, sr, win_ms=win_ms, hop_ms=hop_ms) for m in padded]
    voice_envs = [_voice_env(m) for m in padded]
    wide = torch.stack(wide_envs, dim=0) + 1e-9
    venv = torch.stack(voice_envs, dim=0) + 1e-9
    env = (0.20 * wide) + (0.80 * venv)
    env_db = 20.0 * torch.log10(env)
    wide_db = 20.0 * torch.log10(wide)
    n_ch, n_frames = int(env_db.size(0)), int(env_db.size(1))

    def _q10(x: torch.Tensor) -> float:
        try:
            return float(torch.quantile(x.detach(), 0.1))
        except Exception:
            vals, _ = torch.sort(x.detach())
            if vals.numel() <= 1:
                return float(vals[0]) if vals.numel() else -90.0
            idx = int(max(0, min(vals.numel() - 1, round(0.1 * (vals.numel() - 1)))))
            return float(vals[idx])

    noise_floor = torch.tensor([_q10(env_db[i]) for i in range(n_ch)], dtype=env_db.dtype, device=env_db.device)
    active = env_db > (noise_floor.unsqueeze(1) + cfg.activity_db)
    has_active = active.any(dim=0)
    active_count = active.sum(dim=0)
    multi_active = active_count >= 2
    max_env_db, _ = env_db.max(dim=0)
    near_silence = (~has_active) | (max_env_db < cfg.silence_db)

    score = (0.85 * env_db) + (0.15 * wide_db)
    if cfg.score_smooth_frames > 1 and n_frames > 1:
        k = int(cfg.score_smooth_frames)
        if k % 2 == 0:
            k += 1
        kernel = torch.ones(1, 1, k, dtype=score.dtype, device=score.device) / float(k)
        score = torch.nn.functional.conv1d(score.unsqueeze(1), kernel, padding=k // 2).squeeze(1)
    lookahead_frames = max(0, int(round(cfg.lookahead_ms / max(1e-3, cfg.hop_ms))))
    score_look = _forward_mean_2d(score, lookahead_frames)
    neg_inf = torch.full_like(score, -1e9)
    active_score = torch.where(active, score_look, neg_inf)
    _, top1_idx = active_score.max(dim=0)
    active_score_2 = active_score.clone()
    active_score_2.scatter_(0, top1_idx.unsqueeze(0), -1e9)
    _, top2_idx = active_score_2.max(dim=0)

    winner_env = env_db.gather(0, top1_idx.unsqueeze(0)).squeeze(0)
    runner_env = env_db.gather(0, top2_idx.unsqueeze(0)).squeeze(0)
    single_ref = noise_floor[top1_idx]
    margin_db = torch.where(multi_active, winner_env - runner_env, winner_env - single_ref)
    margin_db = torch.where(has_active, margin_db, torch.zeros_like(margin_db))
    margin_db_look = _forward_mean_1d(margin_db, lookahead_frames)

    spec_sim12 = None
    spec = _spectral_vectors(padded)
    if spec is not None and int(spec.size(-1)) == n_frames:
        s = spec.permute(2, 0, 1)
        idxf = torch.arange(n_frames, device=env_db.device)
        v1 = s[idxf, top1_idx]
        v2 = s[idxf, top2_idx]
        n1 = torch.sqrt(torch.sum(v1 * v1, dim=1) + 1e-9)
        n2 = torch.sqrt(torch.sum(v2 * v2, dim=1) + 1e-9)
        spec_sim12 = torch.clamp(torch.sum(v1 * v2, dim=1) / (n1 * n2), 0.0, 1.0)
    if spec_sim12 is None:
        spec_sim12 = torch.zeros_like(margin_db)

    base_uncertain_overlap = multi_active & (margin_db_look < cfg.overlap_margin_db)
    uncertain_overlap = base_uncertain_overlap
    if cfg.hard_isolation:
        if spec is not None:
            uncertain_overlap = base_uncertain_overlap & (spec_sim12 <= cfg.hard_doubletalk_sim_max)
        else:
            uncertain_overlap = multi_active & (margin_db_look < max(0.5, cfg.overlap_margin_db * 0.35))

    conf = (margin_db_look - cfg.start_db) / max(1e-6, (cfg.full_db - cfg.start_db))
    conf = torch.clamp(conf, 0.0, 1.0)
    if spec is not None:
        sim_range = max(1e-6, cfg.spectral_sim_high - cfg.spectral_sim_low)
        sim_boost = torch.clamp((spec_sim12 - cfg.spectral_sim_low) / sim_range, 0.0, 1.0)
        conf = torch.clamp((0.75 * conf) + (0.25 * sim_boost), 0.0, 1.0)
    conf = torch.where(uncertain_overlap | near_silence, torch.zeros_like(conf), conf)

    winner_idx = top1_idx.clone()
    hold_frames = max(1, int(round(cfg.hold_ms / max(1e-3, cfg.hop_ms))))
    contender_need = max(1, int(cfg.contender_frames))
    switches = 0
    prev_winner = -1
    hold_left = 0
    contender = -1
    contender_count = 0
    for t in range(n_frames):
        cand = int(top1_idx[t].item())
        if bool(near_silence[t].item()):
            winner_idx[t] = cand if prev_winner < 0 else prev_winner
            contender = -1
            contender_count = 0
            continue
        if prev_winner < 0:
            prev_winner = cand
            hold_left = hold_frames
            winner_idx[t] = cand
            continue
        if cand == prev_winner:
            winner_idx[t] = prev_winner
            contender = -1
            contender_count = 0
            if hold_left > 0:
                hold_left -= 1
            continue
        cand_vs_prev = float(env_db[cand, t] - env_db[prev_winner, t])
        if hold_left > 0 and cand_vs_prev < cfg.switch_db:
            winner_idx[t] = prev_winner
            conf[t] = conf[t] * 0.35
            hold_left -= 1
            contender = -1
            contender_count = 0
            continue
        if cand_vs_prev >= cfg.switch_db:
            if contender == cand:
                contender_count += 1
            else:
                contender = cand
                contender_count = 1
            if contender_count >= contender_need:
                prev_winner = cand
                winner_idx[t] = cand
                hold_left = hold_frames
                contender = -1
                contender_count = 0
                switches += 1
            else:
                winner_idx[t] = prev_winner
                conf[t] = conf[t] * 0.5
        else:
            winner_idx[t] = prev_winner
            conf[t] = conf[t] * 0.35
            contender = -1
            contender_count = 0
            if hold_left > 0:
                hold_left -= 1

    return {
        "cfg": cfg,
        "padded": padded,
        "T": T,
        "n_ch": n_ch,
        "env_db": env_db,
        "active": active,
        "near_silence": near_silence,
        "uncertain_overlap": uncertain_overlap,
        "winner_idx": winner_idx,
        "conf": conf,
        "gains_template": torch.zeros_like(env_db),
        "switches": switches,
        "hop_ms": hop_ms,
    }


def _load_audio_tracks_any(path: str | Path, target_sr: int = 48000):
    """Load audio tracks from WAV/MOV/etc. Returns list of mono tensors at target_sr."""
    try:
        import torchaudio
        import torch
        from torchaudio.functional import resample as ta_resample
    except Exception:
        return []
    p = Path(path)
    try:
        rp = str(p.resolve(strict=False))
        st = p.stat()
        ckey = (rp, int(target_sr), int(st.st_mtime), int(st.st_size))
        cached = _AUDIO_TRACKS_CACHE.get(ckey)
        if cached is not None:
            return cached
    except Exception:
        ckey = None

    def _wav_to_tracks(wav, sr) -> list:
        if wav is None or sr is None:
            return []
        try:
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            tracks = []
            for ch in range(int(wav.size(0))):
                mono = wav[ch].to(torch.float32)
                if int(sr) != int(target_sr):
                    mono = ta_resample(mono, orig_freq=int(sr), new_freq=int(target_sr))
                tracks.append(mono)
            return tracks
        except Exception:
            return []

    # Fast-path: direct loader.
    try:
        wav, sr = torchaudio.load(str(p))
        tracks = _wav_to_tracks(wav, sr)
        if len(tracks) >= 2:
            if ckey is not None:
                _AUDIO_TRACKS_CACHE[ckey] = tracks
            return tracks
    except Exception:
        pass

    ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    fp = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ff:
        return []

    # Probe audio stream layout for robust extraction of multi-stream / multi-channel containers.
    streams = []
    if fp:
        try:
            probe_cmd = [fp, "-v", "error", "-show_streams", "-of", "json", str(p)]
            probe = subprocess.run(probe_cmd, capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout:
                parsed = json.loads(probe.stdout)
                all_streams = parsed.get("streams", []) if isinstance(parsed, dict) else []
                for s in all_streams:
                    try:
                        if str(s.get("codec_type", "")).lower() != "audio":
                            continue
                        streams.append(
                            {
                                "index": int(s.get("index", 0)),
                                "channels": int(s.get("channels", 1) or 1),
                            }
                        )
                    except Exception:
                        continue
        except Exception:
            streams = []

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            extracted_tracks: list = []

            # Strategy A: multiple audio streams -> one mono track per stream.
            if len(streams) >= 2:
                for i in range(len(streams)):
                    tmp_wav = tdir / f"stream_{i}.wav"
                    cmd = [
                        ff,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(p),
                        "-map",
                        f"0:a:{i}",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(tmp_wav),
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if proc.returncode != 0 or not tmp_wav.exists():
                        continue
                    wav_i, sr_i = torchaudio.load(str(tmp_wav))
                    tracks_i = _wav_to_tracks(wav_i, sr_i)
                    if tracks_i:
                        extracted_tracks.append(tracks_i[0])
                if len(extracted_tracks) >= 2:
                    if ckey is not None:
                        _AUDIO_TRACKS_CACHE[ckey] = extracted_tracks
                    return extracted_tracks

            # Strategy B: single stream with multiple channels -> split each channel.
            ch_count = 0
            if streams:
                try:
                    ch_count = max(1, int(streams[0].get("channels", 1)))
                except Exception:
                    ch_count = 0
            if ch_count >= 2:
                for ch in range(ch_count):
                    tmp_wav = tdir / f"ch_{ch}.wav"
                    cmd = [
                        ff,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(p),
                        "-filter_complex",
                        f"[0:a:0]pan=mono|c0=c{ch}[aout]",
                        "-map",
                        "[aout]",
                        "-c:a",
                        "pcm_s16le",
                        str(tmp_wav),
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if proc.returncode != 0 or not tmp_wav.exists():
                        continue
                    wav_i, sr_i = torchaudio.load(str(tmp_wav))
                    tracks_i = _wav_to_tracks(wav_i, sr_i)
                    if tracks_i:
                        extracted_tracks.append(tracks_i[0])
                if len(extracted_tracks) >= 2:
                    if ckey is not None:
                        _AUDIO_TRACKS_CACHE[ckey] = extracted_tracks
                    return extracted_tracks

            # Final fallback: single decode (may be mono/stereo depending on container/decoder).
            tmp_wav = tdir / "audio_extract.wav"
            cmd = [
                ff,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(p),
                "-vn",
                "-c:a",
                "pcm_s16le",
                str(tmp_wav),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if proc.returncode == 0 and tmp_wav.exists():
                wav, sr = torchaudio.load(str(tmp_wav))
                tracks = _wav_to_tracks(wav, sr)
                if ckey is not None:
                    _AUDIO_TRACKS_CACHE[ckey] = tracks
                return tracks
    except Exception:
        return []
    return []


def _apply_bleed_gate(monos: list, sr: int) -> list:
    """Attenuate bleed with confidence + hysteresis winner selection across N channels."""
    try:
        import torch
    except Exception:
        return monos
    analysis = _analyze_speaker_activity(monos, sr)
    if not analysis:
        return monos
    cfg = analysis["cfg"]
    padded = analysis["padded"]
    T = int(analysis["T"])
    n_ch = int(analysis["n_ch"])
    env_db = analysis["env_db"]
    active = analysis["active"]
    near_silence = analysis["near_silence"]
    uncertain_overlap = analysis["uncertain_overlap"]
    winner_idx = analysis["winner_idx"]
    conf = analysis["conf"]
    switches = int(analysis["switches"])
    hop_ms = float(analysis["hop_ms"])

    gains_db = torch.zeros_like(env_db)
    hard_loser_masks: list[torch.Tensor] = []
    for idx in range(n_ch):
        loser_mask = (winner_idx != idx) & (~uncertain_overlap) & (~near_silence)
        hard_loser_masks.append(loser_mask)
        base_att = cfg.max_att_db * conf
        floor_mask = conf >= cfg.min_att_conf
        eff_att = torch.where(floor_mask, torch.maximum(base_att, torch.full_like(base_att, cfg.min_att_db)), base_att)
        if cfg.hard_isolation:
            eff_att = torch.maximum(eff_att, torch.full_like(eff_att, cfg.hard_att_db))
        gains_db[idx] = torch.where(loser_mask, -eff_att, torch.zeros_like(conf))

    gains_lin: list[torch.Tensor] = []
    for idx in range(env_db.size(0)):
        if cfg.hard_isolation and cfg.hard_mute:
            open_mask = ~hard_loser_masks[idx]
            pad_frames = max(0, int(round(cfg.hard_open_pad_ms / max(1e-3, hop_ms))))
            if pad_frames > 0 and open_mask.numel() > 1:
                k = (2 * pad_frames) + 1
                kern = torch.ones(1, 1, k, dtype=conf.dtype, device=conf.device)
                dil = torch.nn.functional.conv1d(
                    open_mask.float().unsqueeze(0).unsqueeze(0),
                    kern,
                    padding=pad_frames,
                ).squeeze()
                open_mask = dil > 0.0
            hold_frames = max(0, int(round(cfg.hard_open_hold_ms / max(1e-3, hop_ms))))
            if hold_frames > 0 and open_mask.numel() > 1:
                # Keep a channel open briefly only around speech-active frames
                # to avoid choppy reactions without extending wrong-winner leakage.
                ch_speech = active[idx] if idx < active.size(0) else open_mask
                held = open_mask.clone()
                hold_left = 0
                for t in range(int(open_mask.numel())):
                    if bool(open_mask[t].item()) and bool(ch_speech[t].item()):
                        hold_left = hold_frames
                        held[t] = True
                    elif hold_left > 0 and bool(ch_speech[t].item()):
                        held[t] = True
                        hold_left -= 1
                    else:
                        hold_left = 0
                open_mask = held
            g = torch.where(open_mask, torch.ones_like(conf), torch.zeros_like(conf))
            # De-click hard transitions without re-opening bleed materially.
            g = _smooth_gain_envelope(
                g,
                attack_ms=cfg.hard_mute_attack_ms,
                release_ms=cfg.hard_mute_release_ms,
                sr=sr,
                hop_ms=hop_ms,
            )
            g = torch.clamp(g, 0.0, 1.0)
        else:
            g = 10.0 ** (gains_db[idx] / 20.0)
            g = _smooth_gain_envelope(g, attack_ms=cfg.attack_ms, release_ms=cfg.release_ms, sr=sr, hop_ms=hop_ms)
        gains_lin.append(g)

    hop = max(1, int(sr * hop_ms / 1000.0))
    gated: list[torch.Tensor] = []
    for mono, g in zip(padded, gains_lin):
        if g.numel() == 0:
            gated.append(mono)
            continue
        g_up = torch.repeat_interleave(g, hop)
        last = float(g[-1])
        if g_up.size(-1) < T:
            g_up = torch.nn.functional.pad(g_up, (0, T - g_up.size(-1)), value=last)
        g_up = g_up[:T]
        filt_len = max(1, int(sr * (cfg.upsample_smooth_ms / 1000.0)))
        if filt_len > 1:
            kernel = torch.ones(filt_len, dtype=g_up.dtype, device=g_up.device) / float(filt_len)
            g_up = torch.nn.functional.conv1d(
                g_up.unsqueeze(0).unsqueeze(0),
                kernel.view(1, 1, -1),
                padding=filt_len // 2,
            ).squeeze()
            g_up = g_up[:T]
        gated.append(mono * g_up)

    out: list[torch.Tensor] = []
    for orig, ga in zip(monos, gated):
        out.append(ga[..., :orig.size(-1)])
    try:
        dur_s = float(T / max(1, sr))
        dur_min = max(1e-6, dur_s / 60.0)
        confident_ratio = float((conf > 0.66).float().mean().item()) if conf.numel() else 0.0
        uncertain_ratio = float(uncertain_overlap.float().mean().item()) if uncertain_overlap.numel() else 0.0
        neg = gains_db[gains_db < 0]
        mean_att = float((-neg).mean().item()) if neg.numel() else 0.0
        switch_per_min = float(switches / dur_min)
        print(
            f"[bleed] confident={confident_ratio*100.0:.1f}% "
            f"uncertain_overlap={uncertain_ratio*100.0:.1f}% "
            f"switches_per_min={switch_per_min:.2f} "
            f"mean_loser_att_db={mean_att:.2f} "
            f"hard_iso={1 if cfg.hard_isolation else 0} "
            f"hard_mute={1 if (cfg.hard_isolation and cfg.hard_mute) else 0}"
        )
        diag_mode = str(os.environ.get("RESEMBLE_DIAG_MINIMAL", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if diag_mode:
            active_ratio = active.float().mean(dim=1)
            margin_like = ((conf * max(1e-6, (cfg.full_db - cfg.start_db))) + cfg.start_db)
            med_margin = float(torch.median(margin_like).item()) if margin_like.numel() else 0.0
            details = ", ".join([f"ch{i+1}_active={float(active_ratio[i].item())*100.0:.1f}%" for i in range(n_ch)])
            print(f"[bleed] median_margin_db={med_margin:.2f} {details}")
    except Exception:
        pass
    return out


def _parse_bypass_env() -> list[tuple[float, float]]:
    """Parse RESEMBLE_BYPASS env as comma-separated start:dur seconds, e.g. "40.0:0.3,12.5:0.2""" 
    raw = os.environ.get("RESEMBLE_BYPASS", "").strip()
    if not raw:
        return []
    out: list[tuple[float, float]] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
        else:
            a, b = part, '0.2'
        try:
            start = float(a)
            dur = float(b)
            if start >= 0 and dur > 0:
                out.append((start, dur))
        except Exception:
            continue
    return out


def _bypass_time_windows(proc, orig, sr: int, windows: list[tuple[float, float]]):
    """Crossfade to original for specified time windows (seconds).
    proc, orig: 1D tensors at same sr; windows: list of (start_s, dur_s)
    """
    try:
        import torch
        if not windows:
            return proc
        x = proc.clone()
        y = orig
        n = min(x.numel(), y.numel())
        x = x[:n]
        y = y[:n]
        for (start_s, dur_s) in windows:
            a = int(max(0, start_s * sr))
            b = int(min(n, (start_s + dur_s) * sr))
            if b <= a + 8:
                continue
            # 10 ms easing on both sides
            ease = max(16, int(sr * 0.01))
            a0 = max(0, a - ease)
            b0 = min(n, b + ease)
            mlen = b0 - a0
            w = torch.linspace(0, 1, steps=mlen, dtype=x.dtype, device=x.device)
            # 0..1 fade-in to original across [a0,b0]
            mask = w.clone()
            x[a0:b0] = x[a0:b0] * (1 - mask) + y[a0:b0] * mask
        return x
    except Exception:
        return proc


def _get_console_python() -> str:
    exe = sys.executable or "python"
    lower = exe.lower()
    if lower.endswith("pythonw.exe"):
        cand = Path(exe).with_name("python.exe")
        if cand.exists():
            return str(cand)
        return "python"
    return exe


def _get_enhancer_run_dir() -> Path | None:
    """Optional override for GUI inference weights via env.

    Set RESEMBLE_ENHANCER_RUN_DIR to a trained run folder.
    """
    raw = str(os.environ.get("RESEMBLE_ENHANCER_RUN_DIR", "")).strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.exists() and p.is_dir():
        return p
    return None


class _Cancelled(Exception):
    pass


class _Control:
    def __init__(self) -> None:
        import threading as _th
        self.pause = _th.Event()        # when set, pause at next chunk boundary
        self.stop_after_chunk = _th.Event()  # when set, cancel gracefully after current chunk
        self.cancel_now = _th.Event()   # immediate cancel request (treated same at chunk boundary)


def _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb, seam_safe: bool = True, control: _Control | None = None, denoise_only: bool = True, noise_only: bool = False, output_dir: str | Path | None = None):
    from resemble_enhance.enhancer.inference import denoise, enhance
    import torchaudio
    from torchaudio.functional import resample as ta_resample
    run_dir = _get_enhancer_run_dir()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expected = len(files)
    noise_flag = noise_only
    try:
        noise_flag = noise_flag or os.environ.get("RESEMBLE_NOISE_ONLY", "0") == "1"
    except Exception:
        pass
    if progress_cb:
        progress_cb(0, expected)

    def on_chunk(evt, name, i, n):
        # Pause support: wait here at chunk boundary
        if control is not None:
            while control.pause.is_set():
                time.sleep(0.05)
            if control.cancel_now.is_set() or control.stop_after_chunk.is_set():
                raise _Cancelled()
        if chunk_progress_cb:
            chunk_progress_cb(name or "", i, n)

    done = 0
    out_dirs = set()
    output_base = Path(output_dir) if output_dir else None
    results: list[tuple[str, str]] = []
    for f in files:
        p = Path(f)
        dest_dir = _build_output_dest_dir(p, output_base, stamp)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.add(dest_dir)
        name = p.name
        out_name = _build_output_name(p, stamp)
        try:
            os.environ["RESEMBLE_FILE"] = str(p)
        except Exception:
            pass
        if control is not None and (control.cancel_now.is_set() or control.stop_after_chunk.is_set()):
            break
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0)
        wav, orig_len = _pad_audio_tensor_min_samples(wav, MIN_AUDIO_SAMPLES)
        if seam_safe:
            kwargs = dict(chunk_seconds=60.0, overlap_seconds=4.0, align_max_shift_ratio=0.05, align_disable=False)
        else:
            kwargs = dict(chunk_seconds=31.0, overlap_seconds=1.0, align_max_shift_ratio=0.25, align_disable=True)
        # Allow env overrides for chunk sizing (set by GUI controls)
        try:
            cs = float(os.environ.get('RESEMBLE_CHUNK_SECONDS', '') or 0) or None
            ov = float(os.environ.get('RESEMBLE_OVERLAP_SECONDS', '') or 0) or None
            if cs is not None and cs > 0:
                kwargs['chunk_seconds'] = float(cs)
            if ov is not None and ov >= 0:
                # clamp overlap to quarter of chunk for safety
                max_ov = max(0.0, float(kwargs.get('chunk_seconds', 31.0)) / 4.0)
                kwargs['overlap_seconds'] = float(min(ov, max_ov))
        except Exception:
            pass
        if denoise_only:
            # Denoise-only safety profile:
            # 1) disable cross-chunk alignment shifts (can create doubling/echo feel)
            # 2) use larger chunks + overlap for smoother seams
            cs_cur = float(kwargs.get("chunk_seconds", 31.0))
            ov_cur = float(kwargs.get("overlap_seconds", 1.0))
            cs_safe = max(45.0, cs_cur)
            ov_safe = max(2.5, ov_cur)
            ov_safe = min(ov_safe, cs_safe / 4.0)
            kwargs.update(
                chunk_seconds=cs_safe,
                overlap_seconds=ov_safe,
                align_disable=True,
                align_max_shift_ratio=0.0,
            )
        # Force single-chunk mode via env for diagnostics
        try:
            if os.environ.get("RESEMBLE_FORCE_SINGLE_CHUNK", "0") == "1":
                dur = float(wav.shape[-1]) / float(sr)
                kwargs.update(chunk_seconds=max(1.0, dur + 1.0), overlap_seconds=0.0, align_disable=True)
        except Exception:
            pass
        try:
            # OOM-resilient loop: progressively shrink chunk size; fallback to CPU if needed
            max_retries = 4
            attempt = 0
            cur_kwargs = dict(**kwargs)
            cur_device = device
            while True:
                try:
                    if denoise_only:
                        hwav, model_sr = denoise(
                            dwav=wav,
                            sr=sr,
                            device=cur_device,
                            run_dir=run_dir,
                            progress_cb=lambda evt, nm, i, n, p=p: on_chunk(evt, str(p), i, n),
                            **cur_kwargs,
                        )
                    else:
                        extra = {}
                        try:
                            fast = os.environ.get('RESEMBLE_FAST_ENHANCE','0') == '1'
                        except Exception:
                            fast = False
                        if fast or str(cur_device).lower() == "cpu":
                            # Prefer a faster configuration to avoid stalls in diagnostics or on CPU
                            if str(cur_device).lower() == "cuda":
                                extra.update(nfe=AI_SYNTHESIS_DEFAULT_NFE, solver="midpoint")
                            else:
                                extra.update(nfe=8, solver="euler")
                        hwav, model_sr = enhance(
                            dwav=wav,
                            sr=sr,
                            device=cur_device,
                            run_dir=run_dir,
                            lambd=AI_SYNTHESIS_DEFAULT_LAMBD,
                            tau=AI_SYNTHESIS_DEFAULT_TAU,
                            progress_cb=lambda evt, nm, i, n, p=p: on_chunk(evt, str(p), i, n),
                            **cur_kwargs,
                            **extra,
                        )
                    break
                except _Cancelled:
                    raise
                except Exception as e:
                    # Detect CUDA OOM
                    if "CUDA out of memory" in str(e) or getattr(type(e), "__name__", "").lower().startswith("outofmemory"):
                        attempt += 1
                        # Shrink chunk size, keep overlap reasonable
                        cs = float(cur_kwargs.get("chunk_seconds", 31.0))
                        ov = float(cur_kwargs.get("overlap_seconds", 1.0))
                        new_cs = max(7.0, cs / 2.0)
                        new_ov = min(ov, new_cs / 4.0)
                        cur_kwargs.update(chunk_seconds=new_cs, overlap_seconds=new_ov)
                        try:
                            import torch as _t
                            if _t.cuda.is_available():
                                _t.cuda.empty_cache()
                        except Exception:
                            pass
                        if attempt >= max_retries and cur_device == "cuda":
                            # Final fallback to CPU
                            cur_device = "cpu"
                            attempt = 0
                        if attempt > max_retries and cur_device == "cpu":
                            raise
                        # optional user feedback
                        if progress_cb:
                            progress_cb(done, expected)
                        continue
                    else:
                        raise
        except _Cancelled:
            break
        # Record model sample rate for diagnostics
        try:
            global LAST_MODEL_SR, LAST_MODEL_SR_PATH
            LAST_MODEL_SR = int(model_sr) if model_sr is not None else None
            LAST_MODEL_SR_PATH = str(p)
        except Exception:
            pass
        dest_sr = 48000 if profile else sr
        if model_sr != dest_sr:
            hwav = ta_resample(hwav, orig_freq=model_sr, new_freq=dest_sr)
        if not orig_len:
            orig_len = int(wav.shape[-1])
        exp_len = round(orig_len * (dest_sr / sr)) if dest_sr != sr else orig_len
        if hwav.shape[-1] > exp_len:
            hwav = hwav[:exp_len]
        elif hwav.shape[-1] < exp_len:
            import torch
            hwav = torch.nn.functional.pad(hwav, (0, exp_len - hwav.shape[-1]))
        try:
            base = wav
            if dest_sr != sr:
                base = ta_resample(base, orig_freq=sr, new_freq=dest_sr)
        except Exception:
            base = wav
        try:
            if base.shape[-1] > exp_len:
                base = base[:exp_len]
            elif base.shape[-1] < exp_len:
                import torch as _t
                base = _t.nn.functional.pad(base, (0, exp_len - base.shape[-1]))
        except Exception:
            pass
        # Transient-safe blend: add a little original back where mismatch is sharp and loud
        try:
            # Optional debug dump before post-processing (RAW model output at model_sr)
            if os.environ.get("RESEMBLE_DEBUG_DUMP", "0") == "1":
                try:
                    raw_dbg = dest_dir / (Path(name).stem + "_RAW.wav")
                    torchaudio.save(str(raw_dbg), hwav[None], model_sr)
                except Exception:
                    pass
            if os.environ.get('RESEMBLE_DISABLE_TRANSIENT_BLEND', '0') != '1':
                hwav = _adaptive_transient_blend(hwav, base, dest_sr, strength=0.9)
            # Optional explicit bypass windows via env var (e.g., RESEMBLE_BYPASS="40.0:0.3")
            wins = _parse_bypass_env()
            if wins:
                hwav = _bypass_time_windows(hwav, base, dest_sr, wins)
            # Leading transient guard: protect the first pronounced onset by easing to original
            if os.environ.get('RESEMBLE_LEAD_GUARD', '1') == '1':
                try:
                    import torch as _t
                    x = hwav
                    y = base
                    n = min(x.numel(), y.numel())
                    if n > dest_sr // 2:
                        x = x[:n]
                        y = y[:n]
                        k_env = max(8, int(dest_sr * 0.005))  # ~5 ms smoothing
                        pad = k_env // 2
                        w = _t.ones(1, 1, k_env, dtype=x.dtype, device=x.device) / float(k_env)
                        # mismatch and level envelopes
                        d = (x - y).abs().unsqueeze(0).unsqueeze(0)
                        d_s = _t.nn.functional.conv1d(d, w, padding=pad).squeeze()
                        level_env = _t.maximum(x.abs(), y.abs()).unsqueeze(0).unsqueeze(0)
                        l_s = _t.nn.functional.conv1d(level_env, w, padding=pad).squeeze()
                        # thresholds
                        d_med = _t.quantile(d_s, 0.5)
                        d_thr = d_med * 4.0
                        lvl_thr = 10 ** (-18.0 / 20.0)
                        cand = ((d_s > d_thr) & (l_s > lvl_thr)).nonzero(as_tuple=False)
                        if cand.numel() > 0:
                            a = int(cand[0].item())
                            # build ~150 ms guard around first onset
                            dur = int(dest_sr * 0.15)
                            ease = max(16, int(dest_sr * 0.01))
                            a0 = max(0, a - ease)
                            b0 = min(n, a + dur + ease)
                            mlen = b0 - a0
                            wlin = _t.linspace(0, 1, steps=mlen, dtype=x.dtype, device=x.device)
                            xm = x.clone()
                            xm[a0:b0] = x[a0:b0] * (1 - wlin) + y[a0:b0] * wlin
                            hwav = xm
                except Exception:
                    pass
            # Wet/dry mix: allow dialing down denoise strength
            try:
                wet = float(os.environ.get('RESEMBLE_WET', '1.0'))
            except Exception:
                wet = 1.0
            wet = max(0.0, min(1.0, wet))
            if wet < 1.0:
                hwav = wet * hwav + (1.0 - wet) * base
            if noise_flag:
                # Export the residual (background/noise) instead of the cleaned signal
                hwav = base - hwav
        except Exception:
            pass
        out_path = dest_dir / out_name
        hwav = _apply_peak_ceiling(hwav, ceiling_db=-1.0)
        # Optional debug dump after processing (POST)
        if os.environ.get("RESEMBLE_DEBUG_DUMP", "0") == "1":
            try:
                post_dbg = dest_dir / (Path(name).stem + "_POST.wav")
                torchaudio.save(str(post_dbg), hwav[None], dest_sr)
            except Exception:
                pass
        torchaudio.save(str(out_path), hwav[None], dest_sr)
        done += 1
        if progress_cb:
            progress_cb(done, expected)
        results.append((str(p), str(out_path)))
        try:
            import torch as _t
            if str(device).lower() == "cuda" and _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    return results


def run_enhancer_for(files, device="cuda", profile=True, progress_cb=None, chunk_progress_cb=None, seam_safe: bool = True, control: _Control | None = None, denoise_only: bool = True, prefer_cli: bool = False, noise_only: bool = False, output_dir: str | Path | None = None, force_inprocess: bool = False):
    # When frozen into an EXE, run in-process for full portability
    if (getattr(sys, 'frozen', False) or control is not None or noise_only or force_inprocess) and not prefer_cli:
        return _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb, seam_safe=seam_safe, control=control, denoise_only=denoise_only, noise_only=noise_only, output_dir=output_dir)

    run_id = uuid.uuid4().hex[:8]
    in_dir = INPUT_TMP_ROOT / run_id / "input_audio"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = OUTPUT_ROOT / stamp
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_base = Path(output_dir) if output_dir else None
    noise_flag = noise_only
    try:
        noise_flag = noise_flag or os.environ.get("RESEMBLE_NOISE_ONLY", "0") == "1"
    except Exception:
        pass

    # Copy or transcode files into temp input dir; prefer WAV for CLI reliability
    orig_meta: dict[str, tuple[int, int]] = {}
    for f in files:
        srcp = Path(f)
        if srcp.suffix.lower() != ".wav":
            # Transcode to WAV (mono, preserve rate) for CLI default suffix handling
            try:
                ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
                dst = in_dir / (srcp.stem + ".wav")
                if ff:
                    cmd_tx = [ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(srcp), '-ac', '1', str(dst)]
                    subprocess.run(cmd_tx, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                else:
                    import torchaudio as _ta
                    wav, sr = _ta.load(str(srcp))
                    if wav.dim() == 2 and wav.size(0) > 1:
                        wav = wav.mean(0)
                    else:
                        wav = wav.squeeze(0)
                    _ta.save(str(dst), wav.unsqueeze(0), sr)
                meta = _pad_wav_on_disk(dst, MIN_AUDIO_SAMPLES)
                if meta and meta[0] < MIN_AUDIO_SAMPLES:
                    orig_meta[str(f)] = meta
            except Exception:
                # Fallback: copy source if transcode fails
                dst = in_dir / srcp.name
                shutil.copy2(f, dst)
        else:
            dst = in_dir / srcp.name
            shutil.copy2(f, dst)
            meta = _pad_wav_on_disk(dst, MIN_AUDIO_SAMPLES)
            if meta and meta[0] < MIN_AUDIO_SAMPLES:
                orig_meta[str(f)] = meta

    py = _get_console_python()
    cmd = [
        py,
        "-m",
        "resemble_enhance.enhancer",
        str(in_dir),
        str(out_dir),
        "--device",
        device,
    ]
    run_dir = _get_enhancer_run_dir()
    if run_dir is not None:
        cmd += ["--run_dir", str(run_dir)]
    if denoise_only:
        cmd.insert(len(cmd)-2, "--denoise_only")
    # In diagnostics/CLI mode, avoid profile overrides so our small chunk/overlap apply immediately
    if profile and not prefer_cli:
        cmd += ["--profile", "camera_sync"]
    # If enhance on CPU, force fast settings to avoid long stalls
    try:
        if (not denoise_only) and (str(device).lower() == 'cpu'):
            cmd += ["--nfe", "8", "--solver", "euler"]
        elif not denoise_only:
            cmd += [
                "--nfe", str(AI_SYNTHESIS_DEFAULT_NFE),
                "--solver", "midpoint",
                "--lambd", str(AI_SYNTHESIS_DEFAULT_LAMBD),
                "--tau", str(AI_SYNTHESIS_DEFAULT_TAU),
            ]
    except Exception:
        pass
    # Seam handling
    if seam_safe:
        # Allow env overrides from GUI controls
        cs = os.environ.get('RESEMBLE_CHUNK_SECONDS', '60.0') or '60.0'
        ov = os.environ.get('RESEMBLE_OVERLAP_SECONDS', '4.0') or '4.0'
        cmd += [
            "--chunk_seconds", str(cs),
            "--overlap_seconds", str(ov),
            "--align_max_shift_ratio", "0.05",
        ]
    else:
        cs = os.environ.get('RESEMBLE_CHUNK_SECONDS', '31.0') or '31.0'
        ov = os.environ.get('RESEMBLE_OVERLAP_SECONDS', '1.0') or '1.0'
        cmd += ["--align_disable", "--chunk_seconds", str(cs), "--overlap_seconds", str(ov)]
    if denoise_only:
        # Denoise-only safety profile for CLI path (same intent as in-process path).
        try:
            cs_eff = max(20.0, float(cs))
        except Exception:
            cs_eff = 45.0
        try:
            ov_eff = max(2.5, float(ov))
        except Exception:
            ov_eff = 2.5
        ov_eff = min(ov_eff, cs_eff / 4.0)
        cmd += ["--align_disable", "--chunk_seconds", str(cs_eff), "--overlap_seconds", str(ov_eff), "--align_max_shift_ratio", "0.0"]

    # Launch process
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = os.environ.copy()
    env["RESEMBLE_PROGRESS"] = "1"
    log_path = out_dir / "enhancer_cli.log"
    log_lines = deque(maxlen=200)
    log_lock = threading.Lock()

    def _record_log(line: str) -> None:
        if not line:
            return
        try:
            with log_lock:
                log_lines.append(line)
        except Exception:
            pass
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(line + "\n")
        except Exception:
            pass

    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as f:
            f.write("cmd: " + " ".join(cmd) + "\n")
    except Exception:
        pass
    proc = subprocess.Popen(
        cmd,
        creationflags=creationflags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    _register_live_subprocess(proc)

    # Read and parse progress output
    def reader():
        import re
        nonlocal expected
        current_file = None
        cur_n = 0
        start_re = re.compile(r"PROGRESS START file=(.*) n=(\d+)")
        chunk_re = re.compile(r"PROGRESS CHUNK file=(.*) i=(\d+) n=(\d+)")
        end_re = re.compile(r"PROGRESS END file=(.*)")
        stage_re = re.compile(r"PROGRESS STAGE file=(.*?) stage=([^ ]+) detail=(.*)")
        for line in proc.stdout:  # type: ignore[attr-defined]
            line = line.rstrip()
            _record_log(line)
            m = start_re.search(line)
            if m:
                current_file = m.group(1)
                try:
                    cur_n = int(m.group(2))
                except Exception:
                    cur_n = 0
                try:
                    print(f"PROGRESS START file={current_file} n={cur_n}", flush=True)
                except Exception:
                    pass
                if chunk_progress_cb:
                    chunk_progress_cb(current_file or "", 0, cur_n)
                continue
            m = stage_re.search(line)
            if m:
                name = m.group(1) or current_file or ""
                detail = m.group(3).strip()
                if detail:
                    try:
                        print(f"PROGRESS STAGE file={name} detail={detail}", flush=True)
                    except Exception:
                        pass
                    if chunk_progress_cb:
                        chunk_progress_cb(f"{name}|{detail}", 0, cur_n)
                continue
            m = chunk_re.search(line)
            if m:
                name = m.group(1)
                try:
                    i = int(m.group(2))
                    n = int(m.group(3))
                except Exception:
                    i, n = 0, 0
                try:
                    print(f"PROGRESS CHUNK file={name} i={i} n={n}", flush=True)
                except Exception:
                    pass
                if chunk_progress_cb:
                    chunk_progress_cb(name or current_file or "", i, n)
                continue
            m = end_re.search(line)
            if m:
                name = m.group(1)
                try:
                    print(f"PROGRESS END file={name}", flush=True)
                except Exception:
                    pass
                if chunk_progress_cb:
                    chunk_progress_cb(name or current_file or "", cur_n, cur_n)
                continue
            # ignore other logs

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    expected = len(files)
    completed = 0
    try:
        # Poll for outputs while the process runs
        while proc.poll() is None:
            if control is not None and (control.cancel_now.is_set() or control.stop_after_chunk.is_set()):
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise _Cancelled()

            def _has_out(fpath: str) -> bool:
                p = Path(fpath)
                cand1 = out_dir / p.name
                cand2 = out_dir / (p.stem + ".wav")
                cand3 = out_dir / (p.stem + ".mov")
                return any(c.exists() and c.stat().st_size > 44 for c in (cand1, cand2, cand3))
            completed = sum(1 for f in files if _has_out(f))
            if progress_cb:
                progress_cb(completed, expected)
            time.sleep(0.5)

        # Final progress update
        completed = sum(1 for f in files if (out_dir / Path(f).name).exists())
        if progress_cb:
            progress_cb(completed, expected)

        try:
            t.join(timeout=1.0)
        except Exception:
            pass
    finally:
        _unregister_live_subprocess(proc)

    rc = proc.returncode
    if rc != 0:
        tail = []
        try:
            with log_lock:
                tail = list(log_lines)[-10:]
        except Exception:
            tail = []
        if tail:
            raise RuntimeError(
                f"Enhancer returned code {rc}. See log: {log_path}. Tail:\n" + "\n".join(tail)
            )
        raise RuntimeError(f"Enhancer returned code {rc}. See log: {log_path}")

    # Cleanup inputs if outputs look valid
    # Move outputs to selected output folder (if any), otherwise next to originals under Enhanced_<timestamp>
    results: list[tuple[str, str]] = []
    for f in files:
        name = Path(f).name
        out_name = _build_output_name(Path(f), stamp)
        tmp_out = out_dir / name
        if not (tmp_out.exists() and tmp_out.stat().st_size > 44):
            # Try common alternate container/extension
            alt = out_dir / (Path(f).stem + ".wav")
            if alt.exists() and alt.stat().st_size > 44:
                tmp_out = alt
            else:
                alt2 = out_dir / (Path(f).stem + ".mov")
                if alt2.exists() and alt2.stat().st_size > 44:
                    tmp_out = alt2
                else:
                    continue
        dest_dir = _build_output_dest_dir(Path(f), output_base, stamp)
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_out = dest_dir / out_name
        try:
            shutil.move(str(tmp_out), str(final_out))
        except Exception:
            # Best-effort fallback to copy
            shutil.copy2(str(tmp_out), str(final_out))
            tmp_out.unlink(missing_ok=True)
        if str(f) in orig_meta and final_out.suffix.lower() in {".wav", ".wave"}:
            try:
                import torchaudio as _ta
                orig_len, orig_sr = orig_meta[str(f)]
                wav, out_sr = _ta.load(str(final_out))
                if orig_sr and out_sr:
                    target_len = int(round(orig_len * (float(out_sr) / float(orig_sr))))
                else:
                    target_len = int(orig_len)
                if target_len > 0 and wav.size(-1) > target_len:
                    wav = wav[..., :target_len]
                    _ta.save(str(final_out), wav, int(out_sr))
            except Exception:
                pass
        results.append((str(Path(f)), str(final_out)))

    if noise_flag and results:
        try:
            import torchaudio as _ta
            from torchaudio.functional import resample as _ta_resample
            import torch as _t
            updated: list[tuple[str, str]] = []
            for src_path, out_path in results:
                try:
                    orig, orig_sr = _ta.load(str(src_path))
                    if orig.dim() == 2 and orig.size(0) > 1:
                        orig = orig.mean(0, keepdim=True)
                    elif orig.dim() == 1:
                        orig = orig.unsqueeze(0)
                    cleaned, out_sr = _ta.load(str(out_path))
                    if cleaned.dim() == 2 and cleaned.size(0) > 1:
                        cleaned = cleaned.mean(0, keepdim=True)
                    elif cleaned.dim() == 1:
                        cleaned = cleaned.unsqueeze(0)
                    if orig_sr != out_sr:
                        orig = _ta_resample(orig, orig_freq=int(orig_sr), new_freq=int(out_sr))
                    target_len = int(cleaned.size(-1))
                    if orig.size(-1) > target_len:
                        orig = orig[..., :target_len]
                    elif orig.size(-1) < target_len:
                        orig = _t.nn.functional.pad(orig, (0, target_len - int(orig.size(-1))))
                    noise = orig - cleaned
                    noise = _apply_peak_ceiling(noise, ceiling_db=-1.0)
                    _ta.save(str(out_path), noise, int(out_sr))
                    updated.append((src_path, out_path))
                except Exception:
                    updated.append((src_path, out_path))
            results = updated
        except Exception:
            pass

    # Cleanup temporary input dir for this run to avoid accumulating past runs
    try:
        run_root = in_dir.parent  # .../.enhancer_runs_gui/<run_id>
        import shutil as _sh
        _sh.rmtree(run_root, ignore_errors=True)
    except Exception:
        pass

    # Do not modify or delete original files or folders
    return results


_WNDPROC_MAP = {}


def _install_win_dnd(widget, on_files):
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
    except Exception:
        return False

    WM_DROPFILES = 0x0233
    GWL_WNDPROC = -4

    hwnd = widget.winfo_id()
    shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]
    shell32.DragAcceptFiles.restype = None
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, ctypes.c_uint, wintypes.LPWSTR, ctypes.c_uint]
    shell32.DragQueryFileW.restype = ctypes.c_uint
    shell32.DragFinish.argtypes = [wintypes.HANDLE]
    shell32.DragFinish.restype = None

    shell32.DragAcceptFiles(wintypes.HWND(hwnd), True)

    # Configure CallWindowProcW signature
    user32.CallWindowProcW.argtypes = [ctypes.c_void_p, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM]
    user32.CallWindowProcW.restype = ctypes.c_long

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND, ctypes.c_uint, wintypes.WPARAM, wintypes.LPARAM)

    def py_wndproc(h, msg, wparam, lparam):
        if msg == WM_DROPFILES:
            hdrop = ctypes.c_void_p(int(wparam))
            count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
            files = []
            for i in range(count):
                needed = shell32.DragQueryFileW(hdrop, i, None, 0)
                buf = ctypes.create_unicode_buffer(needed + 1)
                shell32.DragQueryFileW(hdrop, i, buf, needed + 1)
                files.append(buf.value)
            shell32.DragFinish(hdrop)
            try:
                on_files(files)
            except Exception:
                pass
        # call original wndproc
        prev_ptr = _WNDPROC_MAP.get(hwnd, (None, None))[0]
        return user32.CallWindowProcW(prev_ptr, h, msg, wparam, lparam)

    newproc = WNDPROC(py_wndproc)

    # Choose correct setter based on arch
    is_64 = ctypes.sizeof(ctypes.c_void_p) == 8
    if is_64:
        set_wndproc = user32.SetWindowLongPtrW
    else:
        set_wndproc = user32.SetWindowLongW
    # Ensure function signatures accept pointer-sized values
    set_wndproc.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_void_p]
    set_wndproc.restype = ctypes.c_void_p

    newptr = ctypes.cast(newproc, ctypes.c_void_p)
    prev = set_wndproc(wintypes.HWND(hwnd), GWL_WNDPROC, newptr)
    if not prev:
        return False
    _WNDPROC_MAP[hwnd] = (prev, newproc)  # keep prev pointer and callback alive
    return True


class App((TkinterDnD.Tk if DND_AVAILABLE else tk.Tk)):
    def __init__(self):
        super().__init__()
        self._packaged_ui_mode = bool(getattr(sys, "frozen", False))
        hide_otio_env = os.environ.get("RESEMBLE_HIDE_OTIO")
        self._hide_otio_ui = (hide_otio_env.strip() == "1") if hide_otio_env is not None else self._packaged_ui_mode
        self._force_sync_export = self._packaged_ui_mode
        self._hide_reduce_gpu_ui = self._packaged_ui_mode
        self._reuse_existing_outputs = not self._packaged_ui_mode
        self.title("Resemble Enhance")
        self.geometry("980x620")
        self.minsize(920, 660)

        self.files = []
        self.history: list[tuple[str, str]] = []
        self.cur_start_time: float | None = None
        self.folders: set[str] = set()
        # Per-file status: queued|running|done|failed
        self.file_status: dict[str, str] = {}
        # Client-aware queue metadata for project/client ingestion mode
        self.client_mode_active: bool = False
        self.client_queue: dict[str, list[str]] = {}
        self.client_meta: dict[str, dict] = {}
        self.client_order: list[str] = []
        self.file_to_client: dict[str, str] = {}
        self._iid_to_path: dict[str, str] = {}
        self._path_to_iid: dict[str, str] = {}
        self._iid_to_folder: dict[str, str] = {}
        self._folder_to_iid: dict[str, str] = {}

        # Basic styling (prefer ttkbootstrap theme if available)
        if _TTKBOOT_AVAILABLE:
            try:
                style = _ttkb.Style('darkly')  # modern dark theme
            except Exception:
                style = ttk.Style()
            # Map palette from ttkbootstrap colors when available
            cols = getattr(style, 'colors', None)
            self._bg = '#0b1016'
            self._panel = '#141b24'
            self._panel_alt = '#1b2430'
            self._button_bg = '#202938'
            self._button_hover = '#2a3647'
            self._border = '#2b3746'
            self._text = getattr(cols, 'fg', '#e5e7eb') if cols is not None else '#e5e7eb'
            self._muted = '#94a3b8'
            self._accent = '#4f8ff7'
            self._accent_hover = '#6aa3ff'
            self._success = getattr(cols, 'success', '#22c55e') if cols is not None else '#22c55e'
        else:
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass
            # Fallback dark palette
            self._bg = '#0b1016'
            self._panel = '#141b24'
            self._panel_alt = '#1b2430'
            self._button_bg = '#202938'
            self._button_hover = '#2a3647'
            self._border = '#2b3746'
            self._text = '#e5e7eb'
            self._muted = '#94a3b8'
            self._accent = '#4f8ff7'
            self._accent_hover = '#6aa3ff'
            self._success = '#22c55e'
        self.configure(bg=self._bg)
        style.configure('.', background=self._bg, foreground=self._text)
        style.configure('TFrame', background=self._bg)
        style.configure('Card.TFrame', background=self._panel, bordercolor=self._border, relief='solid')
        style.configure('Section.TFrame', background=self._panel_alt, bordercolor=self._border, relief='solid')
        style.configure('Title.TLabel', background=self._bg, foreground=self._text, font=('Segoe UI', 13, 'bold'))
        style.configure('CardTitle.TLabel', background=self._panel, foreground=self._text, font=('Segoe UI', 13, 'bold'))
        style.configure('SectionTitle.TLabel', background=self._panel_alt, foreground=self._text, font=('Segoe UI', 11, 'bold'))
        style.configure('Info.TLabel', background=self._bg, foreground=self._muted)
        style.configure('CardInfo.TLabel', background=self._panel, foreground=self._muted)
        style.configure('SectionInfo.TLabel', background=self._panel_alt, foreground=self._muted)
        style.configure('TButton', padding=9, background=self._button_bg, foreground=self._text, bordercolor=self._border)
        style.map('TButton', background=[('active', self._button_hover)])
        style.configure('Drop.TFrame', background=self._panel_alt, bordercolor=self._border, relief='solid')
        style.configure('Horizontal.TProgressbar', troughcolor='#101722', background=self._accent, bordercolor=self._panel_alt, lightcolor=self._accent, darkcolor=self._accent)
        # Tree grid/separators slightly lighter so columns are clearly divided
        style.configure('Treeview', background=self._panel_alt, fieldbackground=self._panel_alt, foreground=self._text, bordercolor=self._border)
        style.map('Treeview', background=[('selected', '#273244')], foreground=[('selected', self._text)])
        # Treeview heading styling (neutral grey shade)
        style.configure('Treeview.Heading', background='#202a36', foreground=self._text, bordercolor=self._border)
        style.map('Treeview.Heading', background=[('active', '#2a3647')])
        # Accent button styles
        style.configure('Accent.TButton', background=self._accent, foreground="#ffffff")
        style.configure('AccentHover.TButton', background=self._accent_hover, foreground="#ffffff")
        # Standard hover style for normal buttons
        style.configure('Hover.TButton', background=self._button_hover, foreground=self._text)
        # Option checkbuttons
        style.configure('Opt.TCheckbutton', background=self._panel_alt, foreground=self._text, padding=4)

        # Helper to apply a simple hover effect to any ttk.Button
        def _bind_hover(b: ttk.Button, base_style: str = 'TButton', hover_style: str = 'Hover.TButton'):
            try:
                b.configure(style=base_style)
                b.bind('<Enter>', lambda e: b.configure(style=hover_style))
                b.bind('<Leave>', lambda e: b.configure(style=base_style))
            except Exception:
                pass
        self._bind_hover = _bind_hover
        self._ui_icons: dict[str, tk.PhotoImage] = {}

        def _load_ui_icon(name: str) -> tk.PhotoImage | None:
            cached = self._ui_icons.get(name)
            if cached is not None:
                return cached
            try:
                path = ICON_ROOT / f"{name}.png"
                if not path.exists():
                    return None
                img = tk.PhotoImage(file=str(path))
                self._ui_icons[name] = img
                return img
            except Exception:
                return None

        def _set_button_icon(button: ttk.Button, icon_name: str) -> None:
            img = _load_ui_icon(icon_name)
            if img is None:
                return
            try:
                button.configure(image=img, compound='left')
                button.image = img
            except Exception:
                pass
        self._set_button_icon = _set_button_icon

        # Main paned layout
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True, padx=12, pady=12)

        left = ttk.Frame(pw, width=360, style='Card.TFrame', padding=(16, 16))
        right = ttk.Frame(pw, style='Card.TFrame', padding=(16, 16))
        # Fix left column width for consistent visibility
        pw.add(left, weight=0)
        pw.add(right, weight=1)
        # Ensure sash is positioned after layout and stays fixed
        self._left_fixed_width = 360
        def _set_sash():
            try:
                pw.sashpos(0, self._left_fixed_width)
            except Exception:
                pass
        self.after(100, _set_sash)
        pw.bind('<Configure>', lambda e: _set_sash())
        self.bind('<Configure>', lambda e: _set_sash())
        # Keep cursor as arrow; only block drags when the sash/handle is targeted
        pw.configure(cursor='arrow')
        def _maybe_block_pane_drag(event):
            try:
                part = pw.identify(event.x, event.y)
                if isinstance(part, str) and 'sash' in part:
                    _set_sash()
                    return 'break'
            except Exception:
                # best effort
                return None
            return None
        pw.bind('<Button-1>', _maybe_block_pane_drag)
        pw.bind('<B1-Motion>', _maybe_block_pane_drag)

        # Left column is scrollable so expanded option groups never get clipped.
        left_canvas = tk.Canvas(left, bg=self._panel, highlightthickness=0, bd=0)
        left_scrollbar = ttk.Scrollbar(left, orient='vertical', command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side='left', fill='both', expand=True)
        left_scrollbar.pack(side='right', fill='y')
        left_body = ttk.Frame(left_canvas, style='Card.TFrame')
        self._left_canvas = left_canvas
        self._left_scrollbar = left_scrollbar
        self._left_body = left_body
        self._left_body_window = left_canvas.create_window((0, 0), window=left_body, anchor='nw')

        def _sync_left_scrollregion(event=None):
            try:
                left_canvas.configure(scrollregion=left_canvas.bbox('all'))
            except Exception:
                pass

        def _sync_left_body_width(event):
            try:
                left_canvas.itemconfigure(self._left_body_window, width=max(1, event.width))
            except Exception:
                pass

        def _on_left_mousewheel(event):
            try:
                delta = getattr(event, 'delta', 0)
                if delta:
                    left_canvas.yview_scroll(int(-1 * (delta / 120)), 'units')
                elif getattr(event, 'num', None) == 4:
                    left_canvas.yview_scroll(-3, 'units')
                elif getattr(event, 'num', None) == 5:
                    left_canvas.yview_scroll(3, 'units')
            except Exception:
                pass
            return 'break'

        left_body.bind('<Configure>', _sync_left_scrollregion)
        left_canvas.bind('<Configure>', _sync_left_body_width)
        left_canvas.bind_all('<MouseWheel>', _on_left_mousewheel, add='+')
        left_canvas.bind_all('<Button-4>', _on_left_mousewheel, add='+')
        left_canvas.bind_all('<Button-5>', _on_left_mousewheel, add='+')

        # Left column: title, buttons, options, queue list
        title = "Drop or select audio files to enhance" if DND_AVAILABLE else "Select audio files to enhance"
        ttk.Label(left_body, text=title, style='CardTitle.TLabel').pack(anchor='w', pady=(0, 10))

        # (Removed dedicated drop zone; drag-and-drop works on the list below.)

        btns = ttk.Frame(left_body, style='Section.TFrame', padding=10)
        btns.pack(fill='x', pady=(0, 10))
        btns_hdr = ttk.Frame(btns, style='Section.TFrame')
        btns_hdr.pack(fill='x', pady=(0, 8))
        ttk.Label(btns_hdr, text='Sources', style='SectionTitle.TLabel').pack(side='left')
        btn_clear = ttk.Button(btns_hdr, text='', width=3, command=self.clear_files)
        btn_clear.pack(side='right')
        self._bind_hover(btn_clear)
        btn_grid = ttk.Frame(btns, style='Section.TFrame')
        btn_grid.pack(fill='x')
        btn_grid.grid_columnconfigure(0, weight=1)
        btn_grid.grid_columnconfigure(1, weight=1)
        btn_add = ttk.Button(btn_grid, text='Add Files', command=self.add_files)
        btn_add.grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=(0, 6))
        self._bind_hover(btn_add)
        btn_add_folder = ttk.Button(btn_grid, text='Add Folder', command=self.add_folder)
        btn_add_folder.grid(row=0, column=1, sticky='ew', pady=(0, 6))
        self._bind_hover(btn_add_folder)
        btn_project = ttk.Button(btn_grid, text='Select Project', command=self._select_project_folder)
        btn_project.grid(row=1, column=0, sticky='ew', padx=(0, 6))
        self._bind_hover(btn_project)
        btn_client = ttk.Button(btn_grid, text='Select Client', command=self._select_single_client_folder)
        btn_client.grid(row=1, column=1, sticky='ew')
        self._bind_hover(btn_client)
        self._set_button_icon(btn_add_folder, 'folder-open')
        self._set_button_icon(btn_project, 'folder-open')
        self._set_button_icon(btn_client, 'folder-open')
        self._set_button_icon(btn_clear, 'x')

        # Diagnostics configuration (advanced controls hidden, but vars remain for logic)
        self.var_profile = tk.BooleanVar(value=True)
        self.var_sync_export = tk.BooleanVar(value=True)
        self.var_postproc = tk.BooleanVar(value=True)
        self.var_skip_fine = tk.BooleanVar(value=False)
        self.var_bw64 = tk.BooleanVar(value=True)
        self.var_seam_safe = tk.BooleanVar(value=True)
        self.var_batch_folders = tk.BooleanVar(value=True)
        self.var_aggressive_denoise = tk.BooleanVar(value=True)
        self.var_chunk_sec = tk.DoubleVar(value=7.0)
        self.var_overlap_sec = tk.DoubleVar(value=0.5)
        self.var_wet = tk.DoubleVar(value=1.0)
        self.var_lead_guard = tk.BooleanVar(value=False)
        self.var_noise_only = tk.BooleanVar(value=False)
        self.var_ai_synthesis = tk.BooleanVar(value=False)
        self.var_denoise_only = tk.BooleanVar(value=True)
        self.var_diag_minimal = tk.BooleanVar(value=True)
        self.var_device = tk.StringVar(value='cuda')
        self.var_disable_blend = tk.BooleanVar(value=False)
        self.var_recursive_folders = tk.BooleanVar(value=True)
        self.var_output_dir = tk.StringVar(value="")
        self.var_output_media_clean = tk.BooleanVar(value=True)
        self.var_reduce_gpu = tk.BooleanVar(value=False)
        self.var_generate_otio = tk.BooleanVar(value=False)
        self.var_otio_client = tk.StringVar(value="")
        self.var_otio_wide = tk.StringVar(value="")
        self.var_otio_guest_closeup = tk.StringVar(value="")
        self.var_otio_host_closeup = tk.StringVar(value="")
        self.var_otio_extras = tk.StringVar(value="")
        self.var_otio_timeline_name = tk.StringVar(value="")
        self.client_camera_roles: dict[str, dict[str, str]] = {}
        self._otio_client_label_to_id: dict[str, str] = {}
        self._otio_label_to_path: dict[str, str] = {}
        self._otio_path_to_label: dict[str, str] = {}
        self._otio_thumb_images: dict[str, tk.PhotoImage] = {}
        self.cmb_otio_client = None
        self.cmb_otio_wide = None
        self.cmb_otio_guest = None
        self.cmb_otio_host = None
        self.lbl_thumb_wide = None
        self.lbl_thumb_guest = None
        self.lbl_thumb_host = None

        folder_opts = ttk.Frame(left_body, style='Section.TFrame', padding=10)
        folder_opts.pack(fill='x', pady=(0, 10))
        ttk.Checkbutton(
            folder_opts,
            text='Search subfolders when adding folders',
            variable=self.var_recursive_folders,
            style='Opt.TCheckbutton',
        ).pack(anchor='w')

        out_box = ttk.Frame(left_body, style='Section.TFrame', padding=10)
        out_box.pack(fill='x', pady=(0, 10))
        ttk.Label(out_box, text='Output folder (optional):', style='SectionTitle.TLabel').pack(anchor='w', pady=(0, 4))
        out_row = ttk.Frame(out_box, style='Section.TFrame')
        out_row.pack(fill='x', pady=(2, 0))
        self._output_dir_row = out_row
        out_entry = ttk.Entry(out_row, textvariable=self.var_output_dir)
        out_entry.pack(side='left', fill='x', expand=True)
        btn_out = ttk.Button(out_row, text='', width=3, command=self._choose_output_dir)
        btn_out.pack(side='left', padx=4)
        self._bind_hover(btn_out)
        btn_out_clear = ttk.Button(out_row, text='', width=3, command=self._clear_output_dir)
        btn_out_clear.pack(side='left')
        self._bind_hover(btn_out_clear)
        self._set_button_icon(btn_out, 'folder-open')
        self._set_button_icon(btn_out_clear, 'x')
        self._output_dir_widgets = (out_entry, btn_out, btn_out_clear)
        ttk.Checkbutton(
            out_box,
            text='Output to 01_MEDIA\\030_AUDIO_CLEAN (per parent)',
            variable=self.var_output_media_clean,
            style='Opt.TCheckbutton',
            command=self._toggle_media_clean_output,
        ).pack(anchor='w', pady=(4, 0))
        self._toggle_media_clean_output()

        adv_hdr = ttk.Frame(left_body, style='Section.TFrame', padding=10)
        adv_hdr.pack(fill='x', pady=(0, 10))
        self._adv_hdr = adv_hdr
        self._adv_btn = ttk.Button(adv_hdr, text='Processing & OTIO Options [+]', command=self._toggle_advanced)
        self._adv_btn.pack(side='left')
        self._bind_hover(self._adv_btn)

        self._adv_open = False
        self._adv_wrap = ttk.Frame(left_body, style='Section.TFrame', padding=12)

        # Diagnostics mode notice and advanced controls (collapsed by default)
        diag_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
        diag_box.pack(fill='x', pady=(0, 8))
        ttk.Label(diag_box, text='Diagnostics alignment mode is locked. Advanced options are temporarily removed.', style='SectionInfo.TLabel', wraplength=260, justify='left').pack(fill='x')
        synthesis_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
        synthesis_box.pack(fill='x', pady=(0, 8))
        ttk.Checkbutton(
            synthesis_box,
            text='Use AI enhancement / synthesis (slower)',
            variable=self.var_ai_synthesis,
            style='Opt.TCheckbutton',
            command=self._on_ai_synthesis_toggle,
        ).pack(anchor='w')
        noise_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
        noise_box.pack(fill='x', pady=(0, 8))
        ttk.Checkbutton(
            noise_box,
            text='Output background/noise only (invert denoise)',
            variable=self.var_noise_only,
            style='Opt.TCheckbutton'
        ).pack(anchor='w')
        if not self._hide_reduce_gpu_ui:
            gpu_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
            gpu_box.pack(fill='x', pady=(0, 8))
            ttk.Checkbutton(
                gpu_box,
                text='Reduce GPU pressure (slower)',
                variable=self.var_reduce_gpu,
                style='Opt.TCheckbutton'
            ).pack(anchor='w')
        else:
            self.var_reduce_gpu.set(False)
        sync_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
        sync_box.pack(fill='x', pady=(0, 8))
        if self._force_sync_export:
            self.var_sync_export.set(True)
        sync_check = ttk.Checkbutton(
            sync_box,
            text='Sync takes and export multichannel\n(uncheck for cleanup-only per file)',
            variable=self.var_sync_export,
            style='Opt.TCheckbutton',
        )
        sync_check.pack(anchor='w')
        if self._force_sync_export:
            sync_check.state(['disabled'])
        if not self._hide_otio_ui:
            otio_box = ttk.Frame(self._adv_wrap, style='Section.TFrame')
            otio_box.pack(fill='x', pady=(0, 8))
            ttk.Checkbutton(
                otio_box,
                text='Generate OTIO active-speaker timeline',
                variable=self.var_generate_otio,
                style='Opt.TCheckbutton',
            ).pack(anchor='w')
            ttk.Label(otio_box, text='Client for camera assignment:', style='SectionInfo.TLabel').pack(anchor='w')
            self.cmb_otio_client = ttk.Combobox(otio_box, textvariable=self.var_otio_client, state='readonly')
            self.cmb_otio_client.pack(fill='x', pady=(1, 2))
            self.cmb_otio_client.bind('<<ComboboxSelected>>', lambda e: self._on_otio_client_selected())
            ttk.Label(otio_box, text='Wide Camera (required for OTIO):', style='SectionInfo.TLabel').pack(anchor='w')
            row_w = ttk.Frame(otio_box, style='Section.TFrame')
            row_w.pack(fill='x', pady=(1, 2))
            self.cmb_otio_wide = ttk.Combobox(row_w, textvariable=self.var_otio_wide, state='readonly')
            self.cmb_otio_wide.pack(side='left', fill='x', expand=True)
            self.lbl_thumb_wide = ttk.Label(row_w, text='No thumb', style='SectionInfo.TLabel')
            self.lbl_thumb_wide.pack(side='left', padx=(6, 0))
            self.cmb_otio_wide.bind('<<ComboboxSelected>>', lambda e: self._on_otio_role_changed())
            ttk.Label(otio_box, text='Guest Closeup (optional):', style='SectionInfo.TLabel').pack(anchor='w')
            row_g = ttk.Frame(otio_box, style='Section.TFrame')
            row_g.pack(fill='x', pady=(1, 2))
            self.cmb_otio_guest = ttk.Combobox(row_g, textvariable=self.var_otio_guest_closeup, state='readonly')
            self.cmb_otio_guest.pack(side='left', fill='x', expand=True)
            self.lbl_thumb_guest = ttk.Label(row_g, text='No thumb', style='SectionInfo.TLabel')
            self.lbl_thumb_guest.pack(side='left', padx=(6, 0))
            self.cmb_otio_guest.bind('<<ComboboxSelected>>', lambda e: self._on_otio_role_changed())
            ttk.Label(otio_box, text='Host Closeup (optional):', style='SectionInfo.TLabel').pack(anchor='w')
            row_h = ttk.Frame(otio_box, style='Section.TFrame')
            row_h.pack(fill='x', pady=(1, 2))
            self.cmb_otio_host = ttk.Combobox(row_h, textvariable=self.var_otio_host_closeup, state='readonly')
            self.cmb_otio_host.pack(side='left', fill='x', expand=True)
            self.lbl_thumb_host = ttk.Label(row_h, text='No thumb', style='SectionInfo.TLabel')
            self.lbl_thumb_host.pack(side='left', padx=(6, 0))
            self.cmb_otio_host.bind('<<ComboboxSelected>>', lambda e: self._on_otio_role_changed())
            ttk.Label(otio_box, text='OTIO Timeline Name (optional):', style='SectionInfo.TLabel').pack(anchor='w')
            ttk.Entry(otio_box, textvariable=self.var_otio_timeline_name).pack(fill='x')
        else:
            self.var_generate_otio.set(False)

        queue_card = ttk.Frame(left_body, style='Section.TFrame', padding=10)
        queue_card.pack(fill='x', pady=(0, 0))
        ttk.Label(queue_card, text='Queue', style='SectionTitle.TLabel').pack(anchor='w', pady=(0, 6))

        # Queue controls: filter + actions
        ctl = ttk.Frame(queue_card, style='Section.TFrame')
        ctl.pack(fill='x', pady=(0, 6))
        ttk.Label(ctl, text='Show:', style='SectionInfo.TLabel').pack(side='left')
        self.status_filter_var = tk.StringVar(value='All')
        self.status_filter = ttk.Combobox(ctl, textvariable=self.status_filter_var, values=['All','Queued','Running','Done','Failed'], state='readonly', width=10)
        self.status_filter.pack(side='left', padx=(6, 0))
        self.status_filter.bind('<<ComboboxSelected>>', lambda e: self._refresh_queue_tree())
        self.trash_btn = ttk.Button(ctl, text='', width=3, command=self._remove_selected)
        self.trash_btn.pack(side='right')
        self._bind_hover(self.trash_btn)
        self.clear_queue_btn = ttk.Button(ctl, text='', width=3, command=self.clear_files)
        self.clear_queue_btn.pack(side='right', padx=(0, 6))
        self._bind_hover(self.clear_queue_btn)
        self._set_button_icon(self.trash_btn, 'trash-2')
        self._set_button_icon(self.clear_queue_btn, 'x')

        # Queue tree: grouped folders with child files; multi-select enabled
        self.queue_tree = ttk.Treeview(queue_card, show='tree', selectmode='extended', height=12)
        self.queue_tree.pack(fill='x', expand=False)
        # Drag-to-reorder support (when filter is 'All')
        self._drag_iid = None
        self._drag_line = None
        self._drag_ghost = None
        self._drag_before = True
        def _drag_cleanup():
            self._drag_iid = None
            # reset cursor and remove indicators/ghost
            try:
                self.queue_tree.configure(cursor='')
            except Exception:
                pass
            try:
                if self._drag_line is not None:
                    self._drag_line.place_forget()
            except Exception:
                pass
            if self._drag_ghost is not None:
                try:
                    self._drag_ghost.destroy()
                except Exception:
                    pass
                self._drag_ghost = None
        def _make_ghost(text: str):
            try:
                g = tk.Toplevel(self)
                g.overrideredirect(True)
                try:
                    g.wm_attributes('-alpha', 0.88)
                except Exception:
                    pass
                lbl = ttk.Label(g, text=text)
                lbl.pack(ipadx=6, ipady=3)
                return g
            except Exception:
                return None
        def _show_line(y: int):
            try:
                if self._drag_line is None:
                    self._drag_line = tk.Frame(self.queue_tree, height=2, bg=self._accent)
                w = max(1, self.queue_tree.winfo_width() - 2)
                self._drag_line.place(x=1, y=y, width=w)
            except Exception:
                pass
        def _hide_line():
            try:
                if self._drag_line is not None:
                    self._drag_line.place_forget()
            except Exception:
                pass
        def _q_on_press(e):
            iid = self.queue_tree.identify_row(e.y)
            # allow dragging files or folders
            if iid and (iid in self._iid_to_path or iid in self._iid_to_folder):
                self._drag_iid = iid
                # ghost preview
                txt = self.queue_tree.item(iid, 'text') or ''
                self._drag_ghost = _make_ghost(txt)
                if self._drag_ghost is not None:
                    try:
                        self._drag_ghost.geometry(f"+{e.x_root+12}+{e.y_root+12}")
                    except Exception:
                        pass
                try:
                    self.queue_tree.configure(cursor='fleur')
                except Exception:
                    pass
            else:
                self._drag_iid = None
                _hide_line()
                _drag_cleanup()
        def _q_on_release(e):
            src = self._drag_iid
            self._drag_iid = None
            if not src or self.status_filter_var.get() != 'All':
                _drag_cleanup()
                return
            dst = self.queue_tree.identify_row(e.y)
            if not dst or dst == src:
                _drag_cleanup()
                return
            # Determine types
            spath = self._iid_to_path.get(src)
            sfolder = self._iid_to_folder.get(src)
            dpath = self._iid_to_path.get(dst)
            dfolder = self._iid_to_folder.get(dst)
            try:
                if spath:  # dragging a file
                    # compute source index
                    si = self.files.index(spath)
                    if dpath:  # drop on file
                        base = self.files.index(dpath)
                        ti = base if self._drag_before else (base + 1)
                    elif dfolder:  # drop on folder row – before/after the whole block
                        # Compute start and end of the folder block
                        indices = [i for i, f in enumerate(self.files) if str(Path(f).parent) == dfolder]
                        if not indices:
                            ti = len(self.files)
                        else:
                            start = min(indices)
                            end = max(indices)
                            ti = start if self._drag_before else (end + 1)
                    else:
                        return
                    itm = self.files.pop(si)
                    if si < ti:
                        ti -= 1
                    self.files.insert(ti, itm)
                    moved = spath
                elif sfolder:  # dragging a folder (group block)
                    # collect block for src folder
                    src_block = [f for f in self.files if str(Path(f).parent) == sfolder]
                    if not src_block:
                        return
                    # remove block
                    remaining = [f for f in self.files if str(Path(f).parent) != sfolder]
                    # determine insertion index
                    if dfolder:
                        # indices for dst folder
                        dinds = [i for i, f in enumerate(remaining) if str(Path(f).parent) == dfolder]
                        if not dinds:
                            ti = len(remaining)
                        else:
                            start = min(dinds)
                            end = max(dinds)
                            ti = start if self._drag_before else (end + 1)
                    elif dpath:
                        # index of that file in remaining (before/after)
                        try:
                            base = remaining.index(dpath)
                            ti = base if self._drag_before else (base + 1)
                        except ValueError:
                            ti = len(remaining)
                    else:
                        return
                    self.files = remaining[:ti] + src_block + remaining[ti:]
                    moved = src_block[0]
                else:
                    return
            except Exception:
                return
            # Refresh and reselect moved item/group head
            self._refresh_queue_tree()
            sel_iid = self._path_to_iid.get(moved)
            if sel_iid:
                try:
                    self.queue_tree.selection_set(sel_iid)
                except Exception:
                    pass
            # cleanup visuals
            _drag_cleanup()
        def _q_on_motion(e):
            if not self._drag_iid:
                return
            # move ghost
            if self._drag_ghost is not None:
                try:
                    self._drag_ghost.geometry(f"+{e.x_root+12}+{e.y_root+12}")
                except Exception:
                    pass
            # update insertion line
            dst = self.queue_tree.identify_row(e.y)
            if not dst:
                _hide_line()
                return
            try:
                x, y, w, h = self.queue_tree.bbox(dst)
            except Exception:
                _hide_line()
                return
            # Determine before/after based on pointer position within row
            self._drag_before = (e.y < (y + h/2))
            y_line = y if self._drag_before else (y + h)
            _show_line(y_line)
        self.queue_tree.bind('<ButtonPress-1>', _q_on_press)
        self.queue_tree.bind('<B1-Motion>', _q_on_motion)
        self.queue_tree.bind('<ButtonRelease-1>', _q_on_release)
        self.queue_tree.bind('<BackSpace>', lambda e: (self._remove_selected(), 'break')[1])
        self.queue_tree.bind('<Delete>', lambda e: (self._remove_selected(), 'break')[1])
        self.bind('<BackSpace>', self._queue_key_remove, add='+')
        self.bind('<Delete>', self._queue_key_remove, add='+')
        # Cleanup on escape, focus-out, or pointer leaving the widget while dragging
        self.queue_tree.bind('<Leave>', lambda e: (_drag_cleanup()))
        self.bind('<Escape>', lambda e: (_drag_cleanup()))
        self.queue_tree.bind('<Escape>', lambda e: (_drag_cleanup()))
        # Click anywhere outside the queue to clear selection
        def _is_child_of(w, parent):
            try:
                while w is not None:
                    if w == parent:
                        return True
                    w = w.master
            except Exception:
                return False
            return False
        def _maybe_clear_selection(e):
            if not _is_child_of(e.widget, self.queue_tree):
                try:
                    self.queue_tree.selection_remove(self.queue_tree.selection())
                except Exception:
                    pass
        self.bind('<Button-1>', _maybe_clear_selection, add='+')

        # Context menu for queue actions
        self._q_menu = tk.Menu(self, tearoff=0)
        self._q_menu.add_command(label='Remove Selected', command=self._remove_selected)
        self._q_menu.add_command(label='Clear Queue', command=self.clear_files)
        self._q_menu.add_command(label='Clear Processed', command=self._clear_processed)
        def _q_menu_popup(e):
            try:
                self._q_menu.tk_popup(e.x_root, e.y_root)
            finally:
                try:
                    self._q_menu.grab_release()
                except Exception:
                    pass
        self.queue_tree.bind('<Button-3>', _q_menu_popup)

        dnd_enabled = False
        if DND_AVAILABLE:
            try:
                self.queue_tree.drop_target_register(DND_FILES)
                self.queue_tree.dnd_bind('<<Drop>>', self._on_drop)
                dnd_enabled = True
            except Exception:
                dnd_enabled = False
        if not dnd_enabled:
            _install_win_dnd(self.queue_tree, lambda files: self._add_paths(files) or self._enable_run())

        progfrm = ttk.Frame(right, style='Section.TFrame', padding=12)
        progfrm.pack(fill='x', pady=(0, 12))
        ttk.Label(progfrm, text='Progress', style='SectionTitle.TLabel').pack(anchor='w', pady=(0, 6))
        self.progress = ttk.Progressbar(progfrm, mode='determinate')
        self.progress.pack(fill='x')
        self.overall_label = ttk.Label(progfrm, text='0 of 0 files', style='SectionInfo.TLabel')
        self.overall_label.pack(anchor='w', pady=(2, 0))
        # Status line below the main progress bar
        self.status_label = ttk.Label(progfrm, text='', style='SectionInfo.TLabel')
        self.status_label.pack(fill='x', pady=(6, 0))

        hist_card = ttk.Frame(right, style='Section.TFrame', padding=12)
        hist_card.pack(fill='x')
        ttk.Label(hist_card, text='Enhanced files', style='SectionTitle.TLabel').pack(anchor='w', pady=(0, 6))
        histfrm = ttk.Frame(hist_card, style='Section.TFrame')
        histfrm.pack(fill='x')
        self.hist = ttk.Treeview(histfrm, columns=('src','out'), show='headings', selectmode='browse', height=7)
        self.hist.heading('src', text='Source')
        self.hist.heading('out', text='Output')
        # Equal width, left-aligned, stretch to fit
        self.hist.column('src', anchor='w', stretch=True, minwidth=160, width=160)
        self.hist.column('out', anchor='w', stretch=True, minwidth=160, width=160)
        # Use grid so scrollbars sit flush with the Treeview
        self.hist.grid(row=0, column=0, sticky='nsew')
        # Subtle alternating row shades to improve readability
        try:
            self.hist.tag_configure('odd', background='#1b1b1b')
            self.hist.tag_configure('even', background=self._panel)
        except Exception:
            pass
        # Scrollbars
        vsb = ttk.Scrollbar(histfrm, orient='vertical', command=self.hist.yview)
        hsb = ttk.Scrollbar(histfrm, orient='horizontal', command=self.hist.xview)
        self.hist.configure(yscroll=vsb.set, xscroll=hsb.set)
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        histfrm.grid_rowconfigure(0, weight=1)
        histfrm.grid_columnconfigure(0, weight=1)
        # Keep columns equal and within frame width
        def _resize_hist_cols(event=None):
            try:
                w = max(0, self.hist.winfo_width() - (vsb.winfo_width() or 18))
                cw = max(120, int(w/2))
                self.hist.column('src', width=cw)
                self.hist.column('out', width=cw)
            except Exception:
                pass
        self.hist.bind('<Configure>', _resize_hist_cols)
        self.after(200, _resize_hist_cols)
        self._hist_menu = tk.Menu(self, tearoff=0)
        self._hist_menu.add_command(label='Open Output', command=self._open_selected_output)
        self._hist_menu.add_command(label='Reveal in Explorer', command=self._reveal_selected_output)
        def _hist_menu_popup(e):
            row_id = self.hist.identify_row(e.y)
            if row_id:
                try:
                    self.hist.selection_set(row_id)
                    self.hist.focus(row_id)
                except Exception:
                    pass
                try:
                    self._hist_menu.tk_popup(e.x_root, e.y_root)
                finally:
                    try:
                        self._hist_menu.grab_release()
                    except Exception:
                        pass
        self.hist.bind('<Button-3>', _hist_menu_popup)
        btnhist = ttk.Frame(hist_card, style='Section.TFrame')
        btnhist.pack(fill='x', pady=(12, 0))
        output_actions = ttk.Frame(btnhist, style='Section.TFrame')
        output_actions.pack(side='left', fill='x', expand=True)
        btn_open = ttk.Button(output_actions, text='Open Selected Output', command=self._open_selected_output)
        btn_open.pack(side='left')
        self._bind_hover(btn_open)
        btn_gr = ttk.Button(output_actions, text='Open Preview', command=self._open_gradio_preview)
        btn_gr.pack(side='left', padx=(8,0))
        self._bind_hover(btn_gr)
        self._set_button_icon(btn_open, 'external-link')
        self._set_button_icon(btn_gr, 'external-link')
        bottom_actions = ttk.Frame(btnhist, style='Section.TFrame')
        bottom_actions.pack(side='right')
        ctrlfrm = ttk.Frame(bottom_actions, style='Section.TFrame')
        ctrlfrm.pack(side='left', padx=(0, 10))
        self.pause_btn = ttk.Button(ctrlfrm, text='', width=3, command=self._toggle_pause, state='disabled')
        self.pause_btn.pack(side='left')
        self.cancel_btn = ttk.Button(ctrlfrm, text='', width=3, command=self._cancel_graceful, state='disabled')
        self.cancel_btn.pack(side='left', padx=(6, 0))
        self._set_button_icon(self.pause_btn, 'pause')
        self._set_button_icon(self.cancel_btn, 'x')
        self.run_btn = ttk.Button(bottom_actions, text='Enhance', command=self.run_task, state='disabled', style='Accent.TButton')
        self.run_btn.pack(side='left')
        self.run_btn.bind('<Enter>', lambda e: self.run_btn.configure(style='AccentHover.TButton'))
        self.run_btn.bind('<Leave>', lambda e: self.run_btn.configure(style='Accent.TButton'))
        self._set_button_icon(self.run_btn, 'play')

        # Preview handled via Gradio in a browser; no inline preview widgets
        self._closing = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[
                ("Audio files", ".wav .WAV .mp3 .MP3"),
                ("WAV files", ".wav .WAV"),
                ("MP3 files", ".mp3 .MP3"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self._add_paths(paths)
        self._enable_run()

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing audio")
        if not folder:
            return
        # Legacy fallback mode
        self.client_mode_active = False
        self._add_paths([folder])
        self._enable_run()

    def _scan_client_folder(self, client_root: Path) -> tuple[Path, list[str], list[str], str | None]:
        try:
            # Accept selecting either client root or the 01_MEDIA folder directly.
            if client_root.name.lower() == MEDIA_ROOT_FOLDER.lower():
                resolved_root = client_root.parent
                media_root = client_root
            else:
                resolved_root = client_root
                media_root = client_root / MEDIA_ROOT_FOLDER
            audio_dir = media_root / MEDIA_AUDIO_RAW_FOLDER
            video_dir = media_root / MEDIA_VIDEO_RAW_FOLDER
            if not audio_dir.exists() or not audio_dir.is_dir():
                return resolved_root, [], [], f"missing {MEDIA_ROOT_FOLDER}\\{MEDIA_AUDIO_RAW_FOLDER}"
            if not video_dir.exists() or not video_dir.is_dir():
                return resolved_root, [], [], f"missing {MEDIA_ROOT_FOLDER}\\{MEDIA_VIDEO_RAW_FOLDER}"
            audio_files: list[str] = []
            for p in sorted(audio_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in {".wav", ".wave", ".mp3"}:
                    audio_files.append(str(p))
            video_files: list[str] = []
            for p in sorted(video_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".mts", ".m2ts"}:
                    video_files.append(str(p))
            if not audio_files:
                return resolved_root, [], video_files, f"no audio in {MEDIA_ROOT_FOLDER}\\{MEDIA_AUDIO_RAW_FOLDER}"
            if not video_files:
                return resolved_root, audio_files, [], f"no video in {MEDIA_ROOT_FOLDER}\\{MEDIA_VIDEO_RAW_FOLDER}"
            return resolved_root, audio_files, video_files, None
        except Exception as exc:
            return client_root, [], [], str(exc)

    def _apply_client_specs(self, specs: list[dict], source_desc: str) -> None:
        # Replace queue with discovered client audio inputs.
        self.files.clear()
        self.folders.clear()
        self.file_status.clear()
        self.client_queue.clear()
        self.client_meta.clear()
        self.client_order.clear()
        self.file_to_client.clear()
        ready_clients = 0
        skipped_clients = 0
        detected_existing = 0
        for spec in specs:
            cid = str(spec.get("client_id", ""))
            croot = str(spec.get("client_root", ""))
            afiles = list(spec.get("audio_files", []) or [])
            vfiles = list(spec.get("video_files", []) or [])
            reason = spec.get("skip_reason")
            self.client_meta[cid] = {
                "client_root": croot,
                "video_files": vfiles,
                "skip_reason": reason,
            }
            self.client_order.append(cid)
            if reason:
                skipped_clients += 1
                self.client_queue[cid] = []
                continue
            self.client_queue[cid] = []
            for fp in afiles:
                if self._has_existing_clean_output(fp):
                    detected_existing += 1
                if fp in self.file_to_client:
                    continue
                self.files.append(fp)
                self.file_status[fp] = "queued"
                self.file_to_client[fp] = cid
                self.client_queue[cid].append(fp)
                try:
                    self.folders.add(str(Path(fp).parent))
                except Exception:
                    pass
            if self.client_queue[cid]:
                ready_clients += 1
            else:
                skipped_clients += 1
                self.client_meta[cid]["skip_reason"] = "no discovered audio files"
        self.client_mode_active = True
        self._refresh_queue_tree()
        self._refresh_otio_client_dropdown()
        self._enable_run()
        total_files = len(self.files)
        self._log(
            f"[client] {source_desc}: ready clients={ready_clients}, skipped={skipped_clients}, "
            f"queued files={total_files}, detected_existing_clean={detected_existing}"
        )
        for cid in self.client_order:
            meta = self.client_meta.get(cid, {})
            reason = str(meta.get("skip_reason") or "").strip()
            if reason:
                root_name = Path(str(meta.get("client_root") or cid)).name
                self._log(f"[client] skipped {root_name}: {reason}")
        try:
            messagebox.showinfo(
                "Client Selection",
                f"Ready clients: {ready_clients}\nSkipped clients: {skipped_clients}\nQueued files: {total_files}",
                parent=self,
            )
        except Exception:
            pass

    def _select_project_folder(self):
        folder = filedialog.askdirectory(title="Select project folder (contains client folders)")
        if not folder:
            return
        parent = Path(folder)
        specs: list[dict] = []
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            resolved_root, audio_files, video_files, reason = self._scan_client_folder(child)
            cid = str(resolved_root.resolve())
            specs.append({
                "client_id": cid,
                "client_root": str(resolved_root),
                "audio_files": audio_files,
                "video_files": video_files,
                "skip_reason": reason,
            })
        if not specs:
            self._log(f"[client] no client folders found under {folder}")
            return
        self._apply_client_specs(specs, source_desc=f"project selected {folder}")

    def _select_single_client_folder(self):
        folder = filedialog.askdirectory(title="Select single client folder")
        if not folder:
            return
        root = Path(folder)
        resolved_root, audio_files, video_files, reason = self._scan_client_folder(root)
        cid = str(resolved_root.resolve())
        specs = [{
            "client_id": cid,
            "client_root": str(resolved_root),
            "audio_files": audio_files,
            "video_files": video_files,
            "skip_reason": reason,
        }]
        self._apply_client_specs(specs, source_desc=f"single client selected {folder}")

    def _confirm_queue_before_run(self) -> bool:
        """Mandatory pre-run confirmation with client/file removal support."""
        if not self.files:
            return False
        temp_files = list(self.files)
        temp_status = dict(self.file_status)
        temp_f2c = dict(self.file_to_client)
        temp_meta = dict(self.client_meta)
        temp_order = list(self.client_order)
        if not temp_f2c:
            # Legacy fallback: synthesize client groups by parent folder.
            for fp in temp_files:
                cid = str(Path(fp).parent)
                temp_f2c[fp] = cid
                if cid not in temp_meta:
                    temp_meta[cid] = {"client_root": cid, "video_files": [], "skip_reason": None}
                if cid not in temp_order:
                    temp_order.append(cid)

        dlg = tk.Toplevel(self)
        dlg.title("Confirm Queue")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("780x500")
        ttk.Label(dlg, text="Review queue before execution", style='Title.TLabel').pack(anchor='w', padx=10, pady=(8, 4))
        info_var = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=info_var, style='Info.TLabel').pack(anchor='w', padx=10, pady=(0, 6))

        tree = ttk.Treeview(dlg, show='tree')
        tree.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        iid_to_file: dict[str, str] = {}
        iid_to_client: dict[str, str] = {}
        result = {"confirmed": False}

        def _groups_from_temp() -> dict[str, list[str]]:
            g: dict[str, list[str]] = {}
            for fp in temp_files:
                cid = temp_f2c.get(fp, str(Path(fp).parent))
                g.setdefault(cid, []).append(fp)
            return g

        def _refresh_dialog_tree():
            for iid in tree.get_children(""):
                tree.delete(iid)
            iid_to_file.clear()
            iid_to_client.clear()
            groups = _groups_from_temp()
            ccount = 0
            for cid in temp_order:
                files = groups.get(cid, [])
                if not files:
                    continue
                ccount += 1
                meta = temp_meta.get(cid, {})
                cname = Path(str(meta.get("client_root") or cid)).name or cid
                cnode = tree.insert("", "end", text=f"{cname} ({len(files)} file(s))", open=True)
                iid_to_client[cnode] = cid
                for fp in files:
                    fnode = tree.insert(cnode, "end", text=Path(fp).name)
                    iid_to_file[fnode] = fp
            info_var.set(f"Clients: {ccount} | Files: {len(temp_files)}")

        def _remove_client():
            sels = list(tree.selection())
            if not sels:
                return
            remove_cids = set()
            for iid in sels:
                cid = iid_to_client.get(iid)
                if cid:
                    remove_cids.add(cid)
                else:
                    parent = tree.parent(iid)
                    if parent:
                        cid2 = iid_to_client.get(parent)
                        if cid2:
                            remove_cids.add(cid2)
            if not remove_cids:
                return
            keep = []
            for fp in temp_files:
                if temp_f2c.get(fp) not in remove_cids:
                    keep.append(fp)
            temp_files[:] = keep
            _refresh_dialog_tree()

        def _remove_file():
            sels = list(tree.selection())
            if not sels:
                return
            remove_files = {iid_to_file.get(iid) for iid in sels if iid_to_file.get(iid)}
            if not remove_files:
                return
            temp_files[:] = [fp for fp in temp_files if fp not in remove_files]
            _refresh_dialog_tree()

        btns = ttk.Frame(dlg)
        btns.pack(fill='x', padx=10, pady=(0, 10))
        b_rm_client = ttk.Button(btns, text="Remove Client", command=_remove_client)
        b_rm_client.pack(side='left')
        self._bind_hover(b_rm_client)
        b_rm_file = ttk.Button(btns, text="Remove File", command=_remove_file)
        b_rm_file.pack(side='left', padx=(6, 0))
        self._bind_hover(b_rm_file)

        def _on_confirm():
            if not temp_files:
                messagebox.showwarning("Empty Queue", "No files remain in the queue. Nothing to run.", parent=dlg)
                return
            self.files = list(temp_files)
            self.file_status = {fp: temp_status.get(fp, "queued") for fp in self.files}
            self.file_to_client = {fp: temp_f2c.get(fp, str(Path(fp).parent)) for fp in self.files}
            # Rebuild client_queue and prune stale client metadata.
            new_client_queue: dict[str, list[str]] = {}
            for fp in self.files:
                cid = self.file_to_client.get(fp, str(Path(fp).parent))
                new_client_queue.setdefault(cid, []).append(fp)
            self.client_queue = new_client_queue
            self.client_order = [cid for cid in temp_order if cid in new_client_queue]
            self.client_meta = {cid: temp_meta.get(cid, {"client_root": cid, "video_files": [], "skip_reason": None}) for cid in self.client_order}
            self._refresh_queue_tree()
            self._enable_run()
            result["confirmed"] = True
            self._log(f"[queue] confirmed: clients={len(self.client_queue)}, files={len(self.files)}")
            dlg.destroy()

        def _on_cancel():
            result["confirmed"] = False
            dlg.destroy()

        b_cancel = ttk.Button(btns, text="Cancel", command=_on_cancel)
        b_cancel.pack(side='right')
        self._bind_hover(b_cancel)
        b_ok = ttk.Button(btns, text="Confirm Run", command=_on_confirm, style='Accent.TButton')
        b_ok.pack(side='right', padx=(0, 6))
        _refresh_dialog_tree()
        dlg.wait_window()
        return bool(result["confirmed"])

    def _choose_output_dir(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder:
            return
        self.var_output_dir.set(folder)
        try:
            if self.var_output_media_clean.get():
                self.var_output_media_clean.set(False)
                self._toggle_media_clean_output()
        except Exception:
            pass

    def _clear_output_dir(self):
        self.var_output_dir.set("")

    def _toggle_media_clean_output(self):
        enabled = bool(self.var_output_media_clean.get())
        state = "disabled" if enabled else "normal"
        try:
            for w in getattr(self, "_output_dir_widgets", ()):
                w.configure(state=state)
        except Exception:
            pass
        try:
            row = getattr(self, "_output_dir_row", None)
            if row is not None:
                if enabled:
                    row.pack_forget()
                elif not row.winfo_manager():
                    row.pack(fill='x', pady=(2, 0))
        except Exception:
            pass

    def _choose_otio_video(self, target_var):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", ".mp4 .MP4 .mov .MOV .mxf .MXF .mkv .MKV .avi .AVI"),
                ("All files", "*.*"),
            ],
        )
        if path:
            try:
                target_var.set(path)
            except Exception:
                pass

    def _auto_suggest_camera_roles(self, video_files: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        if not video_files:
            return out
        files = list(video_files)
        lowers = [Path(p).name.lower() for p in files]
        def _pick(keys: list[str]) -> str | None:
            for k in keys:
                for i, nm in enumerate(lowers):
                    if k in nm:
                        return files[i]
            return None
        wide = _pick(["wide", "master", "program", "all"])
        if not wide and files:
            wide = files[0]
        guest = _pick(["guest", "cam_b", "b_cam"])
        host = _pick(["host", "interviewer", "cam_a", "a_cam"])
        if wide:
            out["wide"] = wide
        if guest and guest != wide:
            out["guest_closeup"] = guest
        if host and host != wide:
            out["host_closeup"] = host
        return out

    def _get_current_otio_client_id(self) -> str | None:
        label = str(self.var_otio_client.get() or "").strip()
        if not label:
            return None
        return self._otio_client_label_to_id.get(label)

    def _save_otio_roles_for_current_client(self) -> None:
        cid = self._get_current_otio_client_id()
        if not cid:
            return
        wide_path = self._otio_label_to_path.get(str(self.var_otio_wide.get() or "").strip(), "")
        guest_path = self._otio_label_to_path.get(str(self.var_otio_guest_closeup.get() or "").strip(), "")
        host_path = self._otio_label_to_path.get(str(self.var_otio_host_closeup.get() or "").strip(), "")
        self.client_camera_roles[cid] = {
            "wide": wide_path,
            "guest_closeup": guest_path,
            "host_closeup": host_path,
        }

    def _ensure_video_thumbnail(self, video_path: str, width: int = 120, height: int = 68) -> tk.PhotoImage | None:
        p = str(video_path or "").strip()
        if not p:
            return None
        if p in self._otio_thumb_images:
            return self._otio_thumb_images[p]
        try:
            thumbs_dir = INPUT_TMP_ROOT / "thumbs"
            thumbs_dir.mkdir(parents=True, exist_ok=True)
            src = Path(p)
            if not src.exists():
                return None
            key = f"{src.stem}_{int(src.stat().st_mtime)}_{abs(hash(str(src.resolve()))) & 0xfffffff}.png"
            outp = thumbs_dir / key
            if not outp.exists():
                ff = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                if not ff:
                    return None
                cmd = [
                    ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "00:00:02.000",
                    "-i", str(src),
                    "-frames:v", "1",
                    "-vf", f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease,pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2",
                    str(outp),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if proc.returncode != 0 or not outp.exists():
                    return None
            img = tk.PhotoImage(file=str(outp))
            self._otio_thumb_images[p] = img
            return img
        except Exception:
            return None

    def _refresh_otio_role_thumbnails(self) -> None:
        role_map = [
            (str(self.var_otio_wide.get() or "").strip(), getattr(self, "lbl_thumb_wide", None)),
            (str(self.var_otio_guest_closeup.get() or "").strip(), getattr(self, "lbl_thumb_guest", None)),
            (str(self.var_otio_host_closeup.get() or "").strip(), getattr(self, "lbl_thumb_host", None)),
        ]
        for label, widget in role_map:
            if widget is None:
                continue
            p = self._otio_label_to_path.get(label, "")
            img = self._ensure_video_thumbnail(p) if p else None
            try:
                if img is None:
                    widget.configure(image="", text="No thumb")
                else:
                    widget.configure(image=img, text="")
                    widget.image = img  # keep ref
            except Exception:
                pass

    def _on_otio_role_changed(self) -> None:
        self._save_otio_roles_for_current_client()
        self._refresh_otio_role_thumbnails()

    def _on_otio_client_selected(self) -> None:
        self._refresh_otio_camera_dropdowns()

    def _refresh_otio_client_dropdown(self) -> None:
        labels: list[str] = []
        self._otio_client_label_to_id.clear()
        if self.client_mode_active and self.client_order:
            for cid in self.client_order:
                meta = self.client_meta.get(cid, {})
                root = str(meta.get("client_root", cid))
                name = Path(root).name or cid
                label = name
                if label in self._otio_client_label_to_id:
                    label = f"{name} [{cid[-6:]}]"
                self._otio_client_label_to_id[label] = cid
                labels.append(label)
        try:
            self.cmb_otio_client.configure(values=labels)
        except Exception:
            pass
        if labels:
            cur = str(self.var_otio_client.get() or "")
            if cur not in labels:
                self.var_otio_client.set(labels[0])
        else:
            self.var_otio_client.set("")
        self._refresh_otio_camera_dropdowns()

    def _refresh_otio_camera_dropdowns(self) -> None:
        cid = self._get_current_otio_client_id()
        videos: list[str] = []
        if cid:
            videos = [str(v) for v in (self.client_meta.get(cid, {}).get("video_files", []) or []) if str(v).strip()]
        self._otio_label_to_path.clear()
        self._otio_path_to_label.clear()
        labels: list[str] = []
        if videos:
            seen_labels: dict[str, int] = {}
            for p in videos:
                base = Path(p).name
                n = seen_labels.get(base, 0) + 1
                seen_labels[base] = n
                label = base if n == 1 else f"{base} ({n})"
                self._otio_label_to_path[label] = p
                self._otio_path_to_label[p] = label
                labels.append(label)
        values = [""] + labels
        for cmb in (getattr(self, "cmb_otio_wide", None), getattr(self, "cmb_otio_guest", None), getattr(self, "cmb_otio_host", None)):
            try:
                if cmb is not None:
                    cmb.configure(values=values)
            except Exception:
                pass
        # Load saved or auto-suggested values for selected client.
        roles = dict(self.client_camera_roles.get(cid or "", {}))
        if not roles and videos:
            roles = self._auto_suggest_camera_roles(videos)
            if cid:
                self.client_camera_roles[cid] = dict(roles)
        self.var_otio_wide.set(str(self._otio_path_to_label.get(str(roles.get("wide", "")), "")))
        self.var_otio_guest_closeup.set(str(self._otio_path_to_label.get(str(roles.get("guest_closeup", "")), "")))
        self.var_otio_host_closeup.set(str(self._otio_path_to_label.get(str(roles.get("host_closeup", "")), "")))
        self._refresh_otio_role_thumbnails()

    def _build_otio_camera_roles(self, client_id: str | None = None) -> dict[str, str]:
        roles: dict[str, str] = {}
        if client_id:
            stored = self.client_camera_roles.get(client_id, {})
            wide = str(stored.get("wide", "") or "").strip()
            guest = str(stored.get("guest_closeup", "") or "").strip()
            host = str(stored.get("host_closeup", "") or "").strip()
            if not wide:
                videos = [str(v) for v in (self.client_meta.get(client_id, {}).get("video_files", []) or []) if str(v).strip()]
                guessed = self._auto_suggest_camera_roles(videos)
                wide = str(guessed.get("wide", "") or "").strip()
                guest = str(guessed.get("guest_closeup", "") or guest).strip()
                host = str(guessed.get("host_closeup", "") or host).strip()
                if guessed:
                    self.client_camera_roles[client_id] = dict(guessed)
        else:
            cid = self._get_current_otio_client_id()
            if cid:
                return self._build_otio_camera_roles(client_id=cid)
            videos = [str(p) for p in self.files if _is_video_file(str(p))]
            guessed = self._auto_suggest_camera_roles(videos)
            wide = str(guessed.get("wide", "") or "").strip()
            guest = str(guessed.get("guest_closeup", "") or "").strip()
            host = str(guessed.get("host_closeup", "") or "").strip()
        if wide:
            roles["wide"] = wide
        if guest:
            roles["guest_closeup"] = guest
        if host:
            roles["host_closeup"] = host
        return roles

    def clear_files(self):
        self._clear_queue()
        self._log_clear()
        self._refresh_otio_client_dropdown()

    # Removed the old modal preview dialog in favor of inline media preview

    def _render_preview_segment(self, src_path: str, start_s: float, dur_s: float) -> str:
        """Process a short segment with current settings and write a temp WAV; returns path."""
        import torchaudio
        from torchaudio.functional import resample as ta_resample
        # Export current env knobs
        try:
            cs = max(1.0, float(self.var_chunk_sec.get()))
            ov = max(0.0, float(self.var_overlap_sec.get()))
            os.environ['RESEMBLE_CHUNK_SECONDS'] = str(cs)
            os.environ['RESEMBLE_OVERLAP_SECONDS'] = str(ov)
            os.environ['RESEMBLE_DISABLE_TRANSIENT_BLEND'] = '1' if (getattr(self, 'var_disable_blend', None) and self.var_disable_blend.get()) else '0'
            try:
                os.environ['RESEMBLE_WET'] = str(max(0.0, min(1.0, float(self.var_wet.get()))))
            except Exception:
                pass
            os.environ['RESEMBLE_NOISE_ONLY'] = '1' if self.var_noise_only.get() else '0'
            aggr = self.var_aggressive_denoise.get() or self.var_diag_minimal.get()
            os.environ['RESEMBLE_DENOISE_AGGRESSIVE'] = '1' if aggr else '0'
        except Exception:
            pass
        noise_only_flag = False
        try:
            noise_only_flag = os.environ.get('RESEMBLE_NOISE_ONLY', '0') == '1'
        except Exception:
            noise_only_flag = False

        # Load source and slice segment
        wav, sr = torchaudio.load(str(src_path))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
        total_len = int(wav.shape[-1]) if wav.ndim == 1 else int(wav.size(-1))
        if total_len <= 0:
            raise RuntimeError('Empty audio file or failed decode')
        # Compute safe segment indices
        seg_start = int(round(max(0.0, float(start_s)) * float(sr)))
        seg_len = int(round(max(0.0, float(dur_s)) * float(sr)))
        if seg_start >= total_len:
            seg_start = max(0, total_len - min(int(sr), total_len))
        if seg_len <= 0:
            seg_len = min(int(5 * sr), total_len - seg_start)
        seg_end = min(total_len, seg_start + seg_len)
        if seg_end <= seg_start:
            seg_end = min(total_len, seg_start + 1)
        wav = wav[seg_start:seg_end]
        # Final guard: ensure non-empty segment
        if wav.numel() <= 0:
            import torch as _t
            wav = _t.zeros(256, dtype=_t.float32)

        # Process with current seam settings
        from resemble_enhance.enhancer.inference import denoise, enhance
        run_dir = _get_enhancer_run_dir()
        if self.var_seam_safe.get():
            kwargs = dict(chunk_seconds=float(os.environ.get('RESEMBLE_CHUNK_SECONDS', '60.0') or 60.0), overlap_seconds=float(os.environ.get('RESEMBLE_OVERLAP_SECONDS', '4.0') or 4.0), align_max_shift_ratio=0.05, align_disable=False)
        else:
            kwargs = dict(chunk_seconds=float(os.environ.get('RESEMBLE_CHUNK_SECONDS', '31.0') or 31.0), overlap_seconds=float(os.environ.get('RESEMBLE_OVERLAP_SECONDS', '1.0') or 1.0), align_max_shift_ratio=0.25, align_disable=True)
        if self.var_denoise_only.get():
            cs_safe = max(45.0, float(kwargs.get("chunk_seconds", 31.0)))
            ov_safe = max(2.5, float(kwargs.get("overlap_seconds", 1.0)))
            ov_safe = min(ov_safe, cs_safe / 4.0)
            kwargs.update(chunk_seconds=cs_safe, overlap_seconds=ov_safe, align_max_shift_ratio=0.0, align_disable=True)
        device = 'cuda'
        try:
            device = str(self.var_device.get() or 'cuda')
        except Exception:
            device = 'cuda'

        def _preview_run(device_):
            if self.var_denoise_only.get():
                return denoise(dwav=wav, sr=sr, device=device_, run_dir=run_dir, **kwargs)
            extra = {}
            if str(device_).lower() == "cuda":
                extra.update(nfe=AI_SYNTHESIS_DEFAULT_NFE, solver="midpoint")
            else:
                extra.update(nfe=8, solver="euler")
            return enhance(
                dwav=wav,
                sr=sr,
                device=device_,
                run_dir=run_dir,
                tau=AI_SYNTHESIS_DEFAULT_TAU,
                lambd=AI_SYNTHESIS_DEFAULT_LAMBD,
                **kwargs,
                **extra,
            )

        try:
            hwav, model_sr = _preview_run(device)
        except Exception:
            # Fallback to CPU or bypass if device fails
            try:
                hwav, model_sr = _preview_run('cpu')
            except Exception:
                hwav = wav
                model_sr = sr

        # Resample for profile (48k) if on
        dest_sr = 48000 if self.var_profile.get() else sr
        if model_sr != dest_sr:
            hwav = ta_resample(hwav, orig_freq=model_sr, new_freq=dest_sr)
        exp_len = round(wav.shape[-1] * (dest_sr / sr)) if dest_sr != sr else wav.shape[-1]
        if hwav.shape[-1] > exp_len:
            hwav = hwav[:exp_len]
        elif hwav.shape[-1] < exp_len:
            import torch
            hwav = torch.nn.functional.pad(hwav, (0, exp_len - hwav.shape[-1]))
        try:
            base = wav
            if dest_sr != sr:
                base = ta_resample(base, orig_freq=sr, new_freq=dest_sr)
        except Exception:
            base = wav
        try:
            if base.shape[-1] > exp_len:
                base = base[:exp_len]
            elif base.shape[-1] < exp_len:
                import torch as _t
                base = _t.nn.functional.pad(base, (0, exp_len - base.shape[-1]))
        except Exception:
            pass

        # Wet/dry mix
        try:
            wet = float(os.environ.get('RESEMBLE_WET', '')) if os.environ.get('RESEMBLE_WET') else float(self.var_wet.get())
        except Exception:
            wet = 1.0
        wet = max(0.0, min(1.0, wet))
        if wet < 1.0:
            hwav = wet * hwav + (1.0 - wet) * base
        if noise_only_flag:
            hwav = base - hwav

        # Peak ceiling
        hwav = _apply_peak_ceiling(hwav, ceiling_db=-1.0)

        # Save to preview file and return path
        out_dir = INPUT_TMP_ROOT / 'preview_out'
        out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / (Path(src_path).stem + f'_PREVIEW_{int(start_s)}s_{int(dur_s)}s.wav')
        # Ensure non-empty on save
        if hwav.numel() <= 0:
            import torch as _t
            hwav = _t.zeros(256, dtype=_t.float32)
        torchaudio.save(str(outp), hwav.unsqueeze(0), dest_sr)
        return str(outp)

    def _get_selected_or_first(self) -> str | None:
        try:
            sel = self.queue_tree.selection()
            if sel:
                iid = sel[0]
                p = self._iid_to_path.get(iid)
                if p:
                    return p
            if self.files:
                return self.files[0]
        except Exception:
            pass
        return None

    def _open_gradio_preview(self):
        try:
            if getattr(self, '_gradio_running', False):
                self._log('Gradio preview already running.')
                return
            import gradio as gr
            import numpy as np
            import soundfile as sf
            self._gradio_running = True

            def _proc(use_selection: bool, upload, start: float, dur: float):
                path = None
                if use_selection:
                    path = self._get_selected_or_first()
                if not path and upload is not None:
                    try:
                        path = upload.name
                    except Exception:
                        path = None
                if not path:
                    return None, 'Select a file in the queue or upload one.'
                try:
                    outp = self._render_preview_segment(str(path), float(start), float(dur))
                    y, sr = sf.read(outp, dtype='float32', always_2d=False)
                    if isinstance(y, np.ndarray) and y.ndim > 1:
                        y = y.mean(axis=1)
                    return (int(sr), y), f'Rendered preview: {Path(outp).name}'
                except Exception as e:
                    return None, f'Error: {e}'

            def _export_full(use_selection: bool, upload):
                path = None
                if use_selection:
                    path = self._get_selected_or_first()
                if not path and upload is not None:
                    try:
                        path = upload.name
                    except Exception:
                        path = None
                if not path:
                    return None, 'Select a file in the queue or upload one.'
                # Compute full duration
                dur = 0.0
                try:
                    info = sf.info(path)
                    dur = float(info.frames) / float(info.samplerate or 1)
                except Exception:
                    dur = 0.0
                if dur <= 0:
                    dur = 99999.0
                try:
                    outp = self._render_preview_segment(str(path), 0.0, float(dur))
                    return outp, f'Exported (Preview Mode): {Path(outp).name}'
                except Exception as e:
                    return None, f'Error: {e}'

            with gr.Blocks(title='Resemble Enhance Preview') as demo:
                gr.Markdown('## Preview')
                with gr.Row():
                    use_sel = gr.Checkbox(label='Use current selection', value=True)
                    upload = gr.File(label='Or upload audio (wav/mp3)')
                with gr.Row():
                    start = gr.Slider(0, 600, value=0, step=0.1, label='Start (s)')
                    dur = gr.Slider(0.5, 180, value=5, step=0.1, label='Duration (s)')
                btn = gr.Button('Render')
                audio = gr.Audio(label='Preview Audio')
                msg = gr.Markdown()
                btn.click(_proc, [use_sel, upload, start, dur], [audio, msg])
                gr.Markdown('---')
                with gr.Row():
                    btn_exp = gr.Button('Export Full (Preview Mode)')
                    dl = gr.File(label='Download', interactive=False)
                msg2 = gr.Markdown()
                btn_exp.click(_export_full, [use_sel, upload], [dl, msg2])

            # Launch from UI thread without blocking; let gradio open the browser
            # Launch without queue to avoid version-specific queue errors
            demo.launch(share=False, inbrowser=True, prevent_thread_lock=True)
            self._log('Gradio preview launched.')
        except Exception as e:
            try:
                self._log(f'Preview failed: {e}')
            except Exception:
                pass
        finally:
            self._gradio_running = False

    # Removed inline preview playback and export; using Gradio-based preview instead
    def _add_paths(self, paths):
        # Legacy ingestion path: disable client mode if explicitly adding raw files/folders.
        if paths:
            self.client_mode_active = False
            self.client_queue.clear()
            self.client_meta.clear()
            self.client_order.clear()
            self.client_camera_roles.clear()
            self.var_otio_client.set("")
            self.file_to_client.clear()
        def _is_audio(path: str) -> bool:
            suf = Path(path).suffix.lower()
            return suf in {'.wav', '.wave', '.mp3'}
        file_set = set(self.files)
        detected_existing = 0
        for p in paths:
            p = str(p)
            try:
                if Path(p).is_dir():
                    found_any = False
                    if self.var_recursive_folders.get():
                        # Collect audio files under all subfolders
                        for root, _dirs, files in os.walk(p):
                            for name in files:
                                if not _is_audio(name):
                                    continue
                                sfp = str(Path(root) / name)
                                if self._has_existing_clean_output(sfp):
                                    detected_existing += 1
                                if sfp not in file_set:
                                    file_set.add(sfp)
                                    self.files.append(sfp)
                                    found_any = True
                    else:
                        # Collect audio files directly under this folder (non-recursive)
                        for f in sorted(Path(p).iterdir()):
                            if f.is_file() and _is_audio(str(f)):
                                sfp = str(f)
                                if self._has_existing_clean_output(sfp):
                                    detected_existing += 1
                                if sfp not in file_set:
                                    file_set.add(sfp)
                                    self.files.append(sfp)
                                    found_any = True
                    if found_any:
                        self.folders.add(p)
                    continue
            except Exception:
                pass
            if _is_audio(p) and p not in self.files:
                if self._has_existing_clean_output(p):
                    detected_existing += 1
                self.files.append(p)
                self.file_status[p] = self.file_status.get(p, 'queued')
        if detected_existing:
            try:
                self._log(
                    f"Detected existing CLEAN outputs for {detected_existing} file(s). "
                    f"Enhance will be skipped for those files, but sync/OTIO can still run."
                )
            except Exception:
                pass
        self._refresh_queue_tree()
        self._refresh_otio_client_dropdown()

    def _enable_run(self):
        enabled = "normal" if self.files else "disabled"
        self.run_btn["state"] = enabled
        # Keep status filter useful
        if not self.files:
            self.status_filter_var.set('All')

    def _refresh_queue_tree(self):
        try:
            for iid in self.queue_tree.get_children(''):
                self.queue_tree.delete(iid)
        except Exception:
            return
        self._iid_to_path.clear()
        self._path_to_iid.clear()
        self._iid_to_folder.clear()
        self._folder_to_iid.clear()
        try:
            self.folders = {d for d in self.folders if any(str(Path(f).parent) == d for f in self.files)}
        except Exception:
            pass
        groups: dict[str, list[str]] = {}
        folder_order: list[str] = []
        if self.client_mode_active:
            for cid in self.client_order:
                groups.setdefault(cid, [])
            for fp in self.files:
                cid = self.file_to_client.get(fp, str(Path(fp).parent))
                groups.setdefault(cid, []).append(fp)
            seen = set()
            for cid in self.client_order:
                if cid not in seen:
                    seen.add(cid)
                    folder_order.append(cid)
            for cid in groups.keys():
                if cid not in seen:
                    seen.add(cid)
                    folder_order.append(cid)
        else:
            for fp in self.files:
                parent = str(Path(fp).parent)
                groups.setdefault(parent, []).append(fp)
            # Include explicitly added empty folders
            for d in self.folders:
                groups.setdefault(d, groups.get(d, []))
            # Determine folder order by first appearance in self.files, then any explicitly added folders
            seen = set()
            for f in self.files:
                parent = str(Path(f).parent)
                if parent not in seen:
                    seen.add(parent)
                    folder_order.append(parent)
            for d in self.folders:
                if d not in seen:
                    seen.add(d)
                    folder_order.append(d)

        gidx = 0
        for folder in folder_order:
            if self.client_mode_active and folder in self.client_meta:
                meta = self.client_meta.get(folder, {})
                cname = Path(str(meta.get("client_root", folder))).name
                reason = str(meta.get("skip_reason") or "").strip()
                if reason and not groups.get(folder):
                    title = f"{cname} (skipped: {reason})"
                else:
                    title = f"{cname}"
            else:
                title = folder
            node = self.queue_tree.insert('', 'end', text=title, open=True)
            self._iid_to_folder[node] = folder
            self._folder_to_iid[folder] = node
            gtag = 'g_odd' if (gidx % 2 == 0) else 'g_even'
            try:
                self.queue_tree.item(node, tags=(gtag,))
            except Exception:
                pass
            gidx += 1
            children = groups[folder]
            if not children:
                self.queue_tree.insert(node, 'end', text='(no queued files)')
            else:
                # Preserve ordering as in self.files; apply status filter
                want = self.status_filter_var.get()
                for f in children:
                    st = self.file_status.get(f, 'queued')
                    if want != 'All' and st.lower() != want.lower():
                        continue
                    iid = self.queue_tree.insert(node, 'end', text=Path(f).name, tags=(gtag,))
                    self._iid_to_path[iid] = f
                    self._path_to_iid[f] = iid
                    # Visual tags by status
                    try:
                        tag = st.lower()
                        # retain group shading and add status tag
                        self.queue_tree.item(iid, tags=(gtag, tag))
                    except Exception:
                        pass
        # Tag styles
        try:
            # Brighter default for better readability on dark background
            self.queue_tree.tag_configure('queued', foreground=self._text)
            self.queue_tree.tag_configure('running', foreground=self._accent)
            self.queue_tree.tag_configure('done', foreground='#21c55d')
            self.queue_tree.tag_configure('failed', foreground='#ef4444')
            # Alternating group shading
            self.queue_tree.tag_configure('g_odd', background='#1b1b1b')
            self.queue_tree.tag_configure('g_even', background='#202020')
        except Exception:
            pass

    def _on_drop(self, event):
        # tkinterdnd2 provides a space-separated list; paths with spaces are braced
        data = event.data
        items = []
        buf = ''
        in_brace = False
        for ch in data:
            if ch == '{':
                in_brace = True
                buf = ''
                continue
            if ch == '}':
                in_brace = False
                items.append(buf)
                buf = ''
                continue
            if ch == ' ' and not in_brace:
                if buf:
                    items.append(buf)
                    buf = ''
                continue
            buf += ch
        if buf:
            items.append(buf)
        self._add_paths(items)
        self._enable_run()

    def _toggle_advanced(self):
        if not getattr(self, '_adv_wrap', None):
            return
        try:
            if self._adv_open:
                self._adv_wrap.pack_forget()
                self._adv_open = False
                try:
                    if self._adv_btn:
                        self._adv_btn.configure(text='Processing & OTIO Options [+]')
                except Exception:
                    pass
            else:
                pack_kwargs = {"fill": "x", "pady": (2, 8)}
                if getattr(self, "_adv_hdr", None) is not None:
                    self._adv_wrap.pack(after=self._adv_hdr, **pack_kwargs)
                else:
                    self._adv_wrap.pack(**pack_kwargs)
                self._adv_open = True
                try:
                    if self._adv_btn:
                        self._adv_btn.configure(text='Processing & OTIO Options [-]')
                except Exception:
                    pass
        except Exception:
            pass

    def _log(self, msg, color: str | None = None):
        text = str(msg)
        try:
            print(text, flush=True)
        except Exception:
            pass
        try:
            self.status_label.configure(text=text, foreground=(color or self._text))
        except Exception:
            pass

    def _log_clear(self):
        try:
            self.status_label.configure(text='', foreground=self._text)
        except Exception:
            pass

    def _get_media_clean_dir(self, src_path: str | Path) -> Path | None:
        try:
            return _find_media_clean_dir(Path(src_path))
        except Exception:
            return None

    def _has_existing_clean_output(self, src_path: str | Path) -> bool:
        try:
            return _has_clean_output_in_media(Path(src_path))
        except Exception:
            return False

    def _get_output_override(self) -> str | None:
        try:
            if bool(self.var_output_media_clean.get()):
                return None
            val = (self.var_output_dir.get() or "").strip()
            return val if val else None
        except Exception:
            return None

    def _on_ai_synthesis_toggle(self) -> None:
        enabled = False
        try:
            enabled = bool(self.var_ai_synthesis.get())
        except Exception:
            enabled = False
        try:
            self.var_denoise_only.set(not enabled)
        except Exception:
            pass
        if enabled:
            try:
                self.var_noise_only.set(False)
            except Exception:
                pass
            try:
                self.var_wet.set(AI_SYNTHESIS_DEFAULT_WET)
            except Exception:
                pass
            try:
                self._log("AI enhancement / synthesis enabled with conservative blend to reduce robotic artifacts.")
            except Exception:
                pass
        else:
            try:
                self.var_wet.set(1.0)
            except Exception:
                pass
            try:
                self._log("Denoise-only mode enabled.")
            except Exception:
                pass

    def _snapshot_run_config(self, chunk_seconds: float, overlap_seconds: float) -> dict:
        try:
            files = [str(p) for p in getattr(self, 'files', [])]
        except Exception:
            files = []
        diag = bool(self.var_diag_minimal.get())
        aggressive_req = bool(self.var_aggressive_denoise.get())
        aggressive_effective = bool(aggressive_req or diag)
        skip_req = bool(self.var_skip_fine.get())
        snapshot = {
            "diagnostics_mode": diag,
            "denoise_only_mode": bool(self.var_denoise_only.get()),
            "ai_synthesis_enabled": not bool(self.var_denoise_only.get()),
            "device": str(self.var_device.get()),
            "profile_camera_sync": bool(self.var_profile.get()),
            "sync_inputs": bool(self.var_sync_export.get()),
            "seam_safe": bool(self.var_seam_safe.get()),
            "post_process": bool(self.var_postproc.get()),
            "bw64_export": bool(self.var_bw64.get()),
            "batch_by_folder": bool(self.var_batch_folders.get()),
            "lead_guard": bool(self.var_lead_guard.get()),
            "noise_only_output": bool(self.var_noise_only.get()),
            "output_override_dir": self._get_output_override() or "",
            "output_media_clean": bool(self.var_output_media_clean.get()),
            "recursive_folder_search": bool(self.var_recursive_folders.get()),
            "skip_fine_requested": skip_req,
            "skip_fine_effective": bool(skip_req and not diag),
            "force_fine_align": bool(diag),
            "force_drift_correction": bool(diag),
            "aggressive_denoise_requested": aggressive_req,
            "aggressive_denoise_effective": aggressive_effective,
            "denoise_mix_wet": float(self.var_wet.get()),
            "chunk_seconds_effective": float(chunk_seconds),
            "overlap_seconds_effective": float(overlap_seconds),
            "generate_otio": bool(self.var_generate_otio.get()),
            "otio_wide": str(self._otio_label_to_path.get(str(self.var_otio_wide.get() or "").strip(), "")),
            "otio_guest_closeup": str(self._otio_label_to_path.get(str(self.var_otio_guest_closeup.get() or "").strip(), "")),
            "otio_host_closeup": str(self._otio_label_to_path.get(str(self.var_otio_host_closeup.get() or "").strip(), "")),
            "otio_extras": str(self.var_otio_extras.get() or ""),
            "otio_timeline_name": str(self.var_otio_timeline_name.get() or ""),
            "files_enqueued": files,
            "file_count": len(files),
            "ffmpeg_path": shutil.which('ffmpeg') or shutil.which('ffmpeg.exe') or '',
            "enhancer_run_dir_override": str(_get_enhancer_run_dir() or ""),
            "run_status": "pending",
        }
        return snapshot

    def _write_run_flag_log(self, snapshot: dict | None) -> None:
        try:
            log_dir = INPUT_TMP_ROOT / "run_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            env_flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith('RESEMBLE_')}
            deps: dict[str, str] = {}
            for name in ('torch', 'torchaudio', 'audalign', 'soundfile'):
                try:
                    module = sys.modules.get(name)
                    if module is None:
                        module = __import__(name)
                    deps[name] = getattr(module, '__version__', 'unknown')
                except Exception as exc:  # noqa: BLE001
                    deps[name] = f"missing ({exc})"
            ffmpeg = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
            deps['ffmpeg'] = ffmpeg or 'not found'
            payload = {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "settings": snapshot or {},
                "env_flags": env_flags,
                "dependencies": deps,
            }
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"run_flags_{stamp}.json"
            log_path.write_text(json.dumps(payload, indent=2))
            status = (snapshot or {}).get("run_status")
            if status == "completed":
                self.after(0, lambda: self._log("Run completed successfully.", color=self._success))
        except Exception as exc:  # noqa: BLE001
            try:
                self.after(0, lambda exc=exc: self._log(f"Run log error: {exc}"))
            except Exception:
                pass

    def _set_status(self, text: str):
        try:
            self.status_label.configure(text=text, foreground=self._text)
        except Exception:
            pass

    def _dz_hover(self, on: bool):
        try:
            self.drop_zone.configure(bg=self._accent if on else self._panel, fg=self._text if on else self._accent)
        except Exception:
            pass

    def _clear_queue(self, reset_progress: bool = True):
        self.files.clear()
        self.client_mode_active = False
        self.client_queue.clear()
        self.client_meta.clear()
        self.client_order.clear()
        self.client_camera_roles.clear()
        self.var_otio_client.set("")
        self.file_to_client.clear()
        try:
            self.folders.clear()
        except Exception:
            self.folders = set()
        self.file_status.clear()
        self._refresh_queue_tree()
        self._enable_run()
        if reset_progress:
            self.progress["value"] = 0
            self.overall_label["text"] = '0 of 0 files'
            self._set_status('')

    def _append_history(self, results: list[tuple[str,str]]):
        # results: list of (src, out)
        for src, out in results:
            self.history.append((src, out))
            self.hist.insert('', 'end', values=(src, out))

    def _set_file_status(self, path: str, status: str):
        self.file_status[path] = status
        iid = self._path_to_iid.get(path)
        if iid:
            try:
                self.queue_tree.item(iid, tags=(status.lower(),))
            except Exception:
                pass
        # If filtering hides this item after status change, refresh
        if self.status_filter_var.get() != 'All':
            self._refresh_queue_tree()

    # --- Queue actions ---
    def _queue_key_remove(self, event=None):
        try:
            focus = self.focus_get()
        except Exception:
            focus = None
        if focus is not self.queue_tree:
            return None
        self._remove_selected()
        return 'break'

    def _selected_paths(self) -> list[str]:
        iids = list(self.queue_tree.selection())
        if not iids:
            focused = self.queue_tree.focus()
            if focused:
                iids = [focused]
        out: list[str] = []
        seen: set[str] = set()
        for iid in iids:
            p = self._iid_to_path.get(iid)
            if p:
                if p not in seen:
                    out.append(p)
                    seen.add(p)
                continue
            folder = self._iid_to_folder.get(iid)
            if folder:
                if self.client_mode_active:
                    for f in self.files:
                        if self.file_to_client.get(f) == folder and f not in seen:
                            out.append(f)
                            seen.add(f)
                    for f in self.client_queue.get(folder, []):
                        if f in self.files and f not in seen:
                            out.append(f)
                            seen.add(f)
                    continue
                for f in self.files:
                    if str(Path(f).parent) == folder and f not in seen:
                        out.append(f)
                        seen.add(f)
        return out

    def _remove_selected(self):
        sel = set(self._selected_paths())
        if not sel:
            return
        self.files = [f for f in self.files if f not in sel]
        for f in sel:
            self.file_to_client.pop(f, None)
        for f in sel:
            self.file_status.pop(f, None)
        if self.client_mode_active:
            rebuilt: dict[str, list[str]] = {}
            for f in self.files:
                cid = self.file_to_client.get(f)
                if cid:
                    rebuilt.setdefault(cid, []).append(f)
            self.client_queue = rebuilt
            self.client_order = [cid for cid in self.client_order if cid in rebuilt]
            self.client_meta = {cid: self.client_meta.get(cid, {"client_root": cid, "video_files": [], "skip_reason": None}) for cid in self.client_order}
        self._refresh_queue_tree()
        self._refresh_otio_client_dropdown()
        self._enable_run()

    def _move_selection(self, direction: int):
        # direction: -1 up, +1 down
        sel = self._selected_paths()
        if not sel:
            return
        idxs = [i for i, f in enumerate(self.files) if f in sel]
        if direction < 0:
            for i in range(1, len(self.files)):
                if i in idxs and (i-1) not in idxs:
                    self.files[i-1], self.files[i] = self.files[i], self.files[i-1]
        else:
            for i in range(len(self.files)-2, -1, -1):
                if i in idxs and (i+1) not in idxs:
                    self.files[i+1], self.files[i] = self.files[i], self.files[i+1]
        self._refresh_queue_tree()
        # Restore selection
        for f in sel:
            iid = self._path_to_iid.get(f)
            if iid:
                self.queue_tree.selection_add(iid)

    def _clear_processed(self):
        self.files = [f for f in self.files if self.file_status.get(f) != 'done']
        if self.client_mode_active:
            self.file_to_client = {f: c for f, c in self.file_to_client.items() if f in set(self.files)}
            rebuilt: dict[str, list[str]] = {}
            for f in self.files:
                cid = self.file_to_client.get(f)
                if cid:
                    rebuilt.setdefault(cid, []).append(f)
            self.client_queue = rebuilt
            self.client_order = [cid for cid in self.client_order if cid in rebuilt]
            self.client_meta = {cid: self.client_meta.get(cid, {"client_root": cid, "video_files": [], "skip_reason": None}) for cid in self.client_order}
        self._refresh_queue_tree()
        self._enable_run()

    def _open_selected_output(self):
        path = self._selected_history_output_path()
        if not path:
            return
        try:
            if os.name == 'nt':
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['open', path])
        except Exception as e:  # noqa: BLE001
            self._log(f"Open failed: {e}")

    def _selected_history_output_path(self) -> str | None:
        sel = self.hist.selection()
        if not sel:
            return None
        item = self.hist.item(sel[0])
        vals = item.get('values') or []
        if len(vals) < 2:
            return None
        path = str(vals[1]).strip()
        return path or None

    def _reveal_selected_output(self):
        path = self._selected_history_output_path()
        if not path:
            return
        try:
            target = Path(path)
            if os.name == 'nt':
                subprocess.Popen(['explorer', '/select,', str(target)])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-R', str(target)])
            else:
                subprocess.Popen(['xdg-open', str(target.parent if target.parent.exists() else target)])
        except Exception as e:  # noqa: BLE001
            self._log(f"Reveal failed: {e}")

    def run_task(self):
        if not self.files:
            return
        if not self._confirm_queue_before_run():
            self._log("[queue] run cancelled at confirmation step.")
            return
        if not self.files:
            self._log("[queue] no files remain after confirmation.")
            return

        # Initialize control flags and UI
        self._control = _Control()
        self._group_done = 0
        self._group_total = 0
        self._group_start_time = None
        self._job_done = 0
        self._job_total = max(len(self.files), 1)
        self._job_start_time = time.time()
        self._chunk_last = {}
        self._chunk_active = False
        self._last_progress_pct = 1.0
        self.run_btn["state"] = "disabled"
        self.pause_btn.config(state='normal', text='Pause')
        self.cancel_btn.config(state='normal')
        self.progress["maximum"] = 100
        self.progress["value"] = 1
        self.overall_label["text"] = f"Starting enhancement job: 0 of {self._job_total} files"

        def set_progress_if_current(pct, label=None, *, allow_decrease=False, update_label_on_decrease=False):
            pct = max(0.0, min(100.0, float(pct)))
            try:
                current = float(self.progress["value"] or 0)
            except Exception:
                current = 0.0
            if not allow_decrease and pct + 0.001 < current:
                if update_label_on_decrease and label is not None:
                    self.overall_label["text"] = label
                return False
            self.progress["maximum"] = 100
            self.progress["value"] = pct
            self._last_progress_pct = pct
            if label is not None:
                self.overall_label["text"] = label
            return True

        def update_prog(done, total):
            total = max(total, 1)
            job_total = int(getattr(self, '_job_total', 0) or total or 1)
            job_done = min(job_total, int(getattr(self, '_job_done', 0) or 0) + int(done or 0))
            pct = job_done * 100.0 / job_total
            set_progress_if_current(pct, f"Job: {job_done} of {job_total} files ({int(pct)}%)")

        def update_chunk(name, i, n):
            n = max(n or 0, 1)
            stage_detail = ""
            raw_name = name or ""
            if "|" in raw_name:
                raw_name, stage_detail = raw_name.split("|", 1)
                stage_detail = stage_detail.strip()
            if stage_detail:
                try:
                    if not hasattr(self, "_chunk_last"):
                        self._chunk_last = {}
                    last_i, last_n = self._chunk_last.get(raw_name, (0, n))
                    i = max(int(i or 0), int(last_i or 0))
                    n = max(int(n or 0), int(last_n or 0), 1)
                except Exception:
                    pass
            else:
                try:
                    if not hasattr(self, "_chunk_last"):
                        self._chunk_last = {}
                    self._chunk_last[raw_name] = (int(i or 0), int(n or 0))
                except Exception:
                    pass
            pct = int((i * 100) / n)
            base = Path(raw_name).name if raw_name else "-"
            self._chunk_active = True
            # Start time and ETA
            now = time.time()
            if i == 0:
                self.cur_start_time = now
            eta_txt = ''
            if self.cur_start_time is not None and i > 0:
                elapsed = now - self.cur_start_time
                try:
                    est_total = elapsed * (n / i)
                    eta = max(0.0, est_total - elapsed)
                    mm = int(eta // 60)
                    ss = int(eta % 60)
                    eta_txt = f" | ETA {mm:02d}:{ss:02d}"
                except Exception:
                    eta_txt = ''
            # Update per-item status
            try:
                if raw_name in self.file_status:
                    if i >= n:
                        self._set_file_status(raw_name, 'done')
                    else:
                        self._set_file_status(raw_name, 'running')
            except Exception:
                pass
            # Smooth overall: include current file fraction + ETA + throughput
            try:
                job_done_base = int(getattr(self, '_job_done', 0) or 0)
                group_done = int(getattr(self, '_group_done', 0) or 0)
                done_files = job_done_base + group_done
                total_files = int(getattr(self, '_job_total', 0) or 0) or int(getattr(self, '_group_total', 0) or 0) or 1
                frac = min(1.0, max(0.0, (i or 0) / float(n)))
                overall = (done_files + frac) * 100.0 / total_files
                jst = getattr(self, '_job_start_time', None)
                label = f"Job file {min(done_files+1, total_files)}/{total_files}: {base} - {pct}%{eta_txt}"
                if stage_detail:
                    label += f" | {stage_detail}"
                if jst:
                    gelapsed = max(0.001, now - jst)
                    units = done_files + frac
                    rate = units / gelapsed
                    eta_total = max(0.0, (total_files - units) / max(rate, 1e-9))
                    gmm = int(eta_total // 60)
                    gss = int(eta_total % 60)
                    fpm = rate * 60.0
                    label += f" | Job ETA {gmm:02d}:{gss:02d} | {fpm:.2f} files/min"
                set_progress_if_current(overall, label, update_label_on_decrease=True)
            except Exception:
                pass

        def worker():
            run_snapshot = None
            staging_root = None
            processed_groups = 0
            run_t0 = time.perf_counter()
            try:
                # Clean up temp audio/log artifacts from prior runs
                cleanup_pre_t0 = time.perf_counter()
                _cleanup_run_artifacts(remove_logs=True)
                cleanup_pre_dt = time.perf_counter() - cleanup_pre_t0
                self.after(0, lambda dt=cleanup_pre_dt: self._log(f"Timing: pre-run cleanup {_format_seconds(dt)}"))
                # Reset status
                self.after(0, lambda: self._set_status('Ready'))
                self.after(0, lambda: self._log("Launching enhancer..."))
                do_sync = True if self._force_sync_export else bool(self.var_sync_export.get())
                if not do_sync:
                    self.after(0, lambda: self._log("Cleanup-only mode: skipping sync/export; writing one output per file."))
                media_clean = bool(self.var_output_media_clean.get())
                # Export run-time env so processing respects GUI settings
                try:
                    cs = max(1.0, float(self.var_chunk_sec.get()))
                    ov = max(0.0, float(self.var_overlap_sec.get()))
                    # In diagnostics mode, force small, fast chunks for quick feedback
                    if self.var_diag_minimal.get():
                        cs = 7.0
                        ov = 0.5
                    os.environ['RESEMBLE_CHUNK_SECONDS'] = str(cs)
                    os.environ['RESEMBLE_OVERLAP_SECONDS'] = str(ov)
                    try:
                        os.environ['RESEMBLE_WET'] = str(max(0.0, min(1.0, float(self.var_wet.get()))))
                    except Exception:
                        pass
                    os.environ['RESEMBLE_NOISE_ONLY'] = '1' if self.var_noise_only.get() else '0'
                    os.environ[OUTPUT_MEDIA_CLEAN_ENV] = '1' if media_clean else '0'
                    os.environ['RESEMBLE_LEAD_GUARD'] = '1' if self.var_lead_guard.get() else '0'
                    aggr = self.var_aggressive_denoise.get() or self.var_diag_minimal.get()
                    os.environ['RESEMBLE_DENOISE_AGGRESSIVE'] = '1' if aggr else '0'
                    run_snapshot = self._snapshot_run_config(cs, ov)
                    run_snapshot["run_status"] = "running"
                    # Prefer fast enhance config when Enhance mode is selected
                    if not self.var_denoise_only.get():
                        os.environ['RESEMBLE_FAST_ENHANCE'] = '1'
                    else:
                        os.environ.pop('RESEMBLE_FAST_ENHANCE', None)
                    # Diagnostics: disable all post FX besides enhance + alignment
                    if self.var_diag_minimal.get():
                        os.environ['RESEMBLE_DISABLE_TRANSIENT_BLEND'] = '1'
                        os.environ['RESEMBLE_LEAD_GUARD'] = '0'
                        if self.var_denoise_only.get():
                            os.environ['RESEMBLE_WET'] = '1.0'
                except Exception:
                    pass
                output_override = self._get_output_override()
                if output_override:
                    try:
                        Path(output_override).mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        self.after(0, lambda exc=exc: self._log(f"Output folder error: {exc}"))
                        output_override = None
                if do_sync:
                    try:
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        staging_root = INPUT_TMP_ROOT / "staging" / f"{stamp}_{uuid.uuid4().hex[:6]}"
                        staging_root.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        self.after(0, lambda exc=exc: self._log(f"Staging folder error: {exc}"))
                        staging_root = None
                # Build groups: always batch by folder when recursive search is enabled
                files_all = list(self.files)
                # Detect reusable CLEAN outputs so we can skip enhance but still run sync/OTIO.
                existing_clean_map: dict[str, str] = {}
                if files_all and self._reuse_existing_outputs:
                    reused = 0
                    for fp in files_all:
                        p = _find_existing_clean_for_source(Path(fp))
                        if p is not None:
                            existing_clean_map[str(fp)] = str(p)
                            reused += 1
                    if reused:
                        try:
                            sample_map = list(existing_clean_map.items())[:4]
                            for src, clean in sample_map:
                                self.after(0, lambda s=Path(src).name, c=Path(clean).name: self._log(f"Reuse: {s} -> {c}"))
                        except Exception:
                            pass
                        self.after(
                            0,
                            lambda n=reused: self._log(
                                f"Reusing existing CLEAN output for {n} file(s); skipping enhance for those while keeping sync/OTIO enabled."
                            ),
                        )
                    else:
                        self.after(
                            0,
                            lambda: self._log(
                                "No reusable CLEAN outputs detected for current selection; running enhance on queued files."
                            ),
                        )
                elif files_all and (not self._reuse_existing_outputs):
                    self.after(0, lambda: self._log("Packaged mode: always reprocessing inputs (existing CLEAN outputs are ignored)."))
                groups: list[tuple[str, list[str]]] = []
                batch_by_folder = bool(self.var_batch_folders.get()) or bool(self.var_recursive_folders.get())
                if batch_by_folder:
                    by_parent: dict[str, list[str]] = {}
                    for fp in files_all:
                        parent = str(Path(fp).parent)
                        by_parent.setdefault(parent, []).append(fp)
                    groups = sorted(by_parent.items(), key=lambda kv: kv[0].lower())
                else:
                    groups = [("(all)", files_all)]

                total_groups = len(groups)
                self._job_done = 0
                self._job_total = max(sum(len(gfiles) for _, gfiles in groups), 1)
                self._job_start_time = time.time()
                self.after(
                    0,
                    lambda total=self._job_total: set_progress_if_current(
                        1.0,
                        f"Starting enhancement job: 0 of {total} files",
                    ),
                )
                for gi, (gname, gfiles) in enumerate(groups, start=1):
                    if not gfiles:
                        continue
                    group_t0 = time.perf_counter()
                    group_media_clean_dir = None
                    if media_clean:
                        group_media_clean_dir = self._get_media_clean_dir(gfiles[0])
                        if group_media_clean_dir is not None:
                            try:
                                group_media_clean_dir.mkdir(parents=True, exist_ok=True)
                            except Exception as exc:
                                self.after(0, lambda exc=exc: self._log(f"Output folder error: {exc}"))
                                group_media_clean_dir = None
                    group_output_dir = output_override
                    group_do_sync = bool(do_sync and len(gfiles) > 1)
                    if group_do_sync:
                        if staging_root is not None:
                            group_output_dir = str(staging_root / f"group_{gi}")
                            try:
                                Path(group_output_dir).mkdir(parents=True, exist_ok=True)
                            except Exception:
                                group_output_dir = None
                        else:
                            group_output_dir = None
                        if group_output_dir is None:
                            self.after(0, lambda: self._log("Staging failed; outputs will not be written to output folder."))
                    self._group_done = 0
                    self._group_total = len(gfiles)
                    self._group_start_time = time.time()
                    self.after(0, lambda gname=gname, gi=gi, total_groups=total_groups: self._log(f"Processing group {gi}/{total_groups}: {gname} ({len(gfiles)} files)"))
                    # group-specific progress wrapper
                    def update_prog_group(done, total, gi=gi, total_groups=total_groups):
                        total = max(total, 1)
                        self._group_done = done
                        self._group_total = total
                        job_total = int(getattr(self, '_job_total', 0) or total or 1)
                        job_done = min(job_total, int(getattr(self, '_job_done', 0) or 0) + int(done or 0))
                        pct = job_done * 100.0 / job_total
                        label = f"Job: {job_done} of {job_total} files ({int(pct)}%) | Group {gi}/{total_groups}: {done} of {total}"
                        set_progress_if_current(pct, label)

                    def finish_group_progress(gi=gi, total_groups=total_groups, group_count=len(gfiles)):
                        self._chunk_active = False
                        self._chunk_last = {}
                        self._group_done = group_count
                        self._group_total = max(group_count, 1)
                        self._job_done = min(int(getattr(self, '_job_total', group_count) or group_count), int(getattr(self, '_job_done', 0) or 0) + group_count)
                        job_total = int(getattr(self, '_job_total', 0) or group_count or 1)
                        pct = self._job_done * 100.0 / job_total
                        label = f"Job: {self._job_done} of {job_total} files ({int(pct)}%) | Group {gi}/{total_groups} complete"
                        self.after(0, lambda pct=pct, label=label: set_progress_if_current(pct, label))

                    use_files = list(gfiles)
                    # Group-level fallback: if a synced CLEAN MOV exists, map all files to it
                    # so we can skip enhance/sync and still generate OTIO.
                    try:
                        if len(gfiles) >= 2:
                            mapped_now = sum(1 for f in gfiles if f in existing_clean_map)
                            if mapped_now < len(gfiles):
                                grp_clean = _find_reusable_synced_clean_for_group(gfiles)
                                if grp_clean is not None:
                                    added = 0
                                    grp_clean_s = str(grp_clean)
                                    for f in gfiles:
                                        if f not in existing_clean_map:
                                            existing_clean_map[f] = grp_clean_s
                                            added += 1
                                    if added > 0:
                                        self.after(
                                            0,
                                            lambda gi=gi, a=added, p=Path(grp_clean_s).name: self._log(
                                                f"Group {gi}: mapped {a} file(s) to reusable synced CLEAN source {p}."
                                            ),
                                        )
                    except Exception:
                        pass
                    existing_results = [(src, existing_clean_map[src]) for src in gfiles if src in existing_clean_map]
                    use_files = [f for f in use_files if f not in existing_clean_map]
                    self.after(
                        0,
                        lambda gi=gi, total=len(gfiles), reused=len(existing_results), pending=len(use_files): self._log(
                            f"Group {gi}: queue summary total={total}, reusable_clean={reused}, to_enhance={pending}"
                        ),
                    )

                    # Enhance this group
                    prefer_cli = (self.var_diag_minimal.get() or (not self.var_denoise_only.get()))
                    if self.var_noise_only.get():
                        prefer_cli = False
                    reduce_gpu = bool(self.var_reduce_gpu.get()) and str(self.var_device.get()).lower() == "cuda"
                    if reduce_gpu:
                        prefer_cli = False
                    enhance_t0 = time.perf_counter()
                    results: list[tuple[str, str]] = []
                    if use_files:
                        results = run_enhancer_for(
                            use_files,
                            device=self.var_device.get(),
                            profile=self.var_profile.get(),
                            progress_cb=lambda d, t: self.after(0, update_prog_group, d, t),
                            chunk_progress_cb=lambda name, i, n: self.after(0, update_chunk, name, i, n),
                            seam_safe=self.var_seam_safe.get(),
                            control=self._control,
                            denoise_only=self.var_denoise_only.get(),
                            prefer_cli=prefer_cli,
                            noise_only=self.var_noise_only.get(),
                            output_dir=group_output_dir,
                            force_inprocess=reduce_gpu,
                        )
                    if existing_results:
                        results.extend(existing_results)
                    enhance_dt = time.perf_counter() - enhance_t0
                    self.after(0, lambda gi=gi, dt=enhance_dt: self._log(f"Timing: group {gi} enhance {_format_seconds(dt)}"))
                    if existing_results and not use_files:
                        self.after(0, lambda gi=gi, n=len(existing_results): self._log(f"Group {gi}: skipped enhance for {n} file(s), using existing CLEAN files."))
                        # If every input in this group already has CLEAN output, skip downstream
                        # post-process/sync. Still try OTIO generation from existing clean files.
                        try:
                            done_set_existing = {src for (src, _out) in existing_results}
                            for f in gfiles:
                                if f in done_set_existing:
                                    self.after(0, self._set_file_status, f, 'done')
                            self.after(0, lambda results=existing_results: self._append_history(results))
                            write_otio = bool(self.var_generate_otio.get())
                            group_client_id = None
                            try:
                                if gfiles:
                                    group_client_id = self.file_to_client.get(gfiles[0])
                            except Exception:
                                group_client_id = None
                            otio_roles = self._build_otio_camera_roles(client_id=group_client_id) if write_otio else None
                            otio_name = (self.var_otio_timeline_name.get() or "").strip() if write_otio else ""
                            otio_out_dir = None
                            if write_otio and group_client_id:
                                try:
                                    croot = str(self.client_meta.get(group_client_id, {}).get("client_root", "")).strip()
                                    if croot:
                                        otio_out_dir = str(Path(croot) / "02_EDIT")
                                except Exception:
                                    otio_out_dir = None
                            if write_otio and (not otio_roles or not str(otio_roles.get("wide", "")).strip()):
                                self.after(0, lambda: self._log("[otio] Wide camera is required. Skipping OTIO for this all-clean group."))
                                write_otio = False
                            if write_otio:
                                try:
                                    target_sr = 48000
                                    clean_src = _find_synced_clean_for_group(gfiles)
                                    if clean_src is None:
                                        self.after(0, lambda: self._log("[otio] clean synced multichannel source not found; skipping OTIO."))
                                    else:
                                        switch_monos = _load_audio_tracks_any(clean_src, target_sr=target_sr)
                                        max_len = max([0] + [int(m.size(-1)) for m in switch_monos]) if switch_monos else 0
                                        aligned_paths = [Path(p) for p in gfiles]
                                        if len(switch_monos) >= 2 and max_len > 0:
                                            out_dir_for_otio = Path(otio_out_dir) if otio_out_dir else (group_media_clean_dir if group_media_clean_dir else Path(gname))
                                            out_dir_for_otio.mkdir(parents=True, exist_ok=True)
                                            stamp_ot = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                                            otio_path, turns_path = _write_active_speaker_otio(
                                                aligned_paths=aligned_paths,
                                                switch_monos=switch_monos,
                                                target_sr=target_sr,
                                                total_samples=max_len,
                                                out_dir=out_dir_for_otio,
                                                stamp=stamp_ot,
                                                camera_roles=otio_roles,
                                                clean_audio_path=str(clean_src),
                                                timeline_name=(otio_name or None),
                                                asr_required=False,
                                                alignment_strategy="cam_scratch_only",
                                                log=lambda m: self.after(0, self._log, m),
                                            )
                                            if otio_path:
                                                self.after(0, lambda p=otio_path: self._append_history([(gname, p)]))
                                                self.after(0, lambda p=otio_path: self._log(f"[otio] written (all-clean group): {p}"))
                                            if turns_path:
                                                self.after(0, lambda p=turns_path: self._log(f"[otio] turns diagnostics: {p}"))
                                        else:
                                            self.after(0, lambda: self._log("[otio] skipped for all-clean group: need >=2 clean channels for switching."))
                                except Exception as exc:
                                    self.after(0, lambda exc=exc: self._log(f"[otio] all-clean generation error: {exc}"))
                            self.after(0, lambda gi=gi: self._log(f"Group {gi}: all inputs already CLEAN; skipped post-process and sync."))
                        except Exception as exc:
                            self.after(0, lambda gi=gi, exc=exc: self._log(f"Group {gi}: all-clean skip path error: {exc}"))
                        processed_groups = gi
                        finish_group_progress()
                        if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                            break
                        group_dt = time.perf_counter() - group_t0
                        self.after(0, lambda gi=gi, dt=group_dt: self._log(f"Timing: group {gi} total {_format_seconds(dt)}"))
                        continue
                    try:
                        if LAST_MODEL_SR is not None:
                            msg = f"Model SR: {LAST_MODEL_SR} Hz"
                            if LAST_MODEL_SR_PATH:
                                msg += f" ({Path(LAST_MODEL_SR_PATH).name})"
                            self.after(0, lambda m=msg: self._log(m))
                    except Exception:
                        pass
                    self.after(0, lambda gi=gi: self._log(f"Group {gi}: enhanced {len(results)} file(s)."))
                    # Mark final statuses for this group
                    try:
                        done_set = {src for (src, _out) in results}
                        cancelled = self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set()
                        for f in gfiles:
                            if f in done_set:
                                self.after(0, self._set_file_status, f, 'done')
                            elif not cancelled:
                                self.after(0, self._set_file_status, f, 'failed')
                    except Exception:
                        pass
                    processed_groups = gi
                    if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                        break
                    if not group_do_sync:
                        self.after(0, lambda results=results: self._append_history(results))

                    # Post-processing per group
                    if self.var_postproc.get() and results:
                        post_t0 = time.perf_counter()
                        try:
                            self.after(0, lambda: self._set_status('Preparing post-process'))
                            outs = [out for _, out in results]
                            self.after(0, lambda: self._log("Applying level normalization and brightening..."))
                            def pp_prog(i, n, msg):
                                n = max(1, n)
                                pct = int(i * 100 / n)
                                self._set_status(f"{msg} - {pct}%")
                                if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                                    raise _Cancelled()
                            _postprocess_level_shape(outs, progress_cb=lambda i, n, m: self.after(0, pp_prog, i, n, m))
                            self.after(0, lambda: self._log("Level + brighten applied."))
                        except Exception as e:
                            if not isinstance(e, _Cancelled):
                                self.after(0, lambda e=e: self._log(f"Post-process error: {e}"))
                        finally:
                            post_dt = time.perf_counter() - post_t0
                            self.after(0, lambda gi=gi, dt=post_dt: self._log(f"Timing: group {gi} post-process {_format_seconds(dt)}"))
                        if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                            break

                    # Sync + export per group
                    if group_do_sync and results:
                        sync_t0 = time.perf_counter()
                        try:
                            self.after(0, lambda: self._set_status('Preparing sync'))
                            self.after(0, lambda: self._log("Syncing with Audalign and exporting multichannel..."))
                            self.after(0, lambda: self._log(f"Group {gi}: auto mic bleed enabled (tracks={len(gfiles)})."))
                            outs = [out for _, out in results]
                            # Deduplicate outputs in case an upstream retry produced duplicates
                            seen_paths = set()
                            unique_outs = []
                            for out in outs:
                                if out in seen_paths:
                                    continue
                                seen_paths.add(out)
                                unique_outs.append(out)
                            outs = unique_outs
                            def stage_prog(i, n, msg):
                                n = max(1, n)
                                pct = int(i * 100 / n)
                                self._set_status(f"{msg} - {pct}%")
                                if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                                    raise _Cancelled()
                            write_otio = bool(self.var_generate_otio.get())
                            group_client_id = None
                            try:
                                if gfiles:
                                    group_client_id = self.file_to_client.get(gfiles[0])
                            except Exception:
                                group_client_id = None
                            otio_roles = self._build_otio_camera_roles(client_id=group_client_id) if write_otio else None
                            otio_name = (self.var_otio_timeline_name.get() or "").strip() if write_otio else ""
                            otio_out_dir = None
                            if write_otio and group_client_id:
                                try:
                                    croot = str(self.client_meta.get(group_client_id, {}).get("client_root", "")).strip()
                                    if croot:
                                        otio_out_dir = str(Path(croot) / "02_EDIT")
                                except Exception:
                                    otio_out_dir = None
                            if write_otio and (not otio_roles or not str(otio_roles.get("wide", "")).strip()):
                                self.after(0, lambda: self._log("[otio] Wide camera is required. Skipping OTIO for this group."))
                                write_otio = False
                            out_path = _sync_and_export_multichannel_simple(
                                outs,
                                prefer_48k=self.var_profile.get(),
                                log=lambda m: self.after(0, self._log, m),
                                progress_cb=lambda i, n, m: self.after(0, stage_prog, i, n, m),
                                wav_only=False,
                                use_bw64=self.var_bw64.get(),
                                out_base_dir=(str(group_media_clean_dir) if group_media_clean_dir else (output_override or gname)),
                                flat_output=bool(output_override or group_media_clean_dir),
                                enable_bleed_gate=True,
                                write_otio=write_otio,
                                otio_camera_roles=otio_roles,
                                otio_timeline_name=(otio_name or None),
                                otio_out_dir=otio_out_dir,
                            )
                            if out_path:
                                self.after(0, lambda: self._append_history([(gname, out_path)]))
                                self.after(0, lambda: self._log(f"Group {gi}: multichannel export written: {out_path}"))
                                # Remove per-file outputs when a multichannel export is produced
                                if group_do_sync and group_output_dir:
                                    try:
                                        shutil.rmtree(group_output_dir, ignore_errors=True)
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        for _src, out in results:
                                            try:
                                                Path(out).unlink(missing_ok=True)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                            else:
                                self.after(0, lambda: self._log("Multichannel export failed: no output produced"))
                        except Exception as e:
                            if not isinstance(e, _Cancelled):
                                self.after(0, lambda e=e: self._log(f"Sync/export error: {e}"))
                        finally:
                            sync_dt = time.perf_counter() - sync_t0
                            self.after(0, lambda gi=gi, dt=sync_dt: self._log(f"Timing: group {gi} sync/export {_format_seconds(dt)}"))
                    if not (self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set()):
                        finish_group_progress()
                    group_dt = time.perf_counter() - group_t0
                    self.after(0, lambda gi=gi, dt=group_dt: self._log(f"Timing: group {gi} total {_format_seconds(dt)}"))
                if run_snapshot is not None:
                    run_snapshot["groups_total"] = total_groups
                    run_snapshot["groups_completed"] = processed_groups
                    status = 'completed'
                    if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                        status = 'cancelled'
                    run_snapshot["run_status"] = status
                # Clear the queue after successful enhance
                self.after(0, lambda: self._clear_queue(reset_progress=False))
                # Auto-prune old staging after a successful run
                try:
                    pruned = _prune_staging_dirs(max_age_hours=24.0)
                    if pruned > 0:
                        self.after(0, lambda p=pruned: self._log(f"Pruned {p} old staging folder(s)."))
                except Exception:
                    pass
            except Exception as e:  # noqa: BLE001
                if run_snapshot is not None:
                    run_snapshot["run_status"] = f"error: {e}"
                self.after(0, lambda e=e: self._log(f"Error: {e}"))
            finally:
                try:
                    if staging_root is not None:
                        shutil.rmtree(staging_root, ignore_errors=True)
                except Exception:
                    pass
                self._write_run_flag_log(run_snapshot)
                # Always clean temp artifacts after each run
                cleanup_post_t0 = time.perf_counter()
                _cleanup_run_artifacts(remove_logs=True)
                cleanup_post_dt = time.perf_counter() - cleanup_post_t0
                run_dt = time.perf_counter() - run_t0
                self.after(0, lambda dt=cleanup_post_dt: self._log(f"Timing: post-run cleanup {_format_seconds(dt)}"))
                self.after(0, lambda dt=run_dt: self._log(f"Timing: run total {_format_seconds(dt)}"))
                def _reset():
                    self.run_btn.config(state="normal")
                    self.pause_btn.config(state="disabled")
                    self.cancel_btn.config(state="disabled")
                self.after(0, _reset)

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_pause(self):
        if not hasattr(self, '_control'):
            return
        if self._control.pause.is_set():
            self._control.pause.clear()
            self.pause_btn.configure(text='')
            self._set_button_icon(self.pause_btn, 'pause')
            self._log('Resumed.')
        else:
            self._control.pause.set()
            self.pause_btn.configure(text='')
            self._set_button_icon(self.pause_btn, 'play')
            self._log('Pausing after current chunk...')

    def _cancel_graceful(self):
        if not hasattr(self, '_control'):
            return
        self._control.cancel_now.set()
        self._control.stop_after_chunk.set()
        self.cancel_btn.config(state='disabled')
        self._set_status('Cancelling...')
        self._log('Cancellation requested. Stopping active processing...')
        try:
            _terminate_live_subprocesses(timeout_s=1.0)
        except Exception:
            pass

    def _on_close(self):
        if self._closing:
            return
        self._closing = True
        try:
            if hasattr(self, '_control'):
                self._control.cancel_now.set()
                self._control.stop_after_chunk.set()
        except Exception:
            pass
        try:
            _terminate_live_subprocesses(timeout_s=1.0)
        except Exception:
            pass
        try:
            self.quit()
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass
        if getattr(sys, "frozen", False):
            os._exit(0)


# --- Alignment and multichannel export helpers (Audalign-based) ---

_LOW_INFO_FILLERS = {
    "hmm", "hm", "mhmm", "mmhmm", "mmm", "uhhuh", "uh-huh", "uhuh", "mm", "mhm",
}
_LOW_INFO_ACKS = {
    "yeah", "yep", "ok", "okay", "right", "sure", "cool",
}


def _tokenize_words(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z']+", str(text).lower()) if t]


def _is_low_info_text(text: str) -> bool:
    toks = _tokenize_words(text)
    if not toks:
        return True
    if all(t in _LOW_INFO_FILLERS for t in toks):
        return True
    non_fill = [t for t in toks if t not in _LOW_INFO_FILLERS]
    if len(non_fill) <= 1 and all(t in _LOW_INFO_ACKS or t in _LOW_INFO_FILLERS for t in non_fill):
        return True
    return False


def _build_speaker_turns(
    winner_idx,
    conf,
    hop_ms: float,
    n_speakers: int,
    min_conf: float = 0.66,
    min_dur_s: float = 0.8,
    merge_gap_s: float = 0.35,
) -> list[dict]:
    """Convert per-frame winner/confidence to stable speaker turns."""
    if winner_idx is None or conf is None:
        return []
    if int(getattr(winner_idx, "numel", lambda: 0)()) <= 0:
        return []
    hop_s = max(1e-6, float(hop_ms) / 1000.0)
    min_frames = max(1, int(round(float(min_dur_s) / hop_s)))
    merge_gap_frames = max(0, int(round(float(merge_gap_s) / hop_s)))
    conf_thr = float(min_conf)
    n_frames = int(winner_idx.numel())
    turns: list[dict] = []
    cur_spk = None
    cur_start = 0
    conf_sum = 0.0
    conf_cnt = 0

    def _close(end_f: int):
        nonlocal cur_spk, cur_start, conf_sum, conf_cnt
        if cur_spk is None:
            return
        dur = int(end_f - cur_start)
        if dur >= min_frames and conf_cnt > 0:
            avg_conf = conf_sum / max(1, conf_cnt)
            if avg_conf >= conf_thr:
                turns.append({
                    "speaker_idx": int(cur_spk),
                    "start_s": float(cur_start * hop_s),
                    "end_s": float(end_f * hop_s),
                    "avg_conf": float(avg_conf),
                    "text": "",
                    "is_filtered": False,
                })
        cur_spk = None
        conf_sum = 0.0
        conf_cnt = 0

    for i in range(n_frames):
        spk = int(winner_idx[i].item())
        if spk < 0 or spk >= int(n_speakers):
            _close(i)
            continue
        c = float(conf[i].item()) if i < int(conf.numel()) else 0.0
        if c < conf_thr:
            _close(i)
            continue
        if cur_spk is None:
            cur_spk = spk
            cur_start = i
            conf_sum = c
            conf_cnt = 1
            continue
        if spk == cur_spk:
            conf_sum += c
            conf_cnt += 1
            continue
        _close(i)
        cur_spk = spk
        cur_start = i
        conf_sum = c
        conf_cnt = 1
    _close(n_frames)

    if not turns:
        return turns
    merged: list[dict] = [turns[0]]
    for t in turns[1:]:
        prev = merged[-1]
        gap = float(t["start_s"] - prev["end_s"])
        if int(t["speaker_idx"]) == int(prev["speaker_idx"]) and gap <= float(merge_gap_frames * hop_s):
            prev["end_s"] = float(t["end_s"])
            prev["avg_conf"] = float((float(prev["avg_conf"]) + float(t["avg_conf"])) * 0.5)
        else:
            merged.append(t)
    return merged


def _transcribe_words_openai(wav_path: Path) -> list[dict]:
    api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        return []
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        client = OpenAI(api_key=api_key)
        with wav_path.open("rb") as f:
            rsp = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
    except Exception:
        return []
    try:
        words_raw = getattr(rsp, "words", None)
        if words_raw is None and isinstance(rsp, dict):
            words_raw = rsp.get("words")
        out: list[dict] = []
        for w in (words_raw or []):
            if hasattr(w, "word"):
                txt = str(getattr(w, "word", "") or "")
                st = float(getattr(w, "start", 0.0) or 0.0)
                en = float(getattr(w, "end", st) or st)
            else:
                txt = str((w or {}).get("word", "") or "")
                st = float((w or {}).get("start", 0.0) or 0.0)
                en = float((w or {}).get("end", st) or st)
            out.append({"word": txt, "start": st, "end": max(st, en)})
        return out
    except Exception:
        return []


def _transcribe_words_faster_whisper(path: Path, language: str = "en", model_name: str = "small") -> list[dict]:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except Exception:
        return []
    # CUDA-only by requirement: no CPU fallback.
    try:
        mkey = (str(model_name), str(language or "en"))
        model = _FASTER_WHISPER_MODEL_CACHE.get(mkey)
        if model is None:
            model = WhisperModel(str(model_name), device="cuda", compute_type="float16")
            _FASTER_WHISPER_MODEL_CACHE[mkey] = model
    except Exception:
        return []
    try:
        segments, _info = model.transcribe(
            str(path),
            language=str(language or "en"),
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
    except Exception:
        return []
    out: list[dict] = []
    try:
        for seg in segments:
            words = getattr(seg, "words", None) or []
            for w in words:
                txt = str(getattr(w, "word", "") or "").strip()
                st = float(getattr(w, "start", 0.0) or 0.0)
                en = float(getattr(w, "end", st) or st)
                pr = getattr(w, "probability", None)
                conf = float(pr) if pr is not None else 0.0
                if not txt:
                    continue
                out.append({
                    "word": txt,
                    "start": st,
                    "end": max(st, en),
                    "confidence": conf,
                })
    except Exception:
        return []
    return out


def _transcribe_window_faster_whisper(path: Path, start_s: float, dur_s: float, language: str = "en", model_name: str = "small") -> list[dict]:
    import tempfile
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        return []
    ss = max(0.0, float(start_s))
    dd = max(0.5, float(dur_s))
    try:
        rp = str(path.resolve(strict=False))
        st = path.stat()
        wkey = (rp, int(round(ss * 1000.0)), int(round(dd * 1000.0)), str(language or "en"), str(model_name), int(st.st_mtime), int(st.st_size))
        cached = _WHISPER_WINDOW_CACHE.get(wkey)
        if cached is not None:
            return cached
    except Exception:
        wkey = None
    with tempfile.TemporaryDirectory(prefix="otio_asr_window_") as td:
        tmp_wav = Path(td) / "window.wav"
        try:
            cmd = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{ss:.3f}",
                "-t",
                f"{dd:.3f}",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(tmp_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not tmp_wav.exists():
                if wkey is not None:
                    _WHISPER_WINDOW_CACHE[wkey] = []
                return []
        except Exception:
            return []
        words = _transcribe_words_faster_whisper(tmp_wav, language=language, model_name=model_name)
        out: list[dict] = []
        for w in words:
            try:
                st = float(w.get("start", 0.0)) + ss
                en = float(w.get("end", st)) + ss
                out.append({
                    **w,
                    "start": st,
                    "end": max(st, en),
                })
            except Exception:
                continue
        if wkey is not None:
            _WHISPER_WINDOW_CACHE[wkey] = out
        return out


def _select_anchor_windows(duration_s: float, policy: str = "start_mid_end", window_s: float = 30.0, max_windows: int = 3) -> list[dict]:
    dur = max(0.0, float(duration_s))
    ws = max(5.0, float(window_s))
    if dur <= 0.0:
        return []
    starts: list[tuple[str, float]] = []
    if policy == "start_mid_end":
        starts = [
            ("start", 0.0),
            ("middle", max(0.0, (dur * 0.5) - (ws * 0.5))),
            ("end", max(0.0, dur - ws)),
        ]
    else:
        starts = [("start", 0.0)]
    out: list[dict] = []
    seen: list[float] = []
    for label, s in starts:
        s = max(0.0, min(float(s), max(0.0, dur - ws)))
        if any(abs(s - x) < 0.5 for x in seen):
            continue
        seen.append(s)
        out.append({"label": label, "start_s": s, "dur_s": min(ws, max(0.5, dur - s))})
        if len(out) >= int(max_windows):
            break
    return out


def _window_transcript_quality(words: list[dict], min_words: int = 8) -> dict:
    import numpy as np
    cnt = int(len(words or []))
    confs = [float(w.get("confidence", 0.0)) for w in (words or []) if isinstance(w, dict)]
    med = float(np.median(np.asarray(confs, dtype=float))) if confs else 0.0
    cov = 0.0
    try:
        if words:
            s0 = float(min(float(w.get("start", 0.0)) for w in words))
            s1 = float(max(float(w.get("end", 0.0)) for w in words))
            total = max(0.001, s1 - s0)
            voiced = 0.0
            for w in words:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
                voiced += max(0.0, we - ws)
            cov = float(min(1.0, voiced / total))
    except Exception:
        cov = 0.0
    ok = cnt >= int(min_words) and med >= 0.35
    score = float((0.6 * med) + (0.4 * min(1.0, cnt / max(1.0, float(min_words * 3)))))
    return {"ok": ok, "word_count": cnt, "median_confidence": med, "speech_coverage": cov, "score": score}


def _find_best_content_match(clean_window: dict, scratch_signal, sr: int, max_lag_s: float) -> dict:
    import numpy as np
    out = {
        "ok": False,
        "reason": "no_match",
        "source_start_s": 0.0,
        "offset_s": 0.0,
        "confidence": 0.0,
        "peak_ratio": 0.0,
        "residual_s": 0.0,
        "window_start_s": float(clean_window.get("start_s", 0.0)),
    }
    try:
        if hasattr(scratch_signal, "detach"):
            src = scratch_signal.detach().cpu().float().numpy()
        else:
            src = np.asarray(scratch_signal, dtype=np.float32)
        clean_wave = np.asarray(clean_window.get("clean_wave", []), dtype=np.float32)
        words_local = list(clean_window.get("words_local", []) or [])
        dur_s = max(1.0, float(clean_window.get("dur_s", 0.0)))
        if src.ndim != 1 or src.size < int(sr * 5):
            out["reason"] = "short_source"
            return out
        hop_s = 0.05
        src_env = _build_signal_activity(src, sr, hop_s=hop_s)
        if int(getattr(src_env, "size", 0)) < 16:
            out["reason"] = "env_too_short"
            return out

        # Prefer transcript-derived template; fallback to clean waveform envelope.
        template = None
        if words_local:
            template = _build_transcript_activity(words_local, hop_s=hop_s, duration_s=dur_s)
        if template is None or int(getattr(template, "size", 0)) < 16:
            if clean_wave.size < int(sr * 2):
                out["reason"] = "no_template"
                return out
            template = _build_signal_activity(clean_wave, sr, hop_s=hop_s)
        t = np.asarray(template, dtype=np.float32)
        if t.ndim != 1 or t.size < 16:
            out["reason"] = "template_too_short"
            return out

        n = int(t.size)
        if int(src_env.size) < n + 4:
            out["reason"] = "search_too_short"
            return out
        corr = np.correlate(src_env, t, mode="valid")
        denom_t = float(np.sqrt(np.sum(t * t)) + 1e-9)
        run = np.convolve(src_env * src_env, np.ones(n, dtype=np.float32), mode="valid")
        denom_s = np.sqrt(np.maximum(run, 1e-9))
        scores = corr / (denom_s * denom_t + 1e-9)

        cw_start = max(0.0, float(clean_window.get("start_s", 0.0)))
        cw_dur = max(1.0, dur_s)
        src_total_s = float(src.size / max(1, sr))
        min_start_s = max(0.0, cw_start - max(0.0, float(max_lag_s)))
        max_start_s = min(max(0.0, src_total_s - cw_dur), cw_start + max(0.0, float(max_lag_s)))
        i0 = max(0, int(round(min_start_s / hop_s)))
        i1 = min(int(scores.size), max(i0 + 1, int(round(max_start_s / hop_s)) + 1))
        if i1 <= i0:
            out["reason"] = "lag_bounds_empty"
            return out

        view = scores[i0:i1]
        rel = int(np.argmax(view))
        idx = i0 + rel
        best = float(scores[idx])
        med = float(np.median(np.abs(view)) + 1e-6)
        peak_ratio = float(best / med)
        coarse_src_start_s = float(idx * hop_s)

        # Local waveform refine.
        lag_s = 0.0
        if clean_wave.size >= int(sr * 2):
            s0 = max(0, int(round(coarse_src_start_s * sr)))
            s1 = min(src.size, s0 + clean_wave.size)
            seg = src[s0:s1]
            n2 = min(clean_wave.size, seg.size)
            if n2 >= int(sr * 1):
                lag_samples = int(_gcc_phat_lag(clean_wave[:n2], seg[:n2]))
                lag_s = float(lag_samples / float(sr))
        if abs(lag_s) > 1.0:
            lag_s = 0.0
        refined_src_start_s = max(0.0, coarse_src_start_s + lag_s)
        offset_s = float(cw_start - refined_src_start_s)

        base_score = float(max(0.0, min(1.0, best)))
        ratio_score = float(max(0.0, min(1.0, (peak_ratio - 1.0) / 1.0)))
        res_penalty = 1.0 if abs(lag_s) <= 0.25 else max(0.0, 1.0 - ((abs(lag_s) - 0.25) / 1.75))
        conf = float(max(0.0, min(1.0, (0.25 * base_score) + (0.75 * ratio_score))) * res_penalty)
        out.update({
            "ok": True,
            "reason": "ok",
            "source_start_s": refined_src_start_s,
            "offset_s": offset_s,
            "confidence": conf,
            "peak_ratio": peak_ratio,
            "residual_s": float(lag_s),
        })
        return out
    except Exception as exc:
        out["reason"] = f"error:{exc}"
        return out


def _solve_offsets_from_content_matches(matches_by_source: dict[str, list[dict]]) -> dict:
    import numpy as np
    out = {"ok": False, "offsets": {}, "counts": {}, "confidence": 0.0}
    if not matches_by_source:
        return out
    conf_all: list[float] = []
    for src, matches in matches_by_source.items():
        good = [m for m in (matches or []) if bool(m.get("ok"))]
        out["counts"][src] = int(len(good))
        if not good:
            continue
        offs = np.asarray([float(m.get("offset_s", 0.0)) for m in good], dtype=float)
        # Robust outlier rejection before taking median.
        if offs.size >= 3:
            med = float(np.median(offs))
            mad = float(np.median(np.abs(offs - med)) + 1e-9)
            keep = np.abs(offs - med) <= (3.0 * max(0.25, 1.4826 * mad))
            if np.any(keep):
                offs = offs[keep]
        out["offsets"][src] = float(np.median(offs))
        conf_all.extend([float(m.get("confidence", 0.0)) for m in good])
    if not out["offsets"]:
        return out
    # Normalize to earliest-start = 0 to keep downstream refinement stable.
    vals = [float(v) for v in out["offsets"].values()]
    mn = min(vals) if vals else 0.0
    out["offsets"] = {k: float(v - mn) for k, v in out["offsets"].items()}
    out["ok"] = True
    out["confidence"] = float(np.median(np.asarray(conf_all, dtype=float))) if conf_all else 0.0
    return out


def _validate_alignment_quality(
    refined: dict,
    max_median_residual_ms: float = 120.0,
    min_residual_pairs: int = 1,
) -> dict:
    import numpy as np
    residuals = [abs(float(v)) for v in (refined.get("residuals", {}) or {}).values()]
    evidence_count = int(len(residuals))
    med_ms = float(np.median(np.asarray(residuals, dtype=float)) * 1000.0) if residuals else 0.0
    has_evidence = bool(evidence_count >= int(max(1, min_residual_pairs)))
    ok = bool(has_evidence and (med_ms <= float(max_median_residual_ms)))
    return {
        "ok": ok,
        "median_residual_ms": med_ms,
        "evidence_count": evidence_count,
        "has_evidence": has_evidence,
    }


def _validate_offsets_against_clean(
    source_paths: list[str],
    offsets_by_path: dict[str, float],
    clean_path: str,
    target_sr: int = 48000,
    max_abs_lag_s: float = 0.40,
) -> dict:
    """Validate final offsets by checking aligned residual lag vs clean."""
    import numpy as np
    out = {
        "ok": False,
        "pair_lag_s": {},
        "max_abs_lag_s": 999.0,
        "tested_pairs": 0,
        "valid_pairs": 0,
        "has_evidence": False,
    }
    clean_tracks = _load_audio_tracks_any(clean_path, target_sr=target_sr)
    if not clean_tracks:
        return out
    clean = clean_tracks[0]
    clean_np = clean.detach().cpu().float().numpy() if hasattr(clean, "detach") else np.asarray(clean, dtype=np.float32)
    c_off = float(offsets_by_path.get(clean_path, 0.0))
    c_dur_s = float(len(clean_np) / float(target_sr))
    worst = 0.0
    tested_pairs = 0
    valid_pairs = 0
    for sp in source_paths:
        if str(sp) == str(clean_path):
            continue
        tested_pairs += 1
        tt = _load_audio_tracks_any(sp, target_sr=target_sr)
        if not tt:
            continue
        src = tt[0]
        src_np = src.detach().cpu().float().numpy() if hasattr(src, "detach") else np.asarray(src, dtype=np.float32)
        s_off = float(offsets_by_path.get(str(sp), 0.0))
        s_dur_s = float(len(src_np) / float(target_sr))
        ov_start = max(float(c_off), float(s_off))
        ov_end = min(float(c_off + c_dur_s), float(s_off + s_dur_s))
        ov_len = max(0.0, ov_end - ov_start)
        if ov_len < 5.0:
            continue
        # Robust multi-window lag estimate to avoid single-window false matches.
        win_s = 30.0
        step_s = 15.0
        lags: list[float] = []
        max_windows = 40
        n_steps = max(1, int((ov_len - win_s) / step_s) + 1)
        checked = 0
        for ii in range(n_steps):
            if checked >= max_windows:
                break
            gs = ov_start + (ii * step_s)
            if gs + win_s > ov_end:
                gs = max(ov_start, ov_end - win_s)
            c_i = int(round((gs - c_off) * float(target_sr)))
            s_i = int(round((gs - s_off) * float(target_sr)))
            n_i = int(round(win_s * float(target_sr)))
            c_seg = clean_np[c_i:c_i + n_i]
            s_seg = src_np[s_i:s_i + n_i]
            n2 = min(len(c_seg), len(s_seg))
            if n2 < int(target_sr * 8):
                continue
            c_use = c_seg[:n2]
            s_use = s_seg[:n2]
            c_rms = float(np.sqrt(np.mean(c_use * c_use) + 1e-12))
            s_rms = float(np.sqrt(np.mean(s_use * s_use) + 1e-12))
            if c_rms < 1e-4 or s_rms < 1e-4:
                continue
            checked += 1
            lag_s = float(_gcc_phat_lag(c_use, s_use) / float(target_sr))
            # Keep plausible local residuals only.
            if abs(lag_s) <= 2.0:
                lags.append(lag_s)
        if not lags:
            continue
        med_lag = float(np.median(np.asarray(lags, dtype=float)))
        out["pair_lag_s"][Path(sp).name] = med_lag
        worst = max(worst, abs(med_lag))
        valid_pairs += 1
    out["tested_pairs"] = int(tested_pairs)
    out["valid_pairs"] = int(valid_pairs)
    out["has_evidence"] = bool(valid_pairs > 0)
    out["max_abs_lag_s"] = float(worst if valid_pairs > 0 else 999.0)
    out["ok"] = bool((valid_pairs > 0) and (worst <= float(max_abs_lag_s)))
    return out


def _qa_aligned_audio_sync(
    audio_paths: dict[str, str | Path],
    target_sr: int = 48000,
    max_abs_median_lag_s: float = 0.25,
) -> dict:
    import numpy as np
    out = {
        "ok": False,
        "pair_median_lag_s": {},
        "max_abs_median_lag_s": 999.0,
        "tested_pairs": 0,
        "valid_pairs": 0,
        "has_evidence": False,
    }
    names = list(audio_paths.keys())
    waves: dict[str, np.ndarray] = {}
    for name in names:
        p = str(audio_paths.get(name, "") or "")
        if not p:
            continue
        tt = _load_audio_tracks_any(p, target_sr=target_sr)
        if not tt:
            continue
        x = tt[0]
        x_np = x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
        waves[name] = x_np
    pairs = []
    if "cam1" in waves and "cam2" in waves:
        pairs.append(("cam1", "cam2"))
    if "cam1" in waves and "clean1" in waves:
        pairs.append(("cam1", "clean1"))
    if "cam2" in waves and "clean1" in waves:
        pairs.append(("cam2", "clean1"))
    worst = 0.0
    tested_pairs = 0
    valid_pairs = 0
    for a, b in pairs:
        tested_pairs += 1
        xa = waves[a]
        xb = waves[b]
        ov = min(len(xa), len(xb))
        if ov < int(target_sr * 20):
            continue
        win_s = 20.0
        step_s = 10.0
        lags: list[float] = []
        wrms = []
        for s in range(0, max(1, ov - int(win_s * target_sr)), int(step_s * target_sr)):
            e = s + int(win_s * target_sr)
            aa = xa[s:e]
            bb = xb[s:e]
            if len(aa) < int(target_sr * 8) or len(bb) < int(target_sr * 8):
                continue
            ar = float(np.sqrt(np.mean(aa * aa) + 1e-12))
            br = float(np.sqrt(np.mean(bb * bb) + 1e-12))
            wrms.append((ar, br))
        if not wrms:
            continue
        a_thr = float(np.quantile(np.asarray([x[0] for x in wrms], dtype=float), 0.5))
        b_thr = float(np.quantile(np.asarray([x[1] for x in wrms], dtype=float), 0.5))
        for s in range(0, max(1, ov - int(win_s * target_sr)), int(step_s * target_sr)):
            e = s + int(win_s * target_sr)
            aa = xa[s:e]
            bb = xb[s:e]
            if len(aa) < int(target_sr * 8) or len(bb) < int(target_sr * 8):
                continue
            ar = float(np.sqrt(np.mean(aa * aa) + 1e-12))
            br = float(np.sqrt(np.mean(bb * bb) + 1e-12))
            if ar < a_thr or br < b_thr:
                continue
            lag = float(_gcc_phat_lag(aa, bb) / float(target_sr))
            if abs(lag) <= 3.0:
                lags.append(lag)
        if not lags:
            continue
        med = float(np.median(np.asarray(lags, dtype=float)))
        out["pair_median_lag_s"][f"{a}|{b}"] = med
        worst = max(worst, abs(med))
        valid_pairs += 1
    out["tested_pairs"] = int(tested_pairs)
    out["valid_pairs"] = int(valid_pairs)
    out["has_evidence"] = bool(valid_pairs > 0)
    out["max_abs_median_lag_s"] = float(worst if valid_pairs > 0 else 999.0)
    out["ok"] = bool((valid_pairs > 0) and (worst <= float(max_abs_median_lag_s)))
    return out


def _validate_offsets_pairwise(
    source_paths: list[str],
    offsets_by_path: dict[str, float],
    target_sr: int = 48000,
    max_abs_lag_s: float = 0.40,
) -> dict:
    import numpy as np
    out = {
        "ok": False,
        "pair_lag_s": {},
        "max_abs_lag_s": 999.0,
        "tested_pairs": 0,
        "valid_pairs": 0,
        "has_evidence": False,
    }
    srcs = [str(p).strip() for p in (source_paths or []) if str(p).strip()]
    if len(srcs) < 2:
        return out
    waves: dict[str, np.ndarray] = {}
    for sp in srcs:
        tt = _load_audio_tracks_any(sp, target_sr=target_sr)
        if not tt:
            continue
        x = tt[0]
        waves[sp] = x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
    if len(waves) < 2:
        return out

    def _apply_signed_offset(x: np.ndarray, off_s: float) -> np.ndarray:
        off_samples = int(round(float(off_s) * float(target_sr)))
        if off_samples >= 0:
            return np.concatenate([np.zeros(off_samples, dtype=np.float32), x.astype(np.float32, copy=False)])
        drop = min(len(x), int(-off_samples))
        if drop >= len(x):
            return np.zeros(1, dtype=np.float32)
        return x[drop:].astype(np.float32, copy=False)

    worst = 0.0
    tested = 0
    valid = 0
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            a = srcs[i]
            b = srcs[j]
            if a not in waves or b not in waves:
                continue
            tested += 1
            aa = _apply_signed_offset(waves[a], float(offsets_by_path.get(a, 0.0)))
            bb = _apply_signed_offset(waves[b], float(offsets_by_path.get(b, 0.0)))
            ov = min(len(aa), len(bb))
            if ov < int(target_sr * 20):
                continue
            win_s = 20.0
            step_s = 10.0
            lags: list[float] = []
            for s in range(0, max(1, ov - int(win_s * target_sr)), int(step_s * target_sr)):
                e = s + int(win_s * target_sr)
                xa = aa[s:e]
                xb = bb[s:e]
                if len(xa) < int(target_sr * 8) or len(xb) < int(target_sr * 8):
                    continue
                ar = float(np.sqrt(np.mean(xa * xa) + 1e-12))
                br = float(np.sqrt(np.mean(xb * xb) + 1e-12))
                if ar < 1e-4 or br < 1e-4:
                    continue
                lag = float(_gcc_phat_lag(xa, xb) / float(target_sr))
                if abs(lag) <= 3.0:
                    lags.append(lag)
            if not lags:
                continue
            med = float(np.median(np.asarray(lags, dtype=float)))
            out["pair_lag_s"][f"{Path(a).name}|{Path(b).name}"] = med
            worst = max(worst, abs(med))
            valid += 1
    out["tested_pairs"] = int(tested)
    out["valid_pairs"] = int(valid)
    out["has_evidence"] = bool(valid > 0)
    out["max_abs_lag_s"] = float(worst if valid > 0 else 999.0)
    out["ok"] = bool((valid > 0) and (worst <= float(max_abs_lag_s)))
    return out


def _estimate_global_offsets_from_envelopes(
    source_paths: list[str],
    clean_path: str,
    target_sr: int = 48000,
    max_lag_s: float = 1200.0,
) -> dict[str, float]:
    srcs = [str(p).strip() for p in (source_paths or []) if str(p).strip()]
    out: dict[str, float] = {p: 0.0 for p in srcs}
    if not srcs or not clean_path:
        return out
    hop_s = 0.02
    clean_tracks = _load_audio_tracks_any(clean_path, target_sr=target_sr)
    if not clean_tracks:
        return out
    try:
        if len(clean_tracks) > 1:
            import torch
            clean_mono = torch.mean(torch.stack(clean_tracks, dim=0), dim=0)
        else:
            clean_mono = clean_tracks[0]
    except Exception:
        clean_mono = clean_tracks[0]
    clean_env = _build_signal_activity(clean_mono, target_sr, hop_s=hop_s)
    out[str(clean_path)] = 0.0
    max_lag_frames = int(max(1, round(float(max_lag_s) / hop_s)))
    for sp in srcs:
        if str(sp) == str(clean_path):
            continue
        tt = _load_audio_tracks_any(sp, target_sr=target_sr)
        if not tt:
            continue
        src_env = _build_signal_activity(tt[0], target_sr, hop_s=hop_s)
        try:
            lag_frames = int(_gcc_phat_lag(clean_env, src_env))
        except Exception:
            continue
        if abs(lag_frames) > max_lag_frames:
            continue
        lag_s = float(lag_frames * hop_s)
        # clean is anchor (0): positive source offset means source starts later.
        out[str(sp)] = float(-lag_s)
    # Keep clean anchored at zero and preserve signed offsets for others.
    c0 = float(out.get(str(clean_path), 0.0))
    out = {k: float(v - c0) for k, v in out.items()}
    out[str(clean_path)] = 0.0
    return out


def _build_word_anchors(words: list[dict], min_word_dur_s: float = 0.08) -> list[dict]:
    anchors: list[dict] = []
    for w in (words or []):
        try:
            txt = str(w.get("word", "") or "").strip()
            st = float(w.get("start", 0.0))
            en = float(w.get("end", st))
            conf = float(w.get("confidence", 0.0))
            if not txt:
                continue
            dur = max(0.0, en - st)
            if dur < float(min_word_dur_s):
                continue
            if conf < 0.15:
                continue
            anchors.append({"word": txt, "start": st, "end": en, "confidence": conf})
        except Exception:
            continue
    anchors.sort(key=lambda x: float(x.get("start", 0.0)))
    return anchors


def _build_transcript_activity(words: list[dict], hop_s: float, duration_s: float) -> object:
    import numpy as np
    n = max(1, int(round(max(0.1, float(duration_s)) / float(hop_s))))
    act = np.zeros(n, dtype=np.float32)
    for w in (words or []):
        try:
            s = max(0.0, float(w.get("start", 0.0)))
            e = max(s, float(w.get("end", s)))
            c = float(w.get("confidence", 0.0))
            v = max(0.2, min(1.0, c if c > 0 else 0.6))
            i0 = max(0, int(round(s / hop_s)))
            i1 = min(n, max(i0 + 1, int(round(e / hop_s))))
            act[i0:i1] = np.maximum(act[i0:i1], v)
        except Exception:
            continue
    return act


def _build_signal_activity(mono, sr: int, hop_s: float) -> object:
    import numpy as np
    if hasattr(mono, "detach"):
        x = mono.detach().cpu().float().numpy()
    else:
        x = np.asarray(mono, dtype=np.float32)
    if x.ndim != 1 or x.size < 2:
        return np.zeros(1, dtype=np.float32)
    hop = max(1, int(round(float(sr) * float(hop_s))))
    n = max(1, int(x.size // hop))
    z = x[: n * hop].reshape(n, hop)
    env = np.sqrt(np.mean(z * z, axis=1) + 1e-12).astype(np.float32)
    if env.size > 2:
        p20 = float(np.quantile(env, 0.2))
        env = np.maximum(0.0, env - p20)
    m = float(np.max(env) or 0.0)
    if m > 1e-9:
        env = env / m
    return env


def _align_sources_from_word_anchors(source_paths: list[str], anchor_words: list[dict], target_sr: int = 48000) -> dict:
    import numpy as np
    srcs = [str(p).strip() for p in (source_paths or []) if str(p).strip()]
    out = {"ok": False, "offsets": {p: 0.0 for p in srcs}, "confidence": 0.0, "anchor_count": int(len(anchor_words))}
    if len(srcs) < 2 or not anchor_words:
        return out
    hop_s = 0.05
    clean_path = srcs[-1]
    duration_s = max([float(w.get("end", 0.0)) for w in anchor_words] + [1.0]) + 2.0
    transcript_act = _build_transcript_activity(anchor_words, hop_s=hop_s, duration_s=duration_s)
    if transcript_act is None or int(getattr(transcript_act, "size", 0)) < 4:
        return out
    raw_offsets: dict[str, float] = {}
    qvals: list[float] = []
    for p in srcs:
        tracks = _load_audio_tracks_any(p, target_sr=target_sr)
        if not tracks:
            continue
        try:
            if len(tracks) > 1:
                import torch
                mono = torch.mean(torch.stack(tracks, dim=0), dim=0)
            else:
                mono = tracks[0]
        except Exception:
            mono = tracks[0]
        sact = _build_signal_activity(mono, target_sr, hop_s=hop_s)
        if int(getattr(sact, "size", 0)) < 4:
            continue
        n = min(int(sact.size), int(transcript_act.size))
        if n < 8:
            continue
        lag_frames = int(_gcc_phat_lag(transcript_act[:n], sact[:n]))
        off = float(-lag_frames * hop_s)
        raw_offsets[p] = off
        try:
            c = np.correlate(transcript_act[:n], sact[:n], mode="valid")
            q = float(np.max(c) / (np.mean(np.abs(c)) + 1e-6))
            qvals.append(q)
        except Exception:
            pass
    if clean_path in raw_offsets:
        raw_offsets[clean_path] = 0.0
    if len(raw_offsets) < 2:
        return out
    for p in srcs:
        raw_offsets.setdefault(p, 0.0)
    mn = min(raw_offsets.values()) if raw_offsets else 0.0
    norm = {p: float(max(0.0, raw_offsets.get(p, 0.0) - mn)) for p in srcs}
    if any(v > 600.0 for v in norm.values()):
        return out
    out["ok"] = True
    out["offsets"] = norm
    out["confidence"] = float(np.median(np.asarray(qvals, dtype=float))) if qvals else 0.0
    return out


def _refine_offsets_waveform(local_offsets: dict[str, float], source_paths: list[str], target_sr: int = 48000) -> dict:
    srcs = [str(p).strip() for p in (source_paths or []) if str(p).strip()]
    out = {
        "offsets": {p: float(local_offsets.get(p, 0.0)) for p in srcs},
        "residuals": {},
        "confidence": 0.0,
    }
    if len(srcs) < 2:
        return out
    clean_path = srcs[-1]
    clean_tracks = _load_audio_tracks_any(clean_path, target_sr=target_sr)
    if not clean_tracks:
        return out
    try:
        if len(clean_tracks) > 1:
            import torch
            clean = torch.mean(torch.stack(clean_tracks, dim=0), dim=0)
        else:
            clean = clean_tracks[0]
    except Exception:
        clean = clean_tracks[0]
    if hasattr(clean, "detach"):
        clean_np = clean.detach().cpu().float().numpy()
    else:
        import numpy as np
        clean_np = np.asarray(clean, dtype=np.float32)
    qvals: list[float] = []

    def _apply_signed_offset(x, off_s: float):
        import numpy as np
        off_samples = int(round(float(off_s) * float(target_sr)))
        if off_samples >= 0:
            return np.concatenate([np.zeros(off_samples, dtype=np.float32), x.astype(np.float32, copy=False)])
        drop = min(len(x), int(-off_samples))
        if drop >= len(x):
            return np.zeros(1, dtype=np.float32)
        return x[drop:].astype(np.float32, copy=False)

    for p in srcs[:-1]:
        tracks = _load_audio_tracks_any(p, target_sr=target_sr)
        if not tracks:
            continue
        mono = tracks[0]
        if hasattr(mono, "detach"):
            src_np = mono.detach().cpu().float().numpy()
        else:
            import numpy as np
            src_np = np.asarray(mono, dtype=np.float32)
        c_off = float(out["offsets"].get(clean_path, 0.0))
        s_off = float(out["offsets"].get(p, 0.0))
        try:
            ca = _apply_signed_offset(clean_np, c_off)
            sa = _apply_signed_offset(src_np, s_off)
        except Exception:
            continue
        n = min(len(ca), len(sa), int(target_sr * 300))
        if n < int(target_sr * 10):
            continue
        lag = float(_gcc_phat_lag(ca[:n], sa[:n]) / float(target_sr))
        residual = float(-lag)
        if abs(residual) > 2.0:
            # Treat large residuals as correlation failures instead of applying
            # destructive global shifts.
            continue
        out["offsets"][p] = float(out["offsets"].get(p, 0.0)) + residual
        out["residuals"][p] = residual
        qvals.append(abs(residual))
    # Anchor normalization: keep clean at 0, preserve signed deltas for others.
    clean_off = float(out["offsets"].get(clean_path, 0.0))
    out["offsets"] = {k: float(v - clean_off) for k, v in out["offsets"].items()}
    out["offsets"][clean_path] = 0.0
    out["confidence"] = float(1.0 / (1.0 + (sum(qvals) / max(1, len(qvals))))) if qvals else 0.0
    return out


def _build_turns_from_transcript(words: list[dict], clean_offset_s: float) -> list[dict]:
    ww = []
    for w in (words or []):
        try:
            txt = str(w.get("word", "") or "").strip()
            st = float(w.get("start", 0.0))
            en = float(w.get("end", st))
            spk = int(w.get("speaker_idx", 0))
            conf = float(w.get("speaker_conf", w.get("confidence", 0.0)))
            if not txt or en <= st:
                continue
            ww.append({"word": txt, "start": st, "end": en, "speaker_idx": spk, "speaker_conf": conf})
        except Exception:
            continue
    ww.sort(key=lambda x: float(x["start"]))
    if not ww:
        return []
    max_gap_s = 0.6
    turns: list[dict] = []
    cur = None
    for w in ww:
        if cur is None:
            cur = {
                "speaker_idx": int(w["speaker_idx"]),
                "start_s": float(w["start"] + clean_offset_s),
                "end_s": float(w["end"] + clean_offset_s),
                "avg_conf": float(w["speaker_conf"]),
                "text_words": [str(w["word"])],
                "n": 1,
            }
            continue
        gap = float(w["start"] + clean_offset_s) - float(cur["end_s"])
        if int(w["speaker_idx"]) == int(cur["speaker_idx"]) and gap <= max_gap_s:
            cur["end_s"] = float(w["end"] + clean_offset_s)
            cur["avg_conf"] = float((float(cur["avg_conf"]) * float(cur["n"]) + float(w["speaker_conf"])) / float(cur["n"] + 1))
            cur["n"] = int(cur["n"]) + 1
            cur["text_words"].append(str(w["word"]))
        else:
            txt = " ".join(cur["text_words"]).strip()
            turns.append({
                "speaker_idx": int(cur["speaker_idx"]),
                "start_s": float(cur["start_s"]),
                "end_s": float(cur["end_s"]),
                "avg_conf": float(cur["avg_conf"]),
                "text": txt,
                "is_filtered": bool(_is_low_info_text(txt)) if txt else False,
            })
            cur = {
                "speaker_idx": int(w["speaker_idx"]),
                "start_s": float(w["start"] + clean_offset_s),
                "end_s": float(w["end"] + clean_offset_s),
                "avg_conf": float(w["speaker_conf"]),
                "text_words": [str(w["word"])],
                "n": 1,
            }
    if cur is not None:
        txt = " ".join(cur["text_words"]).strip()
        turns.append({
            "speaker_idx": int(cur["speaker_idx"]),
            "start_s": float(cur["start_s"]),
            "end_s": float(cur["end_s"]),
            "avg_conf": float(cur["avg_conf"]),
            "text": txt,
            "is_filtered": bool(_is_low_info_text(txt)) if txt else False,
        })
    return turns


def _filter_short_acks_from_turns(turns: list[dict], min_hold_s: float = 10.0) -> list[dict]:
    seq = sorted(list(turns or []), key=lambda t: float(t.get("start_s", 0.0)))
    if not seq:
        return []
    # Drop low-info short acknowledgements first.
    pre: list[dict] = []
    for t in seq:
        s = float(t.get("start_s", 0.0))
        e = max(s, float(t.get("end_s", s)))
        txt = str(t.get("text", "") or "")
        if (e - s) < 2.0 and _is_low_info_text(txt):
            continue
        pre.append({**t, "start_s": s, "end_s": e})
    if not pre:
        return []
    current_spk = int(pre[0].get("speaker_idx", 0))
    pending_spk = None
    pending_acc = 0.0
    out: list[dict] = []
    for t in pre:
        s = float(t.get("start_s", 0.0))
        e = max(s, float(t.get("end_s", s)))
        desired = int(t.get("speaker_idx", current_spk))
        base = dict(t)
        if desired == current_spk:
            pending_spk = None
            pending_acc = 0.0
            out.append({**base, "speaker_idx": current_spk, "start_s": s, "end_s": e})
            continue
        if pending_spk != desired:
            pending_spk = desired
            pending_acc = 0.0
        need = max(0.0, float(min_hold_s) - float(pending_acc))
        dur = max(0.0, e - s)
        if dur < need:
            pending_acc += dur
            out.append({**base, "speaker_idx": current_spk, "start_s": s, "end_s": e})
            continue
        cross_s = s + need
        if need > 1e-6:
            out.append({**base, "speaker_idx": current_spk, "start_s": s, "end_s": cross_s})
        current_spk = desired
        pending_spk = None
        pending_acc = 0.0
        out.append({**base, "speaker_idx": current_spk, "start_s": cross_s, "end_s": e})
    # Merge adjacent same-speaker segments.
    merged: list[dict] = []
    for t in out:
        if not merged:
            merged.append(t)
            continue
        p = merged[-1]
        if int(p.get("speaker_idx", -1)) == int(t.get("speaker_idx", -2)) and abs(float(p.get("end_s", 0.0)) - float(t.get("start_s", 0.0))) < 1e-6:
            p["end_s"] = float(t.get("end_s", p.get("end_s", 0.0)))
            p["text"] = (str(p.get("text", "") or "") + " " + str(t.get("text", "") or "")).strip()
            p["avg_conf"] = float((float(p.get("avg_conf", 0.0)) + float(t.get("avg_conf", 0.0))) * 0.5)
        else:
            merged.append(t)
    return merged


def _attach_turn_text(turns: list[dict], words_by_speaker: dict[int, list[dict]]) -> None:
    for t in turns:
        si = int(t.get("speaker_idx", -1))
        t0 = float(t.get("start_s", 0.0))
        t1 = float(t.get("end_s", t0))
        words = []
        for w in words_by_speaker.get(si, []):
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            if we >= t0 and ws <= t1:
                words.append(str(w.get("word", "") or "").strip())
        txt = " ".join([w for w in words if w])
        t["text"] = txt
        if txt:
            t["is_filtered"] = bool(_is_low_info_text(txt))
        else:
            t["is_filtered"] = False


def _probe_video_fps(path: Path, default_fps: float = 25.0) -> float:
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ffprobe:
        return float(default_fps)
    try:
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=0",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"(avg_frame_rate|r_frame_rate)=([0-9]+)/([0-9]+)", txt)
        if m:
            num = float(m.group(2))
            den = float(m.group(3))
            if den > 0:
                fps = num / den
                if fps > 1.0:
                    return fps
    except Exception:
        pass
    return float(default_fps)


def _select_camera_for_speaker(speaker_idx: int, host_idx: int, wide: str, camera_roles: dict[str, str]) -> str:
    host_paths: list[str] = []
    guest_paths: list[str] = []
    host_close = str(camera_roles.get("host_closeup", "")).strip()
    guest_close = str(camera_roles.get("guest_closeup", "")).strip()
    if host_close:
        host_paths.append(host_close)
    if guest_close:
        guest_paths.append(guest_close)
    for k, v in camera_roles.items():
        vv = str(v or "").strip()
        if not vv:
            continue
        lk = str(k).lower()
        if lk.startswith("extra_host"):
            host_paths.append(vv)
        elif lk.startswith("extra_guest"):
            guest_paths.append(vv)
    if int(speaker_idx) == int(host_idx):
        return host_paths[0] if host_paths else wide
    return guest_paths[0] if guest_paths else wide


def _infer_host_speaker_idx(turns: list[dict], intro_probe_s: float = 45.0, fallback_idx: int = 0) -> int:
    """Infer host speaker from intro section where host is expected to dominate."""
    scores: dict[int, float] = {}
    for t in (turns or []):
        try:
            s = max(0.0, float(t.get("start_s", 0.0)))
            e = max(s, float(t.get("end_s", s)))
            if s >= float(intro_probe_s):
                continue
            ov = max(0.0, min(e, float(intro_probe_s)) - s)
            if ov <= 0.0:
                continue
            spk = int(t.get("speaker_idx", fallback_idx))
            conf = float(t.get("avg_conf", 0.0))
            w = ov * (0.5 + max(0.0, min(1.0, conf)))
            scores[spk] = float(scores.get(spk, 0.0) + w)
        except Exception:
            continue
    if not scores:
        return int(fallback_idx)
    return int(max(scores.items(), key=lambda kv: kv[1])[0])


def _normalize_phrase_text(s: str) -> str:
    out = []
    for ch in str(s or "").lower():
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _infer_host_from_intro_keywords(
    words: list[dict],
    turns: list[dict],
    keyword_phrases: list[str] | None = None,
    min_word_conf: float = 0.30,
    fuzzy_threshold: float = 0.84,
) -> dict:
    """Infer host by first confident keyword phrase spoken in intro transcript."""
    import difflib

    phrases = keyword_phrases or [
        "hello",
        "viewers",
        "welcome",
        "tim poole",
        "will underwood",
        "rory calland",
    ]
    phrase_norm = [_normalize_phrase_text(p) for p in phrases if str(p or "").strip()]
    if not phrase_norm or not words or not turns:
        return {"ok": False, "host_idx": None, "match": None, "reason": "missing_data"}

    # Build normalized word stream with timing.
    ws: list[dict] = []
    for w in words:
        try:
            txt = _normalize_phrase_text(str(w.get("word", "") or ""))
            if not txt:
                continue
            st = float(w.get("start", 0.0))
            en = float(w.get("end", st))
            cf = float(w.get("confidence", 0.0))
            if cf < float(min_word_conf):
                continue
            ws.append({"text": txt, "start": st, "end": max(st, en), "confidence": cf})
        except Exception:
            continue
    if not ws:
        return {"ok": False, "host_idx": None, "match": None, "reason": "no_confident_words"}

    def _speaker_for_time(ts: float) -> int | None:
        best: tuple[float, int] | None = None
        for t in turns:
            try:
                s = float(t.get("start_s", 0.0))
                e = float(t.get("end_s", s))
                spk = int(t.get("speaker_idx", 0))
                if ts >= s and ts <= e:
                    return spk
                # nearest fallback
                d = min(abs(ts - s), abs(ts - e))
                if best is None or d < best[0]:
                    best = (d, spk)
            except Exception:
                continue
        if best and best[0] <= 1.5:
            return int(best[1])
        return None

    # Phrase matching over 1..4 token windows.
    tokens = [x["text"] for x in ws]
    for i in range(len(ws)):
        for n in range(1, min(4, len(ws) - i) + 1):
            phrase = " ".join(tokens[i:i + n]).strip()
            if not phrase:
                continue
            for target in phrase_norm:
                if not target:
                    continue
                # Robustness: exact contains or fuzzy similarity.
                exact = (phrase == target) or (phrase in target) or (target in phrase)
                ratio = float(difflib.SequenceMatcher(a=phrase, b=target).ratio())
                if exact or ratio >= float(fuzzy_threshold):
                    tmid = float(ws[i]["start"])
                    spk = _speaker_for_time(tmid)
                    if spk is None:
                        continue
                    return {
                        "ok": True,
                        "host_idx": int(spk),
                        "match": {
                            "heard": phrase,
                            "target": target,
                            "time_s": tmid,
                            "confidence": float(ws[i].get("confidence", 0.0)),
                            "ratio": ratio,
                        },
                        "reason": "keyword_match",
                    }
    return {"ok": False, "host_idx": None, "match": None, "reason": "no_keyword_match"}


def _probe_media_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not ffprobe:
        return 0.0
    try:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return max(0.0, float((proc.stdout or "0").strip() or 0.0))
    except Exception:
        return 0.0


def _compute_camera_overlap_window(camera_paths: list[str]) -> tuple[float, float] | None:
    vals: list[float] = []
    for p in camera_paths:
        d = _probe_media_duration(Path(p))
        if d > 0.0:
            vals.append(d)
    if not vals:
        return None
    return (0.0, float(min(vals)))


def _detect_speech_onset_seconds(mono, sr: int, hop_s: float = 0.1, min_stable_s: float = 1.5) -> float | None:
    try:
        import numpy as np
        import torch
        if isinstance(mono, torch.Tensor):
            x = mono.detach().cpu().float().numpy()
        else:
            x = np.asarray(mono, dtype=np.float32)
        if x.ndim != 1 or x.size < int(sr * 0.5):
            return None
        hop = max(1, int(round(float(sr) * float(hop_s))))
        n = int(x.size // hop)
        if n < 8:
            return None
        x = x[: n * hop].reshape(n, hop)
        env = np.sqrt(np.mean(x * x, axis=1) + 1e-12)
        nf = float(np.quantile(env, 0.2))
        thr = max(nf * 3.5, nf + float(np.std(env)) * 2.0)
        voiced = env > thr
        need = max(1, int(round(float(min_stable_s) / float(hop_s))))
        run = 0
        for i, v in enumerate(voiced.tolist()):
            run = run + 1 if v else 0
            if run >= need:
                return float((i - need + 1) * hop_s)
        return None
    except Exception:
        return None


def _estimate_pair_offset_voiced(ref_mono, src_mono, sr: int, ref_onset_s: float, src_onset_s: float, window_s: float = 45.0) -> dict:
    try:
        import numpy as np
        if hasattr(ref_mono, "detach"):
            ref = ref_mono.detach().cpu().float().numpy()
        else:
            ref = np.asarray(ref_mono, dtype=np.float32)
        if hasattr(src_mono, "detach"):
            src = src_mono.detach().cpu().float().numpy()
        else:
            src = np.asarray(src_mono, dtype=np.float32)
        if ref.size < sr or src.size < sr:
            return {"ok": False, "reason": "too_short", "offset_s": 0.0, "residual_s": None}

        w = int(max(1, round(window_s * sr)))
        rs = max(0, int(round(ref_onset_s * sr)) - (w // 2))
        ss = max(0, int(round(src_onset_s * sr)) - (w // 2))
        re = min(ref.size, rs + w)
        se = min(src.size, ss + w)
        rw = ref[rs:re]
        sw = src[ss:se]
        n = min(rw.size, sw.size)
        if n < int(sr * 2):
            return {"ok": False, "reason": "window_too_short", "offset_s": 0.0, "residual_s": None}
        rw = rw[:n]
        sw = sw[:n]
        lag = float(_gcc_phat_lag(rw, sw) / float(sr))
        # src offset relative to ref: positive means src starts later.
        off = float(-lag + (src_onset_s - ref_onset_s))

        # residual check after applying solved offset.
        d = int(round(max(0.0, off) * sr))
        sw2 = sw[d:] if d > 0 and d < sw.size else sw
        n2 = min(rw.size, sw2.size)
        if n2 >= int(sr * 1):
            res = float(_gcc_phat_lag(rw[:n2], sw2[:n2]) / float(sr))
        else:
            res = None
        return {"ok": True, "reason": "ok", "offset_s": off, "residual_s": res}
    except Exception as exc:
        return {"ok": False, "reason": f"error:{exc}", "offset_s": 0.0, "residual_s": None}


def _solve_multisource_offsets_from_anchor(source_paths: list[str], target_sr: int = 48000, log=None) -> dict:
    def _emit(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    srcs = [str(p).strip() for p in source_paths if str(p).strip()]
    out = {
        "offsets": {p: 0.0 for p in srcs},
        "onsets": {p: None for p in srcs},
        "quality": {},
    }
    if len(srcs) < 2:
        return out
    monos: dict[str, object] = {}
    for p in srcs:
        tr = _load_audio_tracks_any(p, target_sr=target_sr)
        if not tr:
            _emit(f"[otio] source decode failed for onset solve: {p}")
            continue
        try:
            if len(tr) > 1:
                import torch
                mono = torch.mean(torch.stack(tr, dim=0), dim=0)
            else:
                mono = tr[0]
            monos[p] = mono
        except Exception:
            monos[p] = tr[0]
    if len(monos) < 2:
        return out

    onsets: dict[str, float] = {}
    for p, m in monos.items():
        o = _detect_speech_onset_seconds(m, target_sr)
        if o is None:
            o = 0.0
        onsets[p] = float(o)
        out["onsets"][p] = float(o)

    # anchor pair: cam1 (first source) vs lav1 (last source).
    cam1 = srcs[0]
    lav1 = srcs[-1]
    if cam1 not in monos or lav1 not in monos:
        return out
    raw: dict[str, float] = {cam1: 0.0}
    q: dict[str, dict] = {}
    p1 = _estimate_pair_offset_voiced(monos[cam1], monos[lav1], target_sr, onsets.get(cam1, 0.0), onsets.get(lav1, 0.0))
    q[f"{Path(cam1).name}->{Path(lav1).name}"] = dict(p1)
    if p1.get("ok"):
        raw[lav1] = float(p1.get("offset_s", 0.0))
    else:
        raw[lav1] = float(onsets.get(lav1, 0.0) - onsets.get(cam1, 0.0))

    # other sources relative to cam1.
    for p in srcs[1:-1]:
        if p not in monos:
            continue
        pr = _estimate_pair_offset_voiced(monos[cam1], monos[p], target_sr, onsets.get(cam1, 0.0), onsets.get(p, 0.0))
        q[f"{Path(cam1).name}->{Path(p).name}"] = dict(pr)
        if pr.get("ok"):
            raw[p] = float(pr.get("offset_s", 0.0))
        else:
            raw[p] = float(onsets.get(p, 0.0) - onsets.get(cam1, 0.0))

    # fill any missing.
    for p in srcs:
        if p not in raw:
            raw[p] = float(onsets.get(p, 0.0) - onsets.get(cam1, 0.0))

    # normalize to earliest start.
    mn = min(raw.values()) if raw else 0.0
    norm = {p: float(max(0.0, float(v - mn))) for p, v in raw.items()}
    # sanity clamp.
    if any(v > 600.0 for v in norm.values()):
        _emit("[otio] onset-anchor solve produced implausible offsets; forcing zero.")
        norm = {p: 0.0 for p in srcs}
    out["offsets"] = norm
    out["quality"] = q
    return out


def _normalize_segments_strict(segments: list[dict], window_start: float, window_end: float, hold_cam: str) -> list[dict]:
    sgs = []
    for s in segments:
        try:
            s0 = float(s.get("start_s", 0.0))
            e0 = float(s.get("end_s", s0))
            cam = str(s.get("camera_path", hold_cam) or hold_cam)
            if e0 <= s0:
                continue
            sgs.append({
                "start_s": max(window_start, s0),
                "end_s": min(window_end, e0),
                "camera_path": cam,
                "speaker_idx": int(s.get("speaker_idx", 0)),
                "avg_conf": float(s.get("avg_conf", 0.0)),
                "text": str(s.get("text", "") or ""),
                "is_filtered": bool(s.get("is_filtered", False)),
            })
        except Exception:
            continue
    sgs = [x for x in sgs if x["end_s"] > x["start_s"]]
    sgs.sort(key=lambda x: (x["start_s"], x["end_s"]))
    out: list[dict] = []
    cursor = float(window_start)
    last_cam = hold_cam
    for s in sgs:
        ss = max(cursor, float(s["start_s"]))
        ee = min(float(window_end), float(s["end_s"]))
        if ee <= ss:
            continue
        if ss > cursor:
            out.append({
                "start_s": cursor,
                "end_s": ss,
                "camera_path": last_cam,
                "speaker_idx": int(s.get("speaker_idx", 0)),
                "avg_conf": 0.0,
                "text": "",
                "is_filtered": False,
            })
        out.append({**s, "start_s": ss, "end_s": ee})
        cursor = ee
        last_cam = str(s.get("camera_path", last_cam))
    if cursor < float(window_end):
        out.append({
            "start_s": cursor,
            "end_s": float(window_end),
            "camera_path": last_cam,
            "speaker_idx": 0,
            "avg_conf": 0.0,
            "text": "",
            "is_filtered": False,
        })
    # merge adjacent same-camera.
    merged: list[dict] = []
    for s in out:
        if not merged:
            merged.append(s)
            continue
        p = merged[-1]
        if str(p.get("camera_path")) == str(s.get("camera_path")) and abs(float(p["end_s"]) - float(s["start_s"])) < 1e-6:
            p["end_s"] = float(s["end_s"])
        else:
            merged.append(s)
    return merged


def _apply_camera_pre_switch(segments: list[dict], lead_s: float = 1.0, min_seg_s: float = 0.12) -> list[dict]:
    """Move camera cut boundaries earlier so incoming shot appears before speech turn."""
    if not segments:
        return []
    out = [dict(s) for s in segments]
    lead = max(0.0, float(lead_s))
    min_seg = max(0.02, float(min_seg_s))
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        try:
            if str(prev.get("camera_path", "")) == str(cur.get("camera_path", "")):
                continue
            b = float(cur.get("start_s", 0.0))
            prev_start = float(prev.get("start_s", 0.0))
            cur_end = float(cur.get("end_s", b))
            target = float(b - lead)
            lo = float(prev_start + min_seg)
            hi = float(cur_end - min_seg)
            if hi <= lo:
                continue
            nb = max(lo, min(hi, target))
            prev["end_s"] = nb
            cur["start_s"] = nb
        except Exception:
            continue
    # Drop any degenerate pieces and merge accidental same-cam joins.
    cleaned = []
    for s in out:
        try:
            ss = float(s.get("start_s", 0.0))
            ee = float(s.get("end_s", ss))
            if ee - ss >= min_seg:
                cleaned.append({**s, "start_s": ss, "end_s": ee})
        except Exception:
            continue
    merged: list[dict] = []
    for s in cleaned:
        if not merged:
            merged.append(s)
            continue
        p = merged[-1]
        if str(p.get("camera_path", "")) == str(s.get("camera_path", "")) and abs(float(p["end_s"]) - float(s["start_s"])) < 1e-6:
            p["end_s"] = float(s["end_s"])
        else:
            merged.append(s)
    return merged


def _estimate_source_offsets_audalign(source_paths: list[str], log=None) -> dict[str, float]:
    def _emit(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    srcs = [str(p).strip() for p in (source_paths or []) if str(p).strip()]
    out: dict[str, float] = {p: 0.0 for p in srcs}
    if len(srcs) < 2:
        return out
    persistent_key = _stable_offsets_cache_key(srcs)
    persistent = _load_persistent_audalign_offsets(persistent_key)
    if persistent:
        _emit("[otio] reuse persistent audalign offsets.")
        return {p: float(persistent.get(p, persistent.get(str(Path(p).name), 0.0))) for p in srcs}
    try:
        cache_sig = []
        for p in srcs:
            pp = Path(p)
            st = pp.stat()
            cache_sig.append((str(pp.resolve(strict=False)), int(st.st_mtime), int(st.st_size)))
        ckey = tuple(sorted(cache_sig))
        cached = _AUDALIGN_OFFSETS_CACHE.get(ckey)
        if cached:
            _emit("[otio] reuse cached audalign offsets.")
            # Return exactly for requested paths.
            return {p: float(cached.get(p, 0.0)) for p in srcs}
    except Exception:
        ckey = None
    try:
        import importlib
        ad = importlib.import_module("audalign")
    except Exception as exc:
        _emit(f"[otio] audalign unavailable for source offsets: {exc}")
        return out

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmp_dir = Path.cwd() / ".enhancer_runs_gui" / "tmp_otio_align" / stamp
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return out

    def _make_recognizer(name: str):
        try:
            rec_cls = getattr(ad, name)
            rec = rec_cls()
        except Exception:
            return None
        try:
            cfg = getattr(rec, "config", None)
            if cfg is not None:
                setattr(cfg, "multiprocessing", False)
                setattr(cfg, "num_processors", 1)
                try:
                    setattr(cfg, "n_jobs", 1)
                except Exception:
                    pass
                try:
                    setattr(cfg, "workers", 1)
                except Exception:
                    pass
        except Exception:
            pass
        return rec

    results = None
    primary = _make_recognizer("FingerprintRecognizer")
    fallback = _make_recognizer("CorrelationSpectrogramRecognizer") or _make_recognizer("CorrelationRecognizer")
    align_fn = getattr(ad, "align_files", None)
    if not callable(align_fn):
        _emit("[otio] audalign.align_files missing; using zero offsets.")
        return out
    try:
        for rec in (primary, fallback, None):
            try:
                kwargs = {"destination_path": str(tmp_dir)}
                if rec is not None:
                    kwargs["recognizer"] = rec
                with _suppress_stdout_stderr():
                    results = align_fn(*srcs, **kwargs)
            except Exception:
                results = None
            if isinstance(results, dict) and results:
                break
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    if not isinstance(results, dict):
        _emit("[otio] audalign source offset solve failed; using zero offsets.")
        return out

    # audalign values are "delay to apply". Build both sign conventions and select
    # the one with better residual sync vs clean to avoid sign mistakes.
    delays: dict[str, float] = {}
    for p in srcs:
        delays[p] = float(results.get(Path(p).name, results.get(p, 0.0)) or 0.0)
    clean_key = srcs[-1]
    clean_delay = float(delays.get(clean_key, 0.0))
    rel_pos = {p: float(delays.get(p, 0.0) - clean_delay) for p in srcs}
    rel_neg = {p: float(-(delays.get(p, 0.0) - clean_delay)) for p in srcs}
    rel_pos[clean_key] = 0.0
    rel_neg[clean_key] = 0.0
    # Soft sanity clamp only for clearly invalid solves.
    if any(abs(v) > 7200.0 for v in rel_pos.values()) or any(abs(v) > 7200.0 for v in rel_neg.values()):
        _emit("[otio] audalign returned implausible offsets; using zero offsets.")
        return out
    try:
        vp = _validate_offsets_against_clean(srcs, rel_pos, clean_path=clean_key, target_sr=48000, max_abs_lag_s=999.0)
        vn = _validate_offsets_against_clean(srcs, rel_neg, clean_path=clean_key, target_sr=48000, max_abs_lag_s=999.0)
        wp = float(vp.get("max_abs_lag_s", 999.0))
        wn = float(vn.get("max_abs_lag_s", 999.0))
        if wn + 1e-6 < wp:
            _emit(f"[otio] audalign sign selected: negative (max_lag={wn:.3f}s vs {wp:.3f}s)")
            if ckey is not None:
                _AUDALIGN_OFFSETS_CACHE[ckey] = dict(rel_neg)
            _save_persistent_audalign_offsets(persistent_key, rel_neg)
            return rel_neg
        _emit(f"[otio] audalign sign selected: positive (max_lag={wp:.3f}s vs {wn:.3f}s)")
        if ckey is not None:
            _AUDALIGN_OFFSETS_CACHE[ckey] = dict(rel_pos)
        _save_persistent_audalign_offsets(persistent_key, rel_pos)
        return rel_pos
    except Exception:
        if ckey is not None:
            _AUDALIGN_OFFSETS_CACHE[ckey] = dict(rel_pos)
        _save_persistent_audalign_offsets(persistent_key, rel_pos)
        return rel_pos


def _estimate_clean_offset_to_cam_base(
    cam_ref_path: str,
    cam_ref_offset_s: float,
    clean_path: str,
    target_sr: int = 48000,
    log=None,
) -> dict:
    def _emit(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    cam_ref = str(cam_ref_path or "").strip()
    clean = str(clean_path or "").strip()
    if not cam_ref or not clean:
        return {"match_found": False, "offset_s": None, "best_lag_s": None, "cam_ref": cam_ref, "reason": "missing_path"}
    try:
        import numpy as np

        def _apply_signed_offset_np(x: np.ndarray, off_s: float) -> np.ndarray:
            off_samples = int(round(float(off_s) * float(target_sr)))
            if off_samples >= 0:
                return np.concatenate([np.zeros(off_samples, dtype=np.float32), x.astype(np.float32, copy=False)])
            drop = min(len(x), int(-off_samples))
            if drop >= len(x):
                return np.zeros(1, dtype=np.float32)
            return x[drop:].astype(np.float32, copy=False)

        def _speech_overlap_score(cam_off: float, clean_off: float) -> float:
            tt_cam = _load_audio_tracks_any(cam_ref, target_sr=target_sr)
            tt_clean = _load_audio_tracks_any(clean, target_sr=target_sr)
            if not tt_cam or not tt_clean:
                return 0.0
            cam_np = tt_cam[0].detach().cpu().float().numpy() if hasattr(tt_cam[0], "detach") else np.asarray(tt_cam[0], dtype=np.float32)
            clean_np = tt_clean[0].detach().cpu().float().numpy() if hasattr(tt_clean[0], "detach") else np.asarray(tt_clean[0], dtype=np.float32)
            ca = _apply_signed_offset_np(cam_np, cam_off)
            la = _apply_signed_offset_np(clean_np, clean_off)
            n = min(len(ca), len(la))
            if n < int(target_sr * 20):
                return 0.0
            hop = max(1, int(round(float(target_sr) * 0.1)))
            m = n // hop
            if m < 32:
                return 0.0
            ca = ca[: m * hop].reshape(m, hop)
            la = la[: m * hop].reshape(m, hop)
            ce = np.sqrt(np.mean(ca * ca, axis=1) + 1e-12)
            le = np.sqrt(np.mean(la * la, axis=1) + 1e-12)
            ct = max(1e-6, float(np.quantile(ce, 0.65)))
            lt = max(1e-6, float(np.quantile(le, 0.65)))
            cv = ce >= ct
            lv = le >= lt
            inter = float(np.sum(cv & lv))
            union = float(np.sum(cv | lv) + 1e-9)
            return max(0.0, min(1.0, inter / union))

        pair = _estimate_source_offsets_audalign([cam_ref, clean], log=log)
        # pair result keeps clean at 0 and returns cam offset in that basis.
        cam_in_pair = float(pair.get(cam_ref, 0.0))
        # Keep cam fixed at cam_ref_offset_s, solve clean in global basis.
        # Try both signs and pick the one with better pairwise residual.
        cand_a = float(cam_ref_offset_s - cam_in_pair)
        cand_b = float(cam_ref_offset_s + cam_in_pair)
        qa_a = _validate_offsets_pairwise(
            [cam_ref, clean],
            {cam_ref: float(cam_ref_offset_s), clean: float(cand_a)},
            target_sr=target_sr,
            max_abs_lag_s=999.0,
        )
        qa_b = _validate_offsets_pairwise(
            [cam_ref, clean],
            {cam_ref: float(cam_ref_offset_s), clean: float(cand_b)},
            target_sr=target_sr,
            max_abs_lag_s=999.0,
        )
        wa = float(qa_a.get("max_abs_lag_s", 999.0))
        wb = float(qa_b.get("max_abs_lag_s", 999.0))
        va = bool(qa_a.get("has_evidence", False))
        vb = bool(qa_b.get("has_evidence", False))
        score_a = _speech_overlap_score(float(cam_ref_offset_s), float(cand_a))
        score_b = _speech_overlap_score(float(cam_ref_offset_s), float(cand_b))
        if va and vb:
            # Primary: speech-overlap similarity. Secondary: residual lag.
            if abs(score_a - score_b) >= 0.02:
                pick = cand_a if score_a > score_b else cand_b
                best = wa if score_a > score_b else wb
            else:
                if abs(wa - wb) <= 0.01:
                    # When both candidates are effectively tied, prefer negative
                    # clean shift so leading clean pre-roll is trimmed.
                    pick = cand_a if cand_a <= cand_b else cand_b
                    best = wa if pick == cand_a else wb
                else:
                    pick = cand_a if wa <= wb else cand_b
                    best = wa if wa <= wb else wb
            _emit(
                "[otio] clean sign resolve: "
                f"candA={cand_a:.3f}s lag={wa:.3f}s score={score_a:.3f}, "
                f"candB={cand_b:.3f}s lag={wb:.3f}s score={score_b:.3f}"
            )
            return {
                "match_found": True,
                "offset_s": float(pick),
                "best_lag_s": float(best),
                "cam_ref": cam_ref,
                "reason": "dual_candidate",
                "score_a": float(score_a),
                "score_b": float(score_b),
            }
        if va:
            return {
                "match_found": True,
                "offset_s": float(cand_a),
                "best_lag_s": float(wa),
                "cam_ref": cam_ref,
                "reason": "cand_a_only",
                "score_a": float(score_a),
                "score_b": float(score_b),
            }
        if vb:
            return {
                "match_found": True,
                "offset_s": float(cand_b),
                "best_lag_s": float(wb),
                "cam_ref": cam_ref,
                "reason": "cand_b_only",
                "score_a": float(score_a),
                "score_b": float(score_b),
            }
        # No evidence: keep deterministic baseline.
        return {
            "match_found": False,
            "offset_s": float(cand_a),
            "best_lag_s": None,
            "cam_ref": cam_ref,
            "reason": "no_evidence",
        }
    except Exception as exc:
        _emit(f"[otio] clean-to-cam offset solve failed: {exc}")
        return {"match_found": False, "offset_s": None, "best_lag_s": None, "cam_ref": cam_ref, "reason": f"error:{exc}"}


def _estimate_camera_offsets_audalign(camera_paths: list[str], log=None) -> dict[str, float]:
    """Backwards-compatible wrapper; uses generic source-offset estimation."""
    return _estimate_source_offsets_audalign(camera_paths, log=log)


def _write_active_speaker_otio(
    aligned_paths: list[Path],
    switch_monos: list,
    target_sr: int,
    total_samples: int,
    out_dir: Path,
    stamp: str,
    camera_roles: dict[str, str] | None,
    clean_audio_path: str | Path | None,
    timeline_name: str | None = None,
    alignment_debug: bool = True,
    alignment_mode: str = "whisper_anchor_refine",
    transcript_mode: str = "local_faster_whisper",
    language: str = "en",
    asr_required: bool = True,
    alignment_strategy: str = "adaptive_content_anchor",
    max_lag_search_seconds: int = 1200,
    min_anchor_match_confidence: float = 0.55,
    min_anchor_matches_required: int = 2,
    log=None,
) -> tuple[str | None, str | None]:
    def _emit(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass
    if not camera_roles:
        _emit("[otio] camera roles not provided; skipping OTIO.")
        return None, None
    wide = str(camera_roles.get("wide", "")).strip()
    if not wide:
        _emit("[otio] wide camera is required; skipping OTIO.")
        return None, None
    clean_src = str(clean_audio_path or "").strip()
    if not clean_src:
        _emit("[otio] clean synced multichannel source not found; skipping OTIO.")
        return None, None
    clean_src_path = Path(clean_src)
    try:
        import opentimelineio as otio  # type: ignore[import-not-found]
    except Exception as exc:
        _emit(f"[otio] opentimelineio unavailable: {exc}")
        return None, None
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    def _skip_with_reason(reason: str) -> tuple[str | None, str | None]:
        _emit(f"[otio] {reason}")
        out_turns = out_dir / f"CLEAN_Synced_Multichannel_{stamp}_turns.json"
        try:
            payload = {
                "timeline_name": timeline_name or f"ActiveSpeaker_{stamp}",
                "switch_source": "transcript_whisper",
                "transcript_mode": transcript_mode,
                "whisper_model": "small",
                "language": language,
                "asr_required": bool(asr_required),
                "alignment_strategy": alignment_strategy,
                "max_lag_search_seconds": int(max_lag_search_seconds),
                "otio_skip_reason": reason,
            }
            out_turns.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return None, str(out_turns)
        except Exception:
            return None, None

    if len(switch_monos) < 2:
        return _skip_with_reason("need >=2 clean channels for switching; skipping OTIO.")

    scratch_first = str(alignment_strategy or "").strip().lower() == "cam_scratch_only"
    if scratch_first:
        _emit("[otio] alignment strategy: cam_scratch_only (scratch-first sync)")
    else:
        _emit(f"[otio] transcription mode: {transcript_mode} small {language}")
    clean_total_samples = max([int(m.numel()) for m in switch_monos] + [max(1, int(total_samples))])
    clean_duration_s = max(0.0, float(clean_total_samples / max(1, target_sr)))
    # Keep ASR overhead bounded: alignment is audalign-driven, ASR is only a quality anchor.
    window_seconds = 15.0
    max_windows = 1
    anchor_windows: list[dict] = []
    anchor_quality_per_window: list[dict] = []
    anchor_words: list[dict] = []
    if not scratch_first:
        anchor_windows = _select_anchor_windows(clean_duration_s, policy="start_mid_end", window_s=window_seconds, max_windows=max_windows)
        # Prefer speech-rich windows using clean-channel VAD proxy energy.
        try:
            import numpy as _np
            probe = switch_monos[0]
            x = probe.detach().cpu().float().numpy() if hasattr(probe, "detach") else _np.asarray(probe, dtype=_np.float32)
            scored = []
            for w in anchor_windows:
                ws = float(w.get("start_s", 0.0))
                wd = float(w.get("dur_s", window_seconds))
                i0 = max(0, int(round(ws * float(target_sr))))
                i1 = min(int(x.size), max(i0 + 1, int(round((ws + wd) * float(target_sr)))))
                seg = x[i0:i1]
                e = float(_np.sqrt(_np.mean(seg * seg) + 1e-12)) if seg.size else 0.0
                scored.append((e, w))
            anchor_windows = [w for _e, w in sorted(scored, key=lambda t: t[0], reverse=True)]
        except Exception:
            pass
        all_window_words: list[dict] = []
        for w in anchor_windows:
            ws = float(w.get("start_s", 0.0))
            wd = float(w.get("dur_s", window_seconds))
            words_win = _transcribe_window_faster_whisper(
                clean_src_path,
                start_s=ws,
                dur_s=wd,
                language=str(language or "en"),
                model_name="small",
            )
            q = _window_transcript_quality(words_win, min_words=8)
            anchor_quality_per_window.append({
                "label": str(w.get("label", "")),
                "start_s": ws,
                "dur_s": wd,
                **q,
            })
            if q.get("ok"):
                all_window_words.extend(words_win)
        anchor_words = _build_word_anchors(all_window_words, min_word_dur_s=0.08)
        if not anchor_words and asr_required:
            return _skip_with_reason("low_anchor_confidence: no usable word anchors; skipping OTIO.")

    # Build full-length turns from clean audio activity; transcript is used for alignment anchors.
    min_switch_turn_s = 10.0
    analysis = _analyze_speaker_activity(switch_monos, target_sr)
    if not analysis:
        return _skip_with_reason("speaker analysis unavailable; skipping OTIO.")
    turns = _build_speaker_turns(
        analysis["winner_idx"],
        analysis["conf"],
        hop_ms=float(analysis["hop_ms"]),
        n_speakers=max(1, int(len(switch_monos))),
        min_conf=0.55,
        min_dur_s=0.8,
    )
    if not turns:
        turns = [{
            "speaker_idx": 0,
            "start_s": 0.0,
            "end_s": max(0.1, float(clean_duration_s)),
            "avg_conf": 0.0,
            "text": "",
            "is_filtered": False,
        }]

    intro_s = 15.0
    host_inference: dict = {"mode": "fallback_intro_dominance", "ok": False, "details": None}
    # Primary host mapping: first confident keyword phrase in intro transcript.
    intro_words: list[dict] = []
    try:
        intro_words = _transcribe_window_faster_whisper(
            clean_src_path,
            start_s=0.0,
            dur_s=90.0,
            language=str(language or "en"),
            model_name="small",
        )
    except Exception:
        intro_words = []
    kw = _infer_host_from_intro_keywords(intro_words, turns)
    if bool(kw.get("ok")) and (kw.get("host_idx") is not None):
        host_idx = int(kw.get("host_idx"))
        host_inference = {"mode": "keyword_intro", "ok": True, "details": kw.get("match")}
    else:
        host_idx = _infer_host_speaker_idx(turns, intro_probe_s=45.0, fallback_idx=0)
        host_inference = {"mode": "fallback_intro_dominance", "ok": True, "details": {"reason": kw.get("reason", "fallback")}}
    try:
        _emit(f"[otio] inferred host speaker index: {host_idx}")
        if host_inference.get("mode") == "keyword_intro":
            d = host_inference.get("details") or {}
            _emit(
                "[otio] host keyword match: "
                f"heard='{d.get('heard', '')}' target='{d.get('target', '')}' "
                f"time={float(d.get('time_s', 0.0)):.2f}s ratio={float(d.get('ratio', 0.0)):.2f}"
            )
        else:
            _emit("[otio] host keyword match not found; using intro dominance fallback.")
    except Exception:
        pass

    turns = sorted(turns, key=lambda t: float(t.get("start_s", 0.0)))
    speaker_order: list[int] = []
    for t in turns:
        si = int(t.get("speaker_idx", 0))
        if si not in speaker_order:
            speaker_order.append(si)
    guest_primary = next((s for s in speaker_order if s != host_idx), None)

    def _pick_cam(spk: int) -> str:
        if int(spk) == int(host_idx):
            return _select_camera_for_speaker(int(spk), host_idx, wide, camera_roles)
        if guest_primary is not None and int(spk) == int(guest_primary):
            return _select_camera_for_speaker(int(spk), host_idx, wide, camera_roles)
        return wide

    for t in turns:
        t["camera_path"] = _pick_cam(int(t.get("speaker_idx", 0)))
    for t in turns:
        if float(t.get("start_s", 0.0)) < intro_s:
            t["camera_path"] = _select_camera_for_speaker(host_idx, host_idx, wide, camera_roles)
            t["speaker_idx"] = host_idx
    if not turns or float(turns[0].get("start_s", 0.0)) > 0.0:
        turns.insert(0, {
            "speaker_idx": host_idx,
            "start_s": 0.0,
            "end_s": min(intro_s, clean_duration_s),
            "avg_conf": 0.0,
            "text": "",
            "is_filtered": False,
            "camera_path": _select_camera_for_speaker(host_idx, host_idx, wide, camera_roles),
        })

    # Enforce minimum speaker-run threshold for camera change:
    # if the candidate run is >= min_switch_turn_s, switch immediately at run start.
    turns = sorted(turns, key=lambda t: float(t.get("start_s", 0.0)))
    current_cam = str(turns[0].get("camera_path", _select_camera_for_speaker(host_idx, host_idx, wide, camera_roles)))
    thresholded: list[dict] = []
    i = 0
    n_turns = len(turns)
    while i < n_turns:
        t0 = turns[i]
        run_cam = str(t0.get("camera_path", wide) or wide)
        j = i
        run_start = max(0.0, float(turns[i].get("start_s", 0.0)))
        run_end = max(run_start, float(turns[i].get("end_s", run_start)))
        while (j + 1) < n_turns:
            nxt = turns[j + 1]
            c = str(nxt.get("camera_path", wide) or wide)
            if c != run_cam:
                break
            run_end = max(run_end, float(nxt.get("end_s", run_end)))
            j += 1
        run_dur = max(0.0, run_end - run_start)
        if run_cam != current_cam and run_dur >= float(min_switch_turn_s):
            current_cam = run_cam
        for k in range(i, j + 1):
            tk = turns[k]
            s = max(0.0, float(tk.get("start_s", 0.0)))
            e = max(s, float(tk.get("end_s", s)))
            thresholded.append({
                "speaker_idx": int(tk.get("speaker_idx", host_idx)),
                "avg_conf": float(tk.get("avg_conf", 0.0)),
                "text": str(tk.get("text", "") or ""),
                "is_filtered": bool(tk.get("is_filtered", False)),
                "start_s": s,
                "end_s": e,
                "camera_path": current_cam,
            })
        i = j + 1
    turns = thresholded or turns

    # Turns are currently in clean-local time; we normalize to strict timeline
    # segments after global offset solving.
    merged: list[dict] = []

    fps = _probe_video_fps(Path(wide), default_fps=25.0)
    rate = float(max(1.0, fps))
    timeline = otio.schema.Timeline(name=(timeline_name or f"ActiveSpeaker_{stamp}"))
    stack = otio.schema.Stack(name="Video")
    timeline.tracks = stack

    def _cam_key(p: str) -> str:
        try:
            return os.path.normcase(os.path.normpath(str(Path(p).resolve(strict=False))))
        except Exception:
            return os.path.normcase(os.path.normpath(str(p)))

    # Include all assigned camera roles (deduped) so expected camera tracks are
    # always present, then include any segment-resolved camera path variants.
    host_close = str(camera_roles.get("host_closeup", "")).strip()
    guest_close = str(camera_roles.get("guest_closeup", "")).strip()
    host_paths = [str(v).strip() for k, v in camera_roles.items() if str(k).lower().startswith("extra_host") and str(v).strip()]
    guest_paths = [str(v).strip() for k, v in camera_roles.items() if str(k).lower().startswith("extra_guest") and str(v).strip()]
    used_order = [str(wide), guest_close, host_close] + host_paths + guest_paths
    used_order.extend([str(seg.get("camera_path", "") or "") for seg in merged])
    cam_actual_by_key: dict[str, str] = {}
    used_cam_keys: list[str] = []
    for raw in used_order:
        p = str(raw or "").strip()
        if not p:
            continue
        k = _cam_key(p)
        if k in cam_actual_by_key:
            continue
        cam_actual_by_key[k] = p
        used_cam_keys.append(k)

    cams: list[tuple[str, str]] = []
    for cam_key in used_cam_keys:
        cam = cam_actual_by_key[cam_key]
        cams.append((cam_key, cam))
    valid_cams: list[tuple[str, str]] = []
    for cam_key, cam in cams:
        try:
            if Path(cam).exists():
                valid_cams.append((cam_key, cam))
            else:
                _emit(f"[otio] camera source missing, excluding: {cam}")
        except Exception:
            _emit(f"[otio] camera source missing, excluding: {cam}")
    cams = valid_cams
    if not cams:
        _emit("[otio] no valid camera sources available; skipping OTIO.")
        return None, None

    clean_monos_for_tracks = _load_audio_tracks_any(clean_src_path, target_sr=target_sr)
    if len(clean_monos_for_tracks) < 2:
        _emit("[otio] need >=2 clean channels for switching; skipping OTIO.")
        return None, None

    cam_sync_sources = [str(cam) for _k, cam in cams]
    sync_sources = list(cam_sync_sources)
    if (not scratch_first) and clean_src_path:
        sync_sources.append(str(clean_src_path))
    # Adaptive content-anchor matching across broad lag search.
    source_mono: dict[str, object] = {}
    for sp in sync_sources:
        tt = _load_audio_tracks_any(sp, target_sr=target_sr)
        if tt:
            source_mono[sp] = tt[0]
    clean_mono = source_mono.get(str(clean_src_path))
    if clean_mono is None and asr_required:
        return _skip_with_reason("clean source decode failed for content-anchor alignment; skipping OTIO.")

    matches_by_source: dict[str, list[dict]] = {sp: [] for sp in sync_sources}
    match_confidence_per_window: list[dict] = []
    if clean_mono is not None:
        try:
            clean_np = clean_mono.detach().cpu().float().numpy() if hasattr(clean_mono, "detach") else np.asarray(clean_mono, dtype=np.float32)
        except Exception:
            clean_np = None
        if clean_np is not None:
            for aw in anchor_windows:
                ws = float(aw.get("start_s", 0.0))
                wd = float(aw.get("dur_s", window_seconds))
                i0 = max(0, int(round(ws * float(target_sr))))
                i1 = min(int(clean_np.size), max(i0 + 1, int(round((ws + wd) * float(target_sr)))))
                if i1 <= i0:
                    continue
                # Alignment solve is waveform-first; transcript anchors select windows only.
                words_local = []
                cwin = {
                    "start_s": ws,
                    "dur_s": wd,
                    "clean_wave": clean_np[i0:i1],
                    "words_local": words_local,
                }
                wdiag = {"window_start_s": ws, "window_dur_s": wd, "per_source": {}}
                for sp in sync_sources:
                    if sp == str(clean_src_path):
                        continue
                    mono = source_mono.get(sp)
                    if mono is None:
                        continue
                    m = _find_best_content_match(cwin, mono, sr=target_sr, max_lag_s=float(max_lag_search_seconds))
                    matches_by_source.setdefault(sp, []).append(m)
                    wdiag["per_source"][Path(sp).name] = {
                        "ok": bool(m.get("ok")),
                        "confidence": float(m.get("confidence", 0.0)),
                        "peak_ratio": float(m.get("peak_ratio", 0.0)),
                        "residual_s": float(m.get("residual_s", 0.0)),
                        "reason": str(m.get("reason", "")),
                    }
                match_confidence_per_window.append(wdiag)

    solve = _solve_offsets_from_content_matches(matches_by_source)
    # Deterministic source offsets from audalign over cams + clean as one unit.
    coarse_offsets = _estimate_source_offsets_audalign(sync_sources, log=_emit)
    coarse_offsets = {str(sp): float(coarse_offsets.get(str(sp), 0.0)) for sp in sync_sources}
    if str(clean_src_path):
        coarse_offsets[str(clean_src_path)] = 0.0
    # Small local waveform refinement around audalign offsets.
    refined = _refine_offsets_waveform(coarse_offsets, sync_sources, target_sr=target_sr)
    source_offsets_by_path = {str(sp): float(coarse_offsets.get(str(sp), 0.0)) for sp in sync_sources}
    for k, v in (refined.get("offsets", {}) or {}).items():
        if str(k) in source_offsets_by_path:
            source_offsets_by_path[str(k)] = float(v)
    # Re-anchor to clean=0 and preserve signed source offsets.
    clean_anchor = float(source_offsets_by_path.get(str(clean_src_path), 0.0)) if str(clean_src_path) else 0.0
    source_offsets_by_path = {k: float(v - clean_anchor) for k, v in source_offsets_by_path.items()}
    if str(clean_src_path):
        source_offsets_by_path[str(clean_src_path)] = 0.0
    # Scratch-first mode: cams are authoritative base; solve clean offset to that
    # base without shifting any camera/scratch offsets.
    clean_sync_result = {"match_found": False, "offset_s": None, "best_lag_s": None, "cam_ref": None, "reason": "not_attempted"}
    if scratch_first and cam_sync_sources and str(clean_src_path):
        cam_ref = str(cam_sync_sources[0])
        cam_ref_off = float(source_offsets_by_path.get(cam_ref, 0.0))
        clean_sync_result = _estimate_clean_offset_to_cam_base(
            cam_ref_path=cam_ref,
            cam_ref_offset_s=cam_ref_off,
            clean_path=str(clean_src_path),
            target_sr=target_sr,
            log=_emit,
        )
        if bool(clean_sync_result.get("match_found")) and (clean_sync_result.get("offset_s") is not None):
            clean_off = float(clean_sync_result.get("offset_s"))
            source_offsets_by_path[str(clean_src_path)] = clean_off
            _emit(
                "[otio] clean aligned to scratch base: "
                f"{Path(clean_src_path).name}={float(clean_off):.3f}s "
                f"(cam_ref={Path(cam_ref).name}={cam_ref_off:.3f}s)"
            )
            _emit(
                "[otio] clean-to-scratch match found: "
                f"lag={float(clean_sync_result.get('best_lag_s', 0.0)):.3f}s"
            )
        else:
            return _skip_with_reason(
                "clean-to-scratch match not found; cannot place clean audio reliably in scratch-first mode."
            )
    first_anchor_s = float(anchor_words[0].get("start", 0.0)) if anchor_words else 0.0
    speech_onsets_seconds = {str(p): max(0.0, first_anchor_s + float(source_offsets_by_path.get(str(p), 0.0))) for p in sync_sources}
    qcheck = _validate_alignment_quality(
        refined,
        max_median_residual_ms=120.0,
        min_residual_pairs=max(1, len(cam_sync_sources) - 1),
    )
    if scratch_first:
        paircheck = _validate_offsets_pairwise(
            cam_sync_sources,
            source_offsets_by_path,
            target_sr=target_sr,
            max_abs_lag_s=0.40,
        )
    else:
        paircheck = _validate_offsets_against_clean(
            sync_sources,
            source_offsets_by_path,
            clean_path=str(clean_src_path),
            target_sr=target_sr,
            max_abs_lag_s=0.40,
        )
    if asr_required and (not bool(paircheck.get("ok"))):
        reason = (
            "alignment quality failed against clean "
            f"(valid_pairs={int(paircheck.get('valid_pairs', 0))}, "
            f"max_abs_pair_lag_s={float(paircheck.get('max_abs_lag_s', 999.0)):.3f}); skipping OTIO."
        )
        return _skip_with_reason(reason)
    if asr_required and bool(qcheck.get("has_evidence")) and (not bool(qcheck.get("ok"))):
        return _skip_with_reason(
            "alignment residual quality failed "
            f"(median_residual_ms={float(qcheck.get('median_residual_ms', 0.0)):.2f}); skipping OTIO."
        )
    alignment_quality = {
        "coarse_confidence": float(solve.get("confidence", 0.0)),
        "refine_confidence": float(refined.get("confidence", 0.0)),
        "residuals_seconds": dict(refined.get("residuals", {}) or {}),
        "audalign_offsets_seconds": {Path(k).name: float(v) for k, v in (coarse_offsets or {}).items()},
        "median_residual_ms": float(qcheck.get("median_residual_ms", 0.0)),
        "pair_lag_s": dict(paircheck.get("pair_lag_s", {}) or {}),
        "max_abs_pair_lag_s": float(paircheck.get("max_abs_lag_s", 0.0)),
        "quality_gate_passed": bool(paircheck.get("ok")) and (bool(qcheck.get("ok")) or (not bool(qcheck.get("has_evidence", False)))),
        "quality_gate_reason": ("paircheck_then_residual"),
        "paircheck_tested_pairs": int(paircheck.get("tested_pairs", 0)),
        "paircheck_valid_pairs": int(paircheck.get("valid_pairs", 0)),
        "paircheck_has_evidence": bool(paircheck.get("has_evidence", False)),
        "residual_evidence_count": int(qcheck.get("evidence_count", 0)),
        "residual_has_evidence": bool(qcheck.get("has_evidence", False)),
        "scratch_first_mode": bool(scratch_first),
    }
    anchor_pair = {
        "cam1": (Path(cam_sync_sources[0]).name if cam_sync_sources else ""),
        "cam2": (Path(cam_sync_sources[1]).name if len(cam_sync_sources) > 1 else ""),
    }

    cam_offsets_by_path: dict[str, float] = {}
    cam_offsets_by_key: dict[str, float] = {}
    for cam_key, cam in cams:
        off = float(source_offsets_by_path.get(str(cam), 0.0))
        cam_offsets_by_path[str(cam)] = off
        cam_offsets_by_key[cam_key] = off
    clean_offset_s = float(source_offsets_by_path.get(str(clean_src_path), 0.0))

    cam_durations: dict[str, float] = {}
    for cam_key_iter, cam in cams:
        d = _probe_media_duration(Path(cam))
        if d is not None and d > 0:
            cam_durations[str(cam)] = float(d)
    clean_duration_probe = _probe_media_duration(clean_src_path)
    if clean_duration_probe is None or clean_duration_probe <= 0.0:
        clean_duration_probe = clean_duration_s

    if not cam_durations:
        _emit("[otio] could not probe camera durations; skipping OTIO.")
        return None, None

    # Keep full clean timeline; unavailable camera ranges become gaps.
    window_start_s = 0.0
    window_end_s = float(max(0.0, float(clean_offset_s + max(0.0, float(clean_duration_probe)))))
    if window_end_s <= window_start_s + 0.25:
        _emit("[otio] source overlap window too small after alignment; skipping OTIO.")
        return None, None

    valid_abs_onsets: list[float] = []
    for src, onset in speech_onsets_seconds.items():
        if onset is None:
            continue
        try:
            valid_abs_onsets.append(float(source_offsets_by_path.get(str(src), 0.0)) + float(onset))
        except Exception:
            continue
    # Do not cut timeline time: keep full aligned window from 0.
    trim_start_s = 0.0
    timeline_end_s = max(0.0, float(window_end_s - trim_start_s))
    if timeline_end_s <= 0.25:
        _emit("[otio] timeline duration too small after anchor trim; skipping OTIO.")
        return None, None

    turns_global: list[dict] = []
    for t in turns:
        s = max(0.0, float(t.get("start_s", 0.0)))
        e = max(s, float(t.get("end_s", s)))
        turns_global.append({
            **t,
            "start_s": float(clean_offset_s + s),
            "end_s": float(clean_offset_s + e),
        })
    hold_cam = _select_camera_for_speaker(host_idx, host_idx, wide, camera_roles)
    merged_global = _normalize_segments_strict(turns_global, window_start_s, window_end_s, hold_cam)
    merged = []
    for seg in merged_global:
        s = float(seg.get("start_s", 0.0)) - float(trim_start_s)
        e = float(seg.get("end_s", 0.0)) - float(trim_start_s)
        if e <= s:
            continue
        merged.append({**seg, "start_s": max(0.0, s), "end_s": min(timeline_end_s, e)})
    merged = _normalize_segments_strict(merged, 0.0, timeline_end_s, hold_cam)
    # Pre-switch camera by ~2s so incoming shot appears before speech begins.
    merged = _apply_camera_pre_switch(merged, lead_s=2.0, min_seg_s=0.12)
    merged = _normalize_segments_strict(merged, 0.0, timeline_end_s, hold_cam)
    if not merged:
        _emit("[otio] no valid segments after normalization; skipping OTIO.")
        return None, None

    if alignment_debug:
        try:
            pretty = ", ".join(
                [f"{Path(cam).name}={cam_offsets_by_key.get(k, 0.0):.3f}s" for k, cam in cams]
                + [f"{clean_src_path.name}={clean_offset_s:.3f}s"]
            )
            _emit(f"[otio] alignment_mode={alignment_mode}; source offsets: {pretty}")
            _emit(f"[otio] overlap window={window_start_s:.3f}s..{window_end_s:.3f}s, trim_start={trim_start_s:.3f}s, timeline={timeline_end_s:.3f}s")
        except Exception:
            pass

    # Reuse one media reference per camera so NLEs can treat these as shared sources.
    ref_cache: dict[str, object] = {}
    clean_ref_key = "__clean__"
    max_cam_offset = 0.0
    try:
        max_cam_offset = max([0.0] + [float(v) for v in source_offsets_by_path.values()])
    except Exception:
        max_cam_offset = 0.0
    full_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, rate),
        duration=otio.opentime.RationalTime(float(max(1, int(round((timeline_end_s + max_cam_offset) * rate)))), rate),
    )
    ffmpeg_bin = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    video_proxy_dir = out_dir / "_otio_sources" / "video_only_cache"
    try:
        video_proxy_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for cam_key, cam in cams:
        abs_path = os.path.abspath(os.path.normpath(str(cam)))
        resolved_path = abs_path
        try:
            resolved_path = str(Path(cam).resolve(strict=False))
        except Exception:
            resolved_path = abs_path
        proxy_path: Path | None = None
        if ffmpeg_bin:
            try:
                st = Path(abs_path).stat()
                cache_key = f"{Path(cam).stem}_{int(st.st_mtime)}_{int(st.st_size)}"
                proxy_path = video_proxy_dir / f"{cache_key}.mov"
                if not proxy_path.exists():
                    cmd = [
                        ffmpeg_bin,
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        abs_path,
                        "-map",
                        "0:v:0",
                        "-c:v",
                        "copy",
                        "-an",
                        str(proxy_path),
                    ]
                    proc = subprocess.run(cmd, check=False)
                    if proc.returncode != 0 or not proxy_path.exists():
                        proxy_path = None
            except Exception:
                proxy_path = None
        target_path = str(proxy_path) if proxy_path is not None else abs_path
        file_uri = ""
        try:
            file_uri = Path(target_path).as_uri()
        except Exception:
            file_uri = ""
        target_url = file_uri or target_path
        ref_cache[cam_key] = otio.schema.ExternalReference(
            target_url=target_url,
            available_range=full_range,
            metadata={
                "resemble_enhance": {
                    "absolute_path": abs_path,
                    "resolved_path": resolved_path,
                    "file_uri": file_uri,
                    "video_only_proxy": bool(proxy_path is not None),
                    "video_only_proxy_path": (str(proxy_path) if proxy_path is not None else None),
                },
            },
        )
    clean_abs = os.path.abspath(os.path.normpath(str(clean_src_path)))
    clean_uri = ""
    try:
        clean_uri = Path(clean_abs).as_uri()
    except Exception:
        clean_uri = ""
    try:
        import torchaudio
    except Exception as exc:
        _emit(f"[otio] clean channel isolation unavailable (torchaudio import failed): {exc}")
        return None, None
    # Keep parent clean source reference for diagnostics only.
    ref_cache[clean_ref_key] = otio.schema.ExternalReference(
        target_url=(clean_uri or clean_abs),
        available_range=full_range,
    )

    # Split-only edit model: no time removal (no ripple). Every camera track is
    # split at the same boundaries so all sources stay fully synced.
    split_points = {0.0, float(timeline_end_s)}
    for seg in merged:
        try:
            split_points.add(max(0.0, min(float(timeline_end_s), float(seg.get("start_s", 0.0)))))
            split_points.add(max(0.0, min(float(timeline_end_s), float(seg.get("end_s", 0.0)))))
        except Exception:
            continue
    pts = sorted(split_points)
    split_segments: list[dict] = []
    for i in range(max(0, len(pts) - 1)):
        s = float(pts[i])
        e = float(pts[i + 1])
        if e <= s:
            continue
        # Find active speaker-selected camera for metadata at this split.
        sel_cam = wide
        for seg in merged:
            ss = float(seg.get("start_s", 0.0))
            ee = float(seg.get("end_s", ss))
            if s >= ss - 1e-6 and s < ee - 1e-6:
                sel_cam = str(seg.get("camera_path", wide) or wide)
                break
        split_segments.append({"start_s": s, "end_s": e, "selected_cam_key": _cam_key(sel_cam)})

    video_tracks: list[tuple[str, str, object]] = []
    for cam_key, cam in cams:
        tr = otio.schema.Track(name=f"V_{Path(cam).stem}", kind=otio.schema.TrackKind.Video)
        stack.append(tr)
        video_tracks.append((cam_key, cam, tr))
    audio_tracks: list[tuple[str, str, object]] = []
    for cam_key, cam in cams:
        tr = otio.schema.Track(name=f"A_CAM_{Path(cam).stem}", kind=otio.schema.TrackKind.Audio)
        stack.append(tr)
        audio_tracks.append((cam_key, cam, tr))
    clean_audio_tracks: list[tuple[int, object]] = []
    for ci in range(len(clean_monos_for_tracks)):
        tr = otio.schema.Track(name=f"A_CLEAN_{ci + 1}", kind=otio.schema.TrackKind.Audio)
        stack.append(tr)
        clean_audio_tracks.append((ci, tr))

    # Build importer-safe aligned audio proxies with offsets baked in.
    aligned_audio_dir = out_dir / "_otio_sources" / "aligned_audio_cache"
    aligned_audio_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torchaudio
        import torch
    except Exception as exc:
        _emit(f"[otio] aligned audio proxy unavailable: {exc}")
        return None, None

    cam_audio_ref_by_key: dict[str, object] = {}
    cam_audio_proxy_by_key: dict[str, str] = {}

    def _cache_token_for_source(path_like: str | Path, extra: str = "") -> str:
        try:
            p = Path(path_like)
            rp = str(p.resolve(strict=False))
            st = p.stat()
            raw = f"{rp}|{int(st.st_mtime)}|{int(st.st_size)}|{extra}"
        except Exception:
            raw = f"{str(path_like)}|{extra}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    for cam_key, cam, _tr in audio_tracks:
        tracks = _load_audio_tracks_any(cam, target_sr=target_sr)
        if not tracks:
            _emit(f"[otio] camera audio decode failed, excluding audio track: {cam}")
            continue
        mono = tracks[0]
        off_s = float(cam_offsets_by_key.get(cam_key, 0.0))
        off_samples = int(round(off_s * float(target_sr)))
        if off_samples >= 0:
            aligned_global = torch.cat([torch.zeros(off_samples, dtype=mono.dtype), mono], dim=0)
        else:
            drop = min(int(mono.numel()), int(-off_samples))
            aligned_global = mono[drop:] if drop < int(mono.numel()) else mono.new_zeros(1)
        trim_samples = max(0, int(round(float(trim_start_s) * float(target_sr))))
        aligned = aligned_global[trim_samples:] if trim_samples < int(aligned_global.numel()) else aligned_global.new_zeros(1)
        token = _cache_token_for_source(
            cam,
            extra=f"cam|off={off_s:.6f}|trim={float(trim_start_s):.6f}|sr={int(target_sr)}",
        )
        outp = aligned_audio_dir / f"A_CAM_{Path(cam).stem}_{token}.wav"
        if not outp.exists():
            torchaudio.save(str(outp), aligned.unsqueeze(0), int(target_sr))
        else:
            _emit(f"[otio] reuse cached audio proxy: {outp.name}")
        cam_audio_proxy_by_key[cam_key] = str(outp)
        p_abs = os.path.abspath(os.path.normpath(str(outp)))
        p_uri = ""
        try:
            p_uri = Path(p_abs).as_uri()
        except Exception:
            p_uri = ""
        cam_audio_ref_by_key[cam_key] = otio.schema.ExternalReference(
            target_url=(p_uri or p_abs),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, float(target_sr)),
                duration=otio.opentime.RationalTime(float(max(1, int(aligned.numel()))), float(target_sr)),
            ),
            metadata={
                "resemble_enhance": {
                    "aligned_audio_proxy": True,
                    "source_path": str(cam),
                    "offset_seconds_applied": off_s,
                    "trim_start_seconds": float(trim_start_s),
                },
            },
        )

    clean_audio_ref_by_idx: dict[int, object] = {}
    clean_audio_proxy_by_idx: dict[int, str] = {}
    for clean_idx, _tr in clean_audio_tracks:
        mono = clean_monos_for_tracks[clean_idx]
        off_samples = int(round(float(clean_offset_s) * float(target_sr)))
        if off_samples >= 0:
            aligned_global = torch.cat([torch.zeros(off_samples, dtype=mono.dtype), mono], dim=0)
        else:
            drop = min(int(mono.numel()), int(-off_samples))
            aligned_global = mono[drop:] if drop < int(mono.numel()) else mono.new_zeros(1)
        trim_samples = max(0, int(round(float(trim_start_s) * float(target_sr))))
        aligned = aligned_global[trim_samples:] if trim_samples < int(aligned_global.numel()) else aligned_global.new_zeros(1)
        token = _cache_token_for_source(
            clean_src_path,
            extra=(
                f"clean_ch={int(clean_idx)}|off={float(clean_offset_s):.6f}|"
                f"trim={float(trim_start_s):.6f}|sr={int(target_sr)}"
            ),
        )
        outp = aligned_audio_dir / f"A_CLEAN_{clean_idx + 1}_{token}.wav"
        if not outp.exists():
            torchaudio.save(str(outp), aligned.unsqueeze(0), int(target_sr))
        else:
            _emit(f"[otio] reuse cached audio proxy: {outp.name}")
        clean_audio_proxy_by_idx[clean_idx] = str(outp)
        p_abs = os.path.abspath(os.path.normpath(str(outp)))
        p_uri = ""
        try:
            p_uri = Path(p_abs).as_uri()
        except Exception:
            p_uri = ""
        clean_audio_ref_by_idx[clean_idx] = otio.schema.ExternalReference(
            target_url=(p_uri or p_abs),
            available_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, float(target_sr)),
                duration=otio.opentime.RationalTime(float(max(1, int(aligned.numel()))), float(target_sr)),
            ),
            metadata={
                "resemble_enhance": {
                    "aligned_audio_proxy": True,
                    "source_path": str(clean_src_path),
                    "clean_channel_index": int(clean_idx),
                    "offset_seconds_applied": float(clean_offset_s),
                    "trim_start_seconds": float(trim_start_s),
                },
            },
        )

    # Strict audio QA on aligned proxies before OTIO write.
    try:
        qa_paths: dict[str, str | Path] = {}
        if cams:
            qa_paths["cam1"] = cam_audio_proxy_by_key.get(cams[0][0], "")
        if len(cams) > 1:
            qa_paths["cam2"] = cam_audio_proxy_by_key.get(cams[1][0], "")
        if (not scratch_first) and clean_audio_proxy_by_idx:
            qa_paths["clean1"] = clean_audio_proxy_by_idx.get(0, "")
        audio_qc = _qa_aligned_audio_sync(qa_paths, target_sr=target_sr, max_abs_median_lag_s=0.25)
        if not bool(audio_qc.get("ok")) and asr_required:
            return _skip_with_reason(
                f"aligned audio QA failed (max median lag {float(audio_qc.get('max_abs_median_lag_s', 0.0)):.3f}s); skipping OTIO."
            )
    except Exception:
        audio_qc = {"ok": False, "pair_median_lag_s": {}, "max_abs_median_lag_s": 999.0}

    for cam_key, cam, tr in video_tracks:
        cam_dur_s = float(cam_durations.get(str(cam), 0.0) or 0.0)
        for seg in split_segments:
            dur_s = max(0.0, float(seg["end_s"]) - float(seg["start_s"]))
            dur_f = max(1, int(round(dur_s * rate)))
            is_selected = bool(str(seg.get("selected_cam_key", "")) == str(cam_key))
            if is_selected:
                cam_off_s = float(cam_offsets_by_key.get(cam_key, 0.0))
                global_start_s = float(trim_start_s) + float(seg["start_s"])
                src_start_s = float(global_start_s - cam_off_s)
                remain_s = float(dur_s)
                # If source starts before camera media starts, insert a gap first.
                if src_start_s < 0.0 and remain_s > 0.0:
                    pre_gap_s = min(remain_s, float(-src_start_s))
                    pre_gap_f = max(1, int(round(pre_gap_s * rate)))
                    tr.append(
                        otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(0, rate),
                                duration=otio.opentime.RationalTime(float(pre_gap_f), rate),
                            )
                        )
                    )
                    src_start_s = 0.0
                    remain_s = max(0.0, remain_s - pre_gap_s)
                # Clip only available source span; trailing unavailable span becomes gap.
                if remain_s > 0.0:
                    avail_s = remain_s
                    if cam_dur_s > 0.0:
                        avail_s = max(0.0, min(remain_s, cam_dur_s - src_start_s))
                    if avail_s > 1e-6:
                        avail_f = max(1, int(round(avail_s * rate)))
                        clip = otio.schema.Clip(
                            name=Path(cam).stem,
                            media_reference=ref_cache.get(cam_key),
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(float(src_start_s * rate), rate),
                                duration=otio.opentime.RationalTime(float(avail_f), rate),
                            ),
                            metadata={
                                "resemble_enhance": {
                                    "active_selected_camera": True,
                                },
                            },
                        )
                        tr.append(clip)
                        remain_s = max(0.0, remain_s - avail_s)
                if remain_s > 1e-6:
                    post_gap_f = max(1, int(round(remain_s * rate)))
                    tr.append(
                        otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(0, rate),
                                duration=otio.opentime.RationalTime(float(post_gap_f), rate),
                            )
                        )
                    )
            else:
                tr.append(
                    otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(float(dur_f), rate),
                        )
                    )
                )

    # Audio stays continuous in timeline (no split-based cutting).
    timeline_samples = max(1, int(round(float(timeline_end_s) * float(target_sr))))

    def _ref_available_samples(ref_obj) -> int:
        try:
            ar = getattr(ref_obj, "available_range", None)
            if ar is None:
                return timeline_samples
            dur = getattr(ar, "duration", None)
            if dur is None:
                return timeline_samples
            return max(1, int(round(float(dur.value))))
        except Exception:
            return timeline_samples

    for cam_key, cam, tr in audio_tracks:
        aref = cam_audio_ref_by_key.get(cam_key)
        if aref is None:
            continue
        avail = _ref_available_samples(aref)
        use = max(1, min(int(timeline_samples), int(avail)))
        tr.append(
            otio.schema.Clip(
                name=Path(cam).stem,
                media_reference=aref,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0.0, float(target_sr)),
                    duration=otio.opentime.RationalTime(float(use), float(target_sr)),
                ),
                metadata={
                    "resemble_enhance": {
                        "audio_role": "scratch_reference",
                        "monitor_default": "off",
                        "color_hint": "grey",
                    },
                },
            )
        )
        if use < timeline_samples:
            tr.append(
                otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0.0, float(target_sr)),
                        duration=otio.opentime.RationalTime(float(timeline_samples - use), float(target_sr)),
                    )
                )
            )

    for clean_idx, tr in clean_audio_tracks:
        cref = clean_audio_ref_by_idx.get(clean_idx)
        if cref is None:
            continue
        avail = _ref_available_samples(cref)
        use = max(1, min(int(timeline_samples), int(avail)))
        clean_role = "host_clean" if int(clean_idx) == int(host_idx) else "guest_clean"
        clean_color = "pink" if clean_role == "host_clean" else "yellow"
        tr.append(
            otio.schema.Clip(
                name=f"CLEAN_{clean_idx + 1}",
                media_reference=cref,
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(0.0, float(target_sr)),
                    duration=otio.opentime.RationalTime(float(use), float(target_sr)),
                ),
                metadata={
                    "resemble_enhance": {
                        "clean_channel_index": int(clean_idx),
                        "audio_role": clean_role,
                        "color_hint": clean_color,
                    },
                },
            )
        )
        if use < timeline_samples:
            tr.append(
                otio.schema.Gap(
                    source_range=otio.opentime.TimeRange(
                        start_time=otio.opentime.RationalTime(0.0, float(target_sr)),
                        duration=otio.opentime.RationalTime(float(timeline_samples - use), float(target_sr)),
                    )
                )
            )
    _emit(f"[otio] camera references used: {len(used_cam_keys)}")
    _emit(f"[otio] split segments: {len(split_segments)}")

    out_otio = out_dir / f"CLEAN_Synced_Multichannel_{stamp}.otio"
    out_turns = out_dir / f"CLEAN_Synced_Multichannel_{stamp}_turns.json"
    try:
        otio.adapters.write_to_file(timeline, str(out_otio))
    except Exception as exc:
        _emit(f"[otio] write failed: {exc}")
        return None, None
    try:
        payload = {
            "timeline_name": timeline_name or f"ActiveSpeaker_{stamp}",
            "fps": rate,
            "host_speaker_idx": host_idx,
            "host_inference": host_inference,
            "wide": wide,
            "segments": merged,
            "turn_count": len(merged),
            "switch_source": "transcript_whisper",
            "alignment_mode": alignment_mode,
            "alignment_strategy": alignment_strategy,
            "max_lag_search_seconds": int(max_lag_search_seconds),
            "transcript_mode": transcript_mode,
            "whisper_model": "small",
            "language": language,
            "asr_required": bool(asr_required),
            "anchor_windows": anchor_windows,
            "matches_per_source": {Path(k).name: int(v) for k, v in (solve.get("counts", {}) or {}).items()},
            "match_confidence_per_window": match_confidence_per_window,
            "word_anchor_count": int(len(anchor_words)),
            "word_anchors_preview": [
                {
                    "word": str(w.get("word", "") or ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "confidence": float(w.get("confidence", 0.0)),
                }
                for w in anchor_words[:24]
            ],
            "anchor_pair": anchor_pair,
            "speech_onsets_seconds": {Path(k).name: v for k, v in speech_onsets_seconds.items()},
            "coarse_offsets_seconds": {Path(k).name: float(v) for k, v in coarse_offsets.items()},
            "refined_offsets_seconds": {Path(k).name: float(v) for k, v in source_offsets_by_path.items()},
            "resolved_offsets_seconds": {Path(k).name: float(v) for k, v in source_offsets_by_path.items()},
            "alignment_quality": alignment_quality,
            "audio_sync_qc": audio_qc,
            "clean_sync": clean_sync_result,
            "scratch_audio_note": "A_CAM tracks are reference scratch audio; monitor A_CLEAN tracks for primary mix.",
            "alignment_confidence": {
                "coarse": float(solve.get("confidence", 0.0)),
                "refine": float(refined.get("confidence", 0.0)),
            },
            "timeline_window_seconds": {
                "window_start": float(window_start_s),
                "window_end": float(window_end_s),
                "trim_start": float(trim_start_s),
                "timeline_duration": float(timeline_end_s),
            },
            "otio_skip_reason": None,
            "audio_tracks_written": [f"A_CAM_{Path(cam).stem}" for _k, cam in cams] + [f"A_CLEAN_{i + 1}" for i in range(len(clean_monos_for_tracks))],
            "source_offsets_seconds": (
                {Path(cam).name: float(cam_offsets_by_key.get(k, 0.0)) for k, cam in cams}
                | {clean_src_path.name: float(clean_offset_s)}
            ),
            "camera_offsets_seconds": {Path(cam).name: float(cam_offsets_by_key.get(k, 0.0)) for k, cam in cams},
        }
        out_turns.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        out_turns = None
    return str(out_otio), (str(out_turns) if out_turns else None)


def _sync_and_export_multichannel_simple(
    file_paths: list[str],
    prefer_48k: bool = True,
    log=None,
    progress_cb=None,
    wav_only: bool = False,
    use_bw64: bool = True,
    out_base_dir: str | None = None,
    flat_output: bool = False,
    enable_bleed_gate: bool = False,
    write_otio: bool = False,
    otio_camera_roles: dict[str, str] | None = None,
    otio_timeline_name: str | None = None,
    otio_out_dir: str | None = None,
) -> str | None:
    """Simplified Audalign-based alignment and multichannel export.

    Uses Audalign directly on the input files and then assembles a multichannel
    WAV/MOV without extra proxy generation, drift correction, or manual offsets.
    """
    sync_t0 = time.perf_counter()
    import importlib
    from datetime import datetime
    import torchaudio
    import torch

    def _emit(message: str, force_console: bool = False) -> None:
        if log:
            try:
                log(message)
            except Exception:
                pass
        try:
            print(f"[sync] {message}")
        except Exception:
            # Best-effort logging only
            pass

    if not file_paths:
        _emit('No files provided for sync; skipping multichannel export.', force_console=True)
        return None

    try:
        ad = importlib.import_module('audalign')
    except Exception as exc:
        _emit(f'Audalign is not available: {exc}', force_console=True)
        return None

    os.environ.setdefault('TQDM_DISABLE', '1')

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_parent = Path(file_paths[0]).parent

    # Temporary working dir for Audalign output
    tmp_dir = (Path.cwd() / ".enhancer_runs_gui" / "tmp_sync" / stamp)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Final folder should contain ONLY the combined multichannel file.
    if out_base_dir:
        base_dir = Path(out_base_dir)
    else:
        base_dir = first_parent.parent if first_parent.name.startswith('Enhanced_') else first_parent
    if flat_output and out_base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)
        final_dir = base_dir
    else:
        final_dir = base_dir / f"Synced_{stamp}"
        final_dir.mkdir(parents=True, exist_ok=True)

    def _abort(reason: str) -> None:
        _emit(reason, force_console=True)
        try:
            if not (flat_output and out_base_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
        except Exception:
            pass
        return None

    # Simple progress helper: alignment, assemble, container/metadata, cleanup
    total_units = 4
    completed_units = 0

    def step(i_inc: int = 1, msg: str = '') -> None:
        nonlocal completed_units
        completed_units = min(total_units, completed_units + i_inc)
        if progress_cb:
            try:
                progress_cb(completed_units, total_units, msg)
            except Exception:
                pass

    def _audalign_worker_count(n_files: int) -> int:
        # Env override: RESEMBLE_AUDALIGN_WORKERS=0/empty means auto.
        override = os.environ.get("RESEMBLE_AUDALIGN_WORKERS", "").strip()
        if override:
            try:
                forced = int(override)
                if forced > 0:
                    return forced
            except Exception:
                pass
        cpu = os.cpu_count() or 1
        if n_files <= 1 or cpu <= 2:
            return 1
        # Keep one core free for UI/audio I/O; cap to avoid oversubscription.
        return max(1, min(cpu - 1, 12))

    audalign_workers = _audalign_worker_count(len(file_paths))
    _emit(f"Audalign workers: {audalign_workers}")

    def _make_recognizer(name: str):
        try:
            rec_cls = getattr(ad, name)
        except Exception:
            return None
        try:
            rec = rec_cls()
        except Exception:
            return None
        try:
            cfg = getattr(rec, 'config', None)
            if cfg is not None:
                mp_enabled = audalign_workers > 1
                # audalign recognizers vary; set common knobs best-effort.
                setattr(cfg, 'multiprocessing', mp_enabled)
                setattr(cfg, 'num_processors', int(audalign_workers))
                try:
                    setattr(cfg, 'n_jobs', int(audalign_workers))
                except Exception:
                    pass
                try:
                    setattr(cfg, 'workers', int(audalign_workers))
                except Exception:
                    pass
        except Exception:
            pass
        return rec

    primary_rec = _make_recognizer('FingerprintRecognizer')
    fallback_rec = _make_recognizer('CorrelationSpectrogramRecognizer') or _make_recognizer('CorrelationRecognizer')

    def _run_align(rec) -> object | None:
        try:
            align_files = getattr(ad, 'align_files')
        except Exception:
            return None
        kwargs = {}
        if rec is not None:
            kwargs['recognizer'] = rec
        with _suppress_stdout_stderr():
            return align_files(*file_paths, destination_path=str(tmp_dir), **kwargs)

    _emit('Running Audalign alignment...')
    align_t0 = time.perf_counter()
    results = None
    for rec in (primary_rec, fallback_rec, None):
        try:
            results = _run_align(rec)
        except Exception:
            results = None
        if results:
            break

    step(1, 'Audalign alignment')
    _emit(f"Timing: Audalign alignment {_format_seconds(time.perf_counter() - align_t0)}")

    if not results:
        return _abort('Audalign could not align the provided files.')

    # 2) Load aligned files from tmp_dir and build multichannel tensor
    _emit('Assembling multichannel file...')
    assemble_t0 = time.perf_counter()
    produced = sorted(tmp_dir.rglob("*.wav"))
    basenames = [Path(p).name for p in file_paths]
    aligned_paths: list[Path] = []

    def _consume_match(match_fn):
        for idx, cand in enumerate(produced):
            if match_fn(cand):
                return produced.pop(idx)
        return None

    for base in basenames:
        stem = Path(base).stem.lower()
        picked = _consume_match(lambda p, b=base: p.name.lower() == b.lower())
        if picked is None:
            picked = _consume_match(lambda p, s=stem: Path(p).stem.lower() == s)
        if picked is None:
            picked = _consume_match(lambda p, s=stem: s in Path(p).stem.lower())
        if picked is None and produced:
            picked = produced.pop(0)
        if picked is not None:
            aligned_paths.append(picked)

    if not aligned_paths:
        step(1, 'Assemble multichannel')
        return _abort('Audalign did not produce any aligned files.')

    # Load, resample to target SR first, then compute max length to avoid truncation
    resampled: list[tuple[torch.Tensor, int]] = []
    srs: list[int] = []
    for p in aligned_paths:
        wav, sr = torchaudio.load(str(p))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # downmix to mono
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        srs.append(int(sr))
        resampled.append((wav, int(sr)))

    target_sr = 48000 if prefer_48k else (srs[0] if srs else 48000)
    from torchaudio.functional import resample as ta_resample

    monos: list[torch.Tensor] = []
    for wav, sr in resampled:
        mono = wav[0]
        if sr != target_sr:
            mono = ta_resample(mono, orig_freq=sr, new_freq=target_sr)
        monos.append(mono)

    if enable_bleed_gate:
        _emit('Applying bleed gate to reduce cross-mic bleed...')
        try:
            monos = _apply_bleed_gate(monos, target_sr)
        except Exception as exc:
            _emit(f'Bleed gate failed: {exc}', force_console=True)

    max_len = 0
    for mono in monos:
        if mono.size(-1) > max_len:
            max_len = int(mono.size(-1))

    chan_tensors: list[torch.Tensor] = []
    for mono in monos:
        cur_len = mono.size(-1)
        if cur_len < max_len:
            mono = torch.nn.functional.pad(mono, (0, max_len - cur_len))
        chan_tensors.append(mono.unsqueeze(0))

    if not chan_tensors:
        return _abort('No aligned audio could be assembled.')

    multich = torch.cat(chan_tensors, dim=0)
    n_ch = int(multich.size(0))
    out_wav = final_dir / f"CLEAN_Synced_Multichannel_{stamp}.wav"
    multich = _apply_peak_ceiling(multich, ceiling_db=-1.0)
    torchaudio.save(str(out_wav), multich, target_sr)
    if write_otio:
        try:
            _emit("[otio] analyzing speaker activity...")
            otio_dir = final_dir
            if otio_out_dir:
                try:
                    od = Path(otio_out_dir)
                    od.mkdir(parents=True, exist_ok=True)
                    otio_dir = od
                except Exception:
                    otio_dir = final_dir
            switch_monos = _load_audio_tracks_any(out_wav, target_sr=target_sr)
            if len(switch_monos) < 2:
                _emit("[otio] need >=2 clean channels for switching; skipping OTIO.")
                switch_monos = []
            otio_path, turns_path = _write_active_speaker_otio(
                aligned_paths=aligned_paths,
                switch_monos=switch_monos,
                target_sr=target_sr,
                total_samples=max_len,
                out_dir=otio_dir,
                stamp=stamp,
                camera_roles=otio_camera_roles,
                clean_audio_path=str(out_wav),
                timeline_name=otio_timeline_name,
                asr_required=False,
                alignment_strategy="cam_scratch_only",
                log=lambda m: _emit(m),
            )
            if otio_path:
                _emit(f"[otio] timeline written: {otio_path}")
            if turns_path:
                _emit(f"[otio] turns diagnostics: {turns_path}")
        except Exception as exc:
            _emit(f"[otio] generation failed: {exc}", force_console=True)

    step(1, 'Assemble multichannel')
    _emit(f"Timing: assemble multichannel {_format_seconds(time.perf_counter() - assemble_t0)}")

    # Optional container / metadata tweaks (reuse existing helpers)
    container_t0 = time.perf_counter()
    try:
        ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
        if ff and not wav_only:
            out_mov = final_dir / f"CLEAN_Synced_Multichannel_{stamp}.mov"
            if n_ch == 2:
                filt = "[0:a]channelsplit=channel_layout=stereo[L][R]"
                cmd = [
                    ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(out_wav),
                    '-filter_complex', filt,
                    '-map', '[L]', '-c:a:0', 'pcm_s24le', '-ac:a:0', '1',
                    '-map', '[R]', '-c:a:1', 'pcm_s24le', '-ac:a:1', '1',
                    str(out_mov),
                ]
            else:
                parts = [f"[0:a]pan=mono|c0=c{idx}[ch{idx}]" for idx in range(n_ch)]
                filt = ";".join(parts)
                cmd = [ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(out_wav), '-filter_complex', filt]
                for idx in range(n_ch):
                    cmd += ['-map', f'[ch{idx}]', f'-c:a:{idx}', 'pcm_s24le', f'-ac:a:{idx}', '1']
                cmd += [str(out_mov)]
            _emit('Converting to dual-/multi-mono MOV...')
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            except subprocess.TimeoutExpired:
                out_mov = None
            if out_mov and out_mov.exists() and out_mov.stat().st_size > 44:
                try:
                    out_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                out_wav = out_mov
            step(1, 'Convert container')
        else:
            if wav_only:
                _emit('WAV-only export selected; finalizing WAV metadata...')
            bw_ok = False
            if use_bw64:
                _emit('Attempting BW64 (ADM) export...')
                try:
                    if _export_bw64_with_adm(str(out_wav), [Path(p).stem for p in aligned_paths], target_sr):
                        bw_ok = True
                        _emit('BW64 (ADM) export complete.')
                except Exception as _e:
                    bw_ok = False
                    _emit(f'BW64 export failed, falling back to BWF iXML: {_e}', force_console=True)
            if not bw_ok:
                try:
                    _finalize_wav_dual_mono(str(out_wav), [Path(p).stem for p in aligned_paths], target_sr)
                except Exception:
                    pass
            step(1, 'Finalize WAV metadata')
    except Exception:
        # Keep WAV if conversion/finalization fails
        pass
    _emit(f"Timing: container/metadata {_format_seconds(time.perf_counter() - container_t0)}")

    cleanup_t0 = time.perf_counter()
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    step(1, 'Cleanup')
    _emit(f"Timing: sync cleanup {_format_seconds(time.perf_counter() - cleanup_t0)}")
    _emit(f"Timing: sync/export total {_format_seconds(time.perf_counter() - sync_t0)}")

    return str(out_wav)


def _sync_and_export_multichannel(file_paths: list[str], prefer_48k: bool = True, log=None, progress_cb=None, wav_only: bool = False, skip_fine_align: bool = False, force_fine_align: bool = False, force_drift_correction: bool = False, use_bw64: bool = True, out_base_dir: str | None = None, enable_bleed_gate: bool = False, write_otio: bool = False, otio_camera_roles: dict[str, str] | None = None, otio_timeline_name: str | None = None, otio_out_dir: str | None = None) -> str | None:
    """Legacy alignment function retained for backwards compatibility.

    Currently unused by the GUI; kept so older scripts can still import it.
    """
    return _sync_and_export_multichannel_simple(
        file_paths=file_paths,
        prefer_48k=prefer_48k,
        log=log,
        progress_cb=progress_cb,
        wav_only=wav_only,
        use_bw64=use_bw64,
        out_base_dir=out_base_dir,
        enable_bleed_gate=enable_bleed_gate,
        write_otio=write_otio,
        otio_camera_roles=otio_camera_roles,
        otio_timeline_name=otio_timeline_name,
        otio_out_dir=otio_out_dir,
    )


def _estimate_drift_ratio(ref_path: str, other_path: str, sr: int, win_s: float = 20.0) -> float:
    """Estimate relative drift (clock ratio) between other and ref using windowed GCC-PHAT.

    Returns a multiplicative ratio such that resampling other by this ratio compensates the drift.
    A ratio > 1.0 means other is slightly fast and should be slowed down.
    """
    import numpy as np
    try:
        import soundfile as sf
    except Exception:
        import torchaudio
        r, sr_r = torchaudio.load(ref_path)
        o, sr_o = torchaudio.load(other_path)
        ref = r.mean(0).numpy() if r.dim() == 2 else r.squeeze(0).numpy()  # type: ignore[attr-defined]
        oth = o.mean(0).numpy() if o.dim() == 2 else o.squeeze(0).numpy()  # type: ignore[attr-defined]
        sr_r = int(sr_r)
        sr_o = int(sr_o)
    else:
        ref, sr_r = sf.read(ref_path, dtype='float32', always_2d=False)
        oth, sr_o = sf.read(other_path, dtype='float32', always_2d=False)
        if ref.ndim > 1:
            ref = ref.mean(axis=1)
        if oth.ndim > 1:
            oth = oth.mean(axis=1)
    if sr_r != sr or sr_o != sr:
        # Proxies should already be at sr
        sr = sr
    n = min(len(ref), len(oth))
    ref = ref[:n]
    oth = oth[:n]
    win = int(win_s * sr)
    if n < 3 * win:
        # Use two windows if too short
        starts = [0, max(0, n - win)]
    else:
        starts = [0, (n - win) // 2, n - win]
    ts = []
    lags = []
    for s in starts:
        rseg = ref[s:s+win]
        oseg = oth[s:s+win]
        lag = _gcc_phat_lag(rseg, oseg)
        ts.append(s / sr)
        lags.append(lag)
    if len(ts) < 2:
        return 1.0
    # Fit lag(samples) = a * t(sec) + b, guard invalids
    try:
        xt = np.asarray(ts, dtype=float)
        yl = np.asarray(lags, dtype=float)
        if not np.isfinite(xt).all() or not np.isfinite(yl).all() or np.unique(xt).size < 2:
            return 1.0
        a, b = np.polyfit(xt, yl, 1)
        if not np.isfinite(a):
            return 1.0
    except Exception:
        return 1.0
    # Drift ratio: resample other by (1 - a/sr)
    ratio = 1.0 - (a / float(sr))
    return float(ratio)


def _gcc_phat_lag(x, y) -> int:
    import numpy as np
    lx = int(len(x) or 0)
    ly = int(len(y) or 0)
    nsum = lx + ly
    if nsum <= 0:
        return 0
    try:
        n = int(2 ** np.ceil(np.log2(nsum)))
        n = max(n, 2)
    except Exception:
        n = max(2, nsum)
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)
    denom = np.abs(R) + 1e-12
    R /= denom
    cc = np.fft.irfft(R, n=n)
    # Shift zero-lag to center
    cc = np.concatenate((cc[-(n//2):], cc[:(n//2)]))
    max_idx = int(np.argmax(np.abs(cc)))
    lag = max_idx - (n // 2)
    return int(lag)


def _resample_with_ratio(src_path: str, dst_path: str, ratio: float) -> None:
    """Resample audio by an arbitrary ratio using polyphase filter, preserving sample rate metadata.

    Writes to dst_path with the same sample rate as input, effectively time-stretching.
    """
    import numpy as np
    import soundfile as sf
    from fractions import Fraction
    from scipy.signal import resample_poly

    y, sr = sf.read(src_path, dtype='float32', always_2d=False)
    was_mono = (y.ndim == 1)
    if y.ndim == 1:
        y = y[:, None]
    # Rational approximation for ratio
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    # Apply on each channel
    ys = []
    for c in range(y.shape[1]):
        ys.append(resample_poly(y[:, c], up, down))
    y2 = np.stack(ys, axis=1)
    if was_mono:
        y2 = y2[:, 0]
    sf.write(dst_path, y2, sr)


@contextmanager
def _suppress_stdout_stderr():
    """Temporarily suppress stdout/stderr (used to quiet audalign progress)."""
    old_out, old_err = _sys.stdout, _sys.stderr
    try:
        _sys.stdout = _io.StringIO()
        _sys.stderr = _io.StringIO()
        yield
    finally:
        _sys.stdout = old_out
        _sys.stderr = old_err


def _finalize_wav_dual_mono(wav_path: str, channel_names: list[str], sr: int) -> None:
    """Ensure a single WAV file behaves as dual-/multi-mono in NLEs.

    - Clears channel mask (if ffmpeg available) so it's not treated as interleaved stereo
    - Injects an iXML chunk with channel names
    """
    import struct
    import shutil as _sh
    ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
    tmp = wav_path + ".tmp.wav"
    try:
        if ff:
            cmd = [ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', wav_path, '-c', 'copy', '-write_channel_mask', '0', tmp]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            if os.path.exists(tmp) and os.path.getsize(tmp) > 44:
                try:
                    os.replace(tmp, wav_path)
                except Exception:
                    _sh.copyfile(tmp, wav_path)
                    os.remove(tmp)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    # Inject iXML chunk with channel names
    try:
        with open(wav_path, 'rb+') as f:
            data = f.read()
            # RIFF header
            if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
                return
            # Build simple iXML content
            def _esc(s: str) -> str:
                return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            tracks = []
            for i, name in enumerate(channel_names, start=1):
                tracks.append(f"<TRACK><NAME>{_esc(name)}</NAME><CHANNEL_INDEX>{i}</CHANNEL_INDEX></TRACK>")
            track_xml = "".join(tracks)
            bwf = (
                f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                f"<BWFXML><IXML_VERSION>1.5</IXML_VERSION>"
                f"<SPEED><SAMPLE_RATE>{sr}</SAMPLE_RATE></SPEED>"
                f"<TRACK_LIST>{track_xml}</TRACK_LIST></BWFXML>"
            ).encode('utf-8')
            # Build iXML chunk
            chunk_id = b'iXML'
            chunk_size = len(bwf)
            pad = b'' if (chunk_size % 2 == 0) else b'\x00'
            new_chunk = chunk_id + struct.pack('<I', chunk_size) + bwf + pad

            # Append at end and fix RIFF size
            f.seek(0, os.SEEK_END)
            f.write(new_chunk)
            riff_size = len(data) - 8 + len(new_chunk)  # size excludes 'RIFF' and size field
            f.seek(4)
            f.write(struct.pack('<I', riff_size))
    except Exception:
        pass


def _export_bw64_with_adm(wav_path: str, channel_names: list[str], sr: int) -> bool:
    """Rewrite WAV as BW64 (RF64) using bw64 if available.

    Minimal implementation: wrap existing PCM data into BW64 and replace the file.
    Returns True on success, False on failure.
    """
    try:
        import soundfile as sf
        bw64_mod = importlib.import_module("bw64")
        write_bw64 = getattr(bw64_mod, "write_bw64", None)
        if write_bw64 is None:
            return False
    except Exception:
        return False

    out_tmp = wav_path + '.bw64'
    try:
        data, samplerate = sf.read(wav_path, always_2d=True)
        write_bw64(out_tmp, data, samplerate)
        os.replace(out_tmp, wav_path)
        return True
    except Exception:
        try:
            if os.path.exists(out_tmp):
                os.remove(out_tmp)
        except Exception:
            pass
        return False


def _seam_smooth_files(paths: list[str], progress_cb=None) -> None:
    """Scan for likely seam spikes and apply localized crossfade + micro-shift.

    Conservative: limits fixes per file; operates per channel independently.
    """
    import torchaudio
    import torch
    import math

    def _smooth_wave(wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav: [C, T]
        C, T = (wav.size(0), wav.size(1)) if wav.dim() == 2 else (1, wav.size(0))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        out = wav.clone()
        max_fixes = 6
        win_ms = 50.0
        win = max(256, int(sr * (win_ms / 1000.0)))
        cross = min(1024, win // 2)
        lag_max = max(1, int(sr * 0.01))  # 10 ms
        guard_start_s = 1.0  # never modify within the first second to protect initial loud onsets
        guard_start = int(sr * guard_start_s)
        for c in range(C):
            x = out[c]
            diff = torch.abs(x[1:] - x[:-1])
            if diff.numel() < 1000:
                continue
            mu = torch.mean(diff)
            sd = torch.std(diff)
            thr = float(mu + 6.0 * sd)
            idxs = (diff > thr).nonzero(as_tuple=False).flatten().tolist()
            # Space out fixes by window length
            filtered = []
            last = -10**9
            for i in idxs:
                if i < guard_start:
                    continue
                if i - last >= win:
                    filtered.append(i)
                    last = i
                if len(filtered) >= max_fixes:
                    break
            for i in filtered:
                a = max(0, i - win // 2)
                b = min(T, i + win // 2)
                if b - a < cross * 2 + 4:
                    continue
                left = x[a:i]
                right = x[i:b]
                lt = left[-cross:]
                rt = right[:cross]
                # find micro lag to maximize correlation
                best_lag = 0
                best_val = -1e18
                for lag in range(-lag_max, lag_max + 1):
                    if lag >= 0:
                        rseg = rt[lag:]
                        lseg = lt[: rseg.numel()]
                    else:
                        lseg = lt[-lag:]
                        rseg = rt[: lseg.numel()]
                    if lseg.numel() < cross // 4:
                        continue
                    val = torch.dot(lseg, rseg)
                    if float(val) > best_val:
                        best_val = float(val)
                        best_lag = lag
                # shift right by best_lag within [a,b]
                right_shift = right
                if best_lag > 0:
                    pad = torch.zeros(best_lag, dtype=right.dtype)
                    right_shift = torch.cat([pad, right[:-best_lag]], dim=0)
                elif best_lag < 0:
                    k = -best_lag
                    right_shift = torch.cat([right[k:], torch.zeros(k, dtype=right.dtype)], dim=0)
                # equal-power crossfade over [a,b]
                segL = left.numel()
                segR = right_shift.numel()
                n = b - a
                # cosine-squared fades across the whole local window
                w = torch.linspace(0, math.pi, steps=n)
                fade_out = (0.5 * (1 + torch.cos(w))).to(x.dtype)  # 1..0
                fade_in = 1.0 - fade_out
                new = torch.zeros(n, dtype=x.dtype)
                Lslice = min(n, segL)
                Rslice = min(n, segR)
                new[:Lslice] += left[-Lslice:] * fade_out[:Lslice]
                new[-Rslice:] += right_shift[:Rslice] * fade_in[-Rslice:]
                x[a:b] = new
            out[c] = x
        return out

    total = max(1, len(paths))
    for i, p in enumerate(paths, start=1):
        try:
            wav, sr = torchaudio.load(str(p))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            fixed = _smooth_wave(wav, sr)
            fixed = _apply_peak_ceiling(fixed, ceiling_db=-1.0)
            torchaudio.save(str(p), fixed, sr)
        except Exception:
            pass
        if progress_cb:
            progress_cb(i, total, f'Seam smoothing {i}/{total}')

    


def _postprocess_level_shape(paths: list[str], target_rms_db: float = -16.0, max_boost_db: float = 8.0, progress_cb=None) -> None:
    """Raise average level with presence lift and a soft limiter."""
    import torchaudio
    import torch
    from torchaudio.functional import highpass_biquad, equalizer_biquad

    target_rms = 10 ** (target_rms_db / 20.0)
    max_boost = 10 ** (max_boost_db / 20.0)
    ceiling_db = -1.0
    ceiling = 10 ** (ceiling_db / 20.0)

    def _presence_shaper(wav, sr):
        try:
            shaped = highpass_biquad(wav, sr, cutoff_freq=30.0, Q=0.707)
            shaped = equalizer_biquad(shaped, sr, center_freq=3200.0, gain=2.0, Q=0.9)
            shaped = equalizer_biquad(shaped, sr, center_freq=7500.0, gain=1.5, Q=0.8)
            return shaped
        except Exception:
            return wav

    def _soft_limiter_tensor(wav):
        thr = 10 ** (-1.2 / 20.0)  # closer to ceiling for more loudness
        knee = 0.3  # amount of transition (linear)
        absw = torch.abs(wav)
        over = absw > thr
        if not torch.any(over):
            return wav
        out = wav.clone()
        excess = absw[over] - thr
        comp = thr + (excess / (1.0 + (excess / max(knee, 1e-6))))
        out[over] = torch.sign(out[over]) * torch.clamp(comp, max=0.999)
        return out

    total = max(1, len(paths))
    for idx, path in enumerate(paths, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, f'Loudness {idx}/{total}')
        wav, sr = torchaudio.load(str(path))
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        wav = _presence_shaper(wav, sr)

        mono = wav.mean(0) if wav.dim() == 2 else wav
        rms = float(torch.sqrt(torch.mean(mono * mono) + 1e-12))
        peak = float(torch.max(torch.abs(wav)))
        gain_rms = target_rms / max(rms, 1e-9) if rms > 0 else 1.0
        gain_peak = ceiling / max(peak, 1e-9) if peak > 0 else max_boost
        gain = min(gain_rms, gain_peak, max_boost)
        wav = wav * float(gain)

        wav = _soft_limiter_tensor(wav)
        wav = _apply_peak_ceiling(wav, ceiling_db=ceiling_db)

        if progress_cb:
            progress_cb(idx, total, f'Loudness {idx}/{total}')

        torchaudio.save(str(path), wav, sr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _dispatch_frozen_module_invocation()
    _mutex_handle = _acquire_single_instance_mutex(_SINGLE_INSTANCE_MUTEX_NAME)
    if _mutex_handle is None:
        _show_single_instance_notice()
        sys.exit(0)
    try:
        app = App()
        app.mainloop()
    finally:
        try:
            _terminate_live_subprocesses(timeout_s=0.5)
        except Exception:
            pass
        _release_single_instance_mutex(_mutex_handle)

