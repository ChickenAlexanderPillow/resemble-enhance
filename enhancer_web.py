import os
import shutil
import tempfile
import time
import logging
from pathlib import Path

import gradio as gr
from typing import Any
import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample as ta_resample

from resemble_enhance.enhancer.inference import denoise

# ----- Logging / Env -----
os.environ.setdefault("RESEMBLE_PROGRESS", "1")  # surface model progress in stdout
_log = logging.getLogger("resemble_web")
if not _log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ----- Helpers -----

def _which_ffmpeg() -> str | None:
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def _apply_peak_ceiling(wav_t: torch.Tensor, ceiling_db: float = -1.0) -> torch.Tensor:
    try:
        ceiling = 10 ** (ceiling_db / 20.0)
        peak = float(wav_t.abs().max().item())
        if peak > ceiling and peak > 0:
            return wav_t * (ceiling / peak)
    except Exception:
        pass
    return wav_t


def _apply_gain_db(wav_t: torch.Tensor, gain_db: float = 0.0) -> torch.Tensor:
    try:
        g = 10 ** (float(gain_db) / 20.0)
        return wav_t * g
    except Exception:
        return wav_t


def _safe_pretranscode(src: str, start: float | None, dur: float | None) -> str:
    ff = _which_ffmpeg()
    dst_dir = Path(tempfile.mkdtemp(prefix="resem_prev_"))
    dst = dst_dir / (Path(src).stem + ".wav")
    try:
        if ff:
            cmd = [ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            if start is not None:
                cmd += ["-ss", str(max(0.0, float(start)))]
            if dur is not None:
                cmd += ["-t", str(max(0.1, float(dur)))]
            cmd += ["-i", str(src), "-c:a", "pcm_s16le", str(dst)]
            import subprocess as sp

            sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        else:
            wav, sr = torchaudio.load(str(src))
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(0)
            else:
                wav = wav.squeeze(0)
            a = int(max(0, (start or 0) * sr))
            b = int(a + max(0.1, (dur or 5.0)) * sr)
            b = min(b, wav.shape[-1])
            seg = wav[a:b]
            torchaudio.save(str(dst), seg.unsqueeze(0), sr)
        if dst.exists() and dst.stat().st_size > 44:
            return str(dst)
    except Exception:
        pass
    return src


def _preview_one(
    use_upload: bool,
    file_sel: Any | None,
    start: float,
    dur: float,
    device: str,
    seam_safe: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    wet: float,
    pretranscode: bool,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    path: str | None = None
    if use_upload and file_sel is not None:
        try:
            path = getattr(file_sel, "name", None) or (str(file_sel) if file_sel is not None else None)
        except Exception:
            path = None
    if not path:
        return None, "Please upload a file."

    try:
        _log.info(
            "Preview start: path=%s start=%.3fs dur=%.3fs device=%s seam=%s chunk=%.1fs overlap=%.1fs wet=%.2f pretx=%s",
            path,
            start,
            dur,
            device,
            seam_safe,
            chunk_seconds,
            overlap_seconds,
            wet,
            pretranscode,
        )
        t0 = time.perf_counter()
        src = _safe_pretranscode(path, start, dur) if pretranscode else path
        wav, sr = torchaudio.load(str(src))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
        # If we didn't pre-slice, slice here safely
        if not pretranscode:
            total = int(wav.shape[-1])
            a = int(max(0, start * sr))
            b = min(total, a + int(max(0.1, dur) * sr))
            if b <= a:
                b = min(total, a + 1)
            wav = wav[a:b]
        if wav.numel() <= 0:
            return None, "Empty segment after slicing."

        kwargs = dict(
            chunk_seconds=float(max(1.0, chunk_seconds)),
            overlap_seconds=float(max(0.0, min(overlap_seconds, chunk_seconds / 4.0))),
            align_max_shift_ratio=0.05 if seam_safe else 0.25,
            align_disable=not seam_safe,
        )
        try:
            hwav, model_sr = denoise(dwav=wav, sr=sr, device=device, run_dir=None, **kwargs)
        except Exception as e:
            _log.warning("Preview CUDA failed (%s); retrying on CPU", e)
            hwav, model_sr = denoise(dwav=wav, sr=sr, device="cpu", run_dir=None, **kwargs)

        # Wet/dry and peak ceiling
        wet = float(max(0.0, min(1.0, wet)))
        base = wav
        if model_sr != sr:
            base = ta_resample(base, orig_freq=sr, new_freq=model_sr)
        if wet < 1.0:
            hwav = wet * hwav + (1.0 - wet) * base
        hwav = _apply_peak_ceiling(hwav, ceiling_db=-1.0)

        y = hwav.detach().cpu().numpy().astype(np.float32)
        _log.info("Preview done: sr=%d samples=%d elapsed=%.3fs", int(model_sr), y.shape[-1], time.perf_counter() - t0)
        return (int(model_sr), y), "OK"
    except Exception as e:  # noqa: BLE001
        _log.exception("Preview error")
        return None, f"Error: {e}"


def _export_batch(
    files: list[Any] | None,
    device: str,
    seam_safe: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    wet: float,
    pretranscode: bool,
    gain_db: float = 0.0,
) -> tuple[list[str], str]:
    if not files:
        return [], "Upload one or more files to export."
    out_paths: list[str] = []
    msgs: list[str] = []
    for f in files:
        try:
            src = getattr(f, "name", None) or str(f)
            _log.info("Export start: %s", src)
            t0 = time.perf_counter()
            if pretranscode:
                src = _safe_pretranscode(src, None, None)
            wav, sr = torchaudio.load(str(src))
            if wav.dim() == 2 and wav.size(0) > 1:
                wav = wav.mean(0)
            else:
                wav = wav.squeeze(0)
            kwargs = dict(
                chunk_seconds=float(max(1.0, chunk_seconds)),
                overlap_seconds=float(max(0.0, min(overlap_seconds, chunk_seconds / 4.0))),
                align_max_shift_ratio=0.05 if seam_safe else 0.25,
                align_disable=not seam_safe,
            )
            try:
                hwav, model_sr = denoise(dwav=wav, sr=sr, device=device, run_dir=None, **kwargs)
            except Exception:
                hwav, model_sr = denoise(dwav=wav, sr=sr, device="cpu", run_dir=None, **kwargs)
            wet = float(max(0.0, min(1.0, wet)))
            base = wav
            if model_sr != sr:
                base = ta_resample(base, orig_freq=sr, new_freq=model_sr)
            if wet < 1.0:
                hwav = wet * hwav + (1.0 - wet) * base
            if abs(float(gain_db)) > 1e-6:
                hwav = _apply_gain_db(hwav, float(gain_db))
            hwav = _apply_peak_ceiling(hwav, ceiling_db=-1.0)
            # Save to temp and add to list
            tmpd = Path(tempfile.mkdtemp(prefix="resem_out_"))
            outp = tmpd / (Path(src).stem + "_ENH.wav")
            torchaudio.save(str(outp), hwav.unsqueeze(0), int(model_sr))
            out_paths.append(str(outp))
            msgs.append(f"OK: {Path(src).name}")
            _log.info(
                "Export done: %s -> %s sr=%d samples=%d elapsed=%.3fs",
                Path(src).name,
                outp.name,
                int(model_sr),
                int(hwav.shape[-1]),
                time.perf_counter() - t0,
            )
        except Exception as e:  # noqa: BLE001
            try:
                nm = Path(getattr(f, 'name', str(f))).name
            except Exception:
                nm = 'file'
            msgs.append(f"Error: {nm} -> {e}")
            _log.exception("Export error on %s", nm)
    return out_paths, "\n".join(msgs)


def _export_and_combine(
    files: list[Any] | None,
    device: str,
    seam_safe: bool,
    chunk_seconds: float,
    overlap_seconds: float,
    wet: float,
    pretranscode: bool,
    prefer_48k: bool,
    skip_fine_align: bool,
    gain_db: float,
    batch_by_folder: bool,
) -> tuple[list[str], str]:
    """Process all uploaded files, then align and export a single multichannel WAV.

    Returns (combined_file_path, message).
    """
    if not files or len(files) < 1:
        return [], "Upload two or more files to combine."
    # group files if requested
    groups: dict[str, list[Any]] = {}
    if batch_by_folder:
        for f in files:
            try:
                p = Path(getattr(f, 'name', str(f)))
            except Exception:
                p = Path(str(f))
            groups.setdefault(str(p.parent), []).append(f)
    else:
        groups["ALL"] = list(files)
    all_outputs: list[str] = []
    msgs: list[str] = []
    # 1) Enhance and 2) Combine per group
    try:
        from enhancer_gui import _sync_and_export_multichannel  # reuse exact logic
        from enhancer_gui import _postprocess_level_brighten  # level + brighten pass
    except Exception as e:  # noqa: BLE001
        _log.exception("Failed to import GUI sync logic")
        return [], f"Sync not available: {e}"
    try:
        for grp, gfiles in groups.items():
            # enhance this group
            out_paths, emsg = _export_batch(gfiles, device, seam_safe, chunk_seconds, overlap_seconds, wet, pretranscode, gain_db)
            if not out_paths:
                msgs.append(f"{Path(grp).name}: {emsg}")
                continue
            comb_wav = _sync_and_export_multichannel(
                out_paths,
                prefer_48k=bool(prefer_48k),
                log=lambda s, g=grp: _log.info("[sync %s] %s", Path(g).name, s),
                progress_cb=None,
                wav_only=True,
                skip_fine_align=bool(skip_fine_align),
                use_bw64=True,
                out_base_dir=str(Path(list(out_paths)[0]).parent),
            )
            if not (comb_wav and Path(comb_wav).exists() and Path(comb_wav).stat().st_size > 44):
                msgs.append(f"{Path(grp).name}: Sync failed.")
                continue
            # postprocess combined wav (level + brighten), then apply extra gain if requested
            try:
                _postprocess_level_brighten([str(comb_wav)])
            except Exception as e:
                _log.warning("Postprocess on combined WAV failed: %s", e)
            try:
                if abs(float(gain_db)) > 1e-6:
                    wav, sr = torchaudio.load(str(comb_wav))
                    wav = _apply_gain_db(wav, float(gain_db))
                    wav = _apply_peak_ceiling(wav, ceiling_db=-1.0)
                    torchaudio.save(str(comb_wav), wav, sr)
            except Exception:
                pass
            mov = _convert_wav_to_dualmono_mov(str(comb_wav))
            if mov and Path(mov).exists() and Path(mov).stat().st_size > 44:
                try:
                    Path(comb_wav).unlink(missing_ok=True)
                except Exception:
                    pass
                all_outputs.append(str(mov))
            else:
                all_outputs.append(str(comb_wav))
        if not all_outputs:
            return [], ("\n".join(msgs) or "No outputs produced.")
        return all_outputs, ("\n".join(msgs) or "OK")
    except Exception as e:  # noqa: BLE001
        _log.exception("Sync/Combine error")
        return [], f"Error during sync/combine: {e}"


def _convert_wav_to_dualmono_mov(wav_path: str) -> str | None:
    """Convert a multichannel WAV into a MOV with one mono stream per channel (dual-/multi-mono).

    Returns the MOV path on success, or None on failure.
    """
    try:
        ff = _which_ffmpeg()
        if not ff:
            return None
        info = torchaudio.info(wav_path)
        n_ch = int(getattr(info, 'num_channels', 0) or 0)
        if n_ch <= 0:
            # Fallback: assume stereo
            n_ch = 2
        out_mov = str(Path(wav_path).with_suffix('.mov'))
        if n_ch == 2:
            filt = "[0:a]channelsplit=channel_layout=stereo[L][R]"
            cmd = [
                ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', wav_path,
                '-filter_complex', filt,
                '-map', '[L]', '-c:a:0', 'pcm_s24le', '-ac:a:0', '1',
                '-map', '[R]', '-c:a:1', 'pcm_s24le', '-ac:a:1', '1',
                out_mov,
            ]
        else:
            parts = [f"[0:a]pan=mono|c0=c{idx}[ch{idx}]" for idx in range(n_ch)]
            filt = ";".join(parts)
            cmd = [ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', wav_path, '-filter_complex', filt]
            for idx in range(n_ch):
                cmd += ['-map', f'[ch{idx}]', f'-c:a:{idx}', 'pcm_s24le', f'-ac:a:{idx}', '1']
            cmd += [out_mov]
        import subprocess as sp
        sp.run(cmd, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        if Path(out_mov).exists() and Path(out_mov).stat().st_size > 44:
            return out_mov
        return None
    except Exception:
        return None


# ----- UI -----

with gr.Blocks(title="Resemble Enhance - Web", css=".gradio-container {max-width: 1280px !important}", analytics_enabled=False) as demo:
    gr.Markdown("## Resemble Enhance - Web")
    files_state = gr.State([])
    groups_state = gr.State({})  # maps folder -> list[str]
    with gr.Tabs():
        with gr.TabItem("Preview"):
            with gr.Row():
                with gr.Column(scale=2, min_width=380):
                    gr.Markdown("Drag & drop or browse to upload audio files, or zip folders.")
                    up = gr.File(label="Upload audio (wav/mp3)", file_count="multiple", file_types=["audio"], type="filepath")
                    up_zip = gr.File(label="Upload folder(s) as ZIP", file_count="multiple", file_types=[".zip"], type="filepath")
                    upload_msg = gr.Markdown()
                    file_pick = gr.Radio(choices=[], label="Choose uploaded file", interactive=True)
                    group_summary = gr.Markdown()
                    gr.Markdown("### Settings")
                    device = gr.Radio(["cuda", "cpu"], value="cuda", label="Device", interactive=True)
                    with gr.Row():
                        seam = gr.Checkbox(value=True, label="Seam-safe joins", interactive=True)
                        pretx = gr.Checkbox(value=False, label="Pre-transcode (ffmpeg)", interactive=True)
                    chunk = gr.Slider(7, 3600, value=60, step=1, label="Chunk size (s)", interactive=True)
                    overlap = gr.Slider(0, 8, value=4, step=0.5, label="Overlap (s)", interactive=True)
                    wet = gr.Slider(0, 1, value=1.0, step=0.05, label="Denoise mix (0..1)", interactive=True)
                with gr.Column(scale=3, min_width=480):
                    start = gr.Slider(0, 600, value=0, step=0.1, label="Start (s)", interactive=True)
                    dur = gr.Slider(0.5, 180, value=5, step=0.1, label="Duration (s)", interactive=True)
                    prev_btn = gr.Button("Render Preview", variant="primary")
                    audio = gr.Audio(label="Preview Audio", interactive=False)
                    prev_msg = gr.Markdown()

        with gr.TabItem("Export"):
            gr.Markdown("Process all uploaded files with the same settings as Preview.")
            with gr.Row():
                with gr.Column(scale=2, min_width=380):
                    gr.Markdown("Uses the same uploaded files and settings from Preview.")
                    exp_btn = gr.Button("Export All (Preview Pipeline)", variant="primary")
                    gr.Markdown("\nSync and combine all processed files into a single multichannel MOV (dual-/multi-mono), like the GUI.")
                    prefer_48k = gr.Checkbox(value=True, label="Camera Sync (48 kHz)")
                    skip_fine = gr.Checkbox(value=True, label="Skip fine alignment")
                    batch_folders = gr.Checkbox(value=True, label="Batch by folder (group files per folder)")
                    gain = gr.Slider(-6, 6, value=2.0, step=0.5, label="Output gain (dB)")
                    exp_sync_btn = gr.Button("Export Synced Multichannel", variant="secondary")
                with gr.Column(scale=3, min_width=480):
                    out_files = gr.Files(label="Downloads")
                    out_sync = gr.Files(label="Synced Multichannel (MOV/WAV)")
                    out_msg = gr.Markdown()

    # Ingest uploads with per-file progress and populate file picker
    def _summarize_groups(files: list[Any]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for f in files:
            try:
                p = Path(str(getattr(f, 'name', str(f))))
            except Exception:
                p = Path(str(f))
            groups.setdefault(p.parent.name or str(p.parent), []).append(str(p))
        return groups

    def _format_group_summary(groups: dict[str, list[str]]) -> str:
        if not groups:
            return "No files."
        lines = ["### Queued Files (by folder)"]
        for g, items in sorted(groups.items()):
            lines.append(f"- {g} ({len(items)})")
        return "\n".join(lines)

    def _ingest_files(new_files, cur_files):
        files = list(cur_files or [])
        if new_files:
            files.extend(list(new_files))
        if not files:
            return [], {}, gr.update(choices=[], value=None), "No files uploaded.", "No files."
        names = [Path(str(getattr(f, 'name', str(f)))).name for f in files]
        groups = _summarize_groups(files)
        _log.info("Upload change: total %d file(s) - %s", len(files), ", ".join(names))
        return files, groups, gr.update(choices=names, value=names[0] if names else None), f"Added {len(new_files or [])} file(s).", _format_group_summary(groups)

    up.change(_ingest_files, inputs=[up, files_state], outputs=[files_state, groups_state, file_pick, upload_msg, group_summary])

    def _safe_extract_zip(zp: str) -> list[str]:
        import zipfile, time
        dest = Path(".web_uploads") / ("zip_" + str(int(time.time()*1000))) / Path(zp).stem
        dest.mkdir(parents=True, exist_ok=True)
        out: list[str] = []
        try:
            with zipfile.ZipFile(zp, 'r') as z:
                for m in z.infolist():
                    # prevent zip-slip
                    if m.is_dir():
                        continue
                    rel = Path(m.filename)
                    if any(part in ("..", "~") for part in rel.parts):
                        continue
                    ext = rel.suffix.lower()
                    if ext not in {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}:
                        continue
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    out.append(str(target))
        except Exception as e:
            _log.warning("Zip extract failed: %s", e)
        return out

    def _ingest_zips(zips, cur_files):
        files = list(cur_files or [])
        added = 0
        if zips:
            for z in zips:
                extracted = _safe_extract_zip(str(getattr(z, 'name', str(z))))
                files.extend(extracted)
                added += len(extracted)
        if not files:
            return [], {}, gr.update(choices=[], value=None), "No files uploaded.", "No files."
        names = [Path(str(getattr(f, 'name', str(f)) if hasattr(f, 'name') else f)).name for f in files]
        groups = _summarize_groups(files)
        _log.info("Zip ingest: +%d file(s); total %d", added, len(files))
        return files, groups, gr.update(choices=names, value=names[0] if names else None), f"Added {added} file(s) from ZIP.", _format_group_summary(groups)

    up_zip.change(_ingest_zips, inputs=[up_zip, files_state], outputs=[files_state, groups_state, file_pick, upload_msg, group_summary])

    # Wire preview
    def _preview_entry(files, pick_name, start, dur, device, seam, chunk, overlap, wet, pretx):
        # Choose the file object by matching base name
        sel = None
        try:
            if files:
                for f in files:
                    if Path(str(f)).name == pick_name:
                        sel = f
                        break
                if sel is None:
                    sel = files[0]
        except Exception:
            sel = None
        return _preview_one(True, sel, start, dur, device, seam, chunk, overlap, wet, pretx)

    prev_btn.click(
        _preview_entry,
        inputs=[files_state, file_pick, start, dur, device, seam, chunk, overlap, wet, pretx],
        outputs=[audio, prev_msg],
    )

    # Wire export (export tab)
    exp_btn.click(
        _export_batch,
        inputs=[files_state, device, seam, chunk, overlap, wet, pretx, gain],
        outputs=[out_files, out_msg],
    )

    # Wire synced multichannel export
    exp_sync_btn.click(
        _export_and_combine,
        inputs=[files_state, device, seam, chunk, overlap, wet, pretx, prefer_48k, skip_fine, gain, batch_folders],
        outputs=[out_sync, out_msg],
    )


if __name__ == "__main__":
    _log.info("Starting Resemble Enhance Web UI")
    # Prefer a fixed localhost port; if busy, retry on any free port
    try:
        demo.launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=7860, show_error=True)
    except OSError as e:
        _log.warning("Port 7860 unavailable (%s); retrying on a free port", e)
        demo.launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=None, show_error=True)

