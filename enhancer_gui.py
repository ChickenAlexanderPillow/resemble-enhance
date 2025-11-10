import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
import ctypes
from ctypes import wintypes

# Optional drag-and-drop support via tkinterdnd2 (if available)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:  # noqa: BLE001
    DND_AVAILABLE = False


BASE = Path.cwd()
INPUT_TMP_ROOT = BASE / ".enhancer_runs_gui"
OUTPUT_ROOT = BASE / "output_audio"


def _get_console_python() -> str:
    exe = sys.executable or "python"
    lower = exe.lower()
    if lower.endswith("pythonw.exe"):
        cand = Path(exe).with_name("python.exe")
        if cand.exists():
            return str(cand)
        return "python"
    return exe


def _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb):
    from resemble_enhance.enhancer.inference import denoise
    import torchaudio
    from torchaudio.functional import resample as ta_resample

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expected = len(files)
    if progress_cb:
        progress_cb(0, expected)

    def on_chunk(evt, name, i, n):
        if chunk_progress_cb:
            chunk_progress_cb(name or "", i, n)

    done = 0
    out_dirs = set()
    results: list[tuple[str, str]] = []
    for f in files:
        p = Path(f)
        dest_dir = p.parent / f"Enhanced_{stamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.add(dest_dir)
        name = p.name
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0)
        kwargs = dict(chunk_seconds=31.0, overlap_seconds=1.0, align_max_shift_ratio=0.25, align_disable=True)
        hwav, model_sr = denoise(dwav=wav, sr=sr, device=device, run_dir=None, progress_cb=on_chunk, **kwargs)
        dest_sr = 48000 if profile else sr
        if model_sr != dest_sr:
            hwav = ta_resample(hwav, orig_freq=model_sr, new_freq=dest_sr)
        exp_len = round(wav.shape[-1] * (dest_sr / sr)) if dest_sr != sr else wav.shape[-1]
        if hwav.shape[-1] > exp_len:
            hwav = hwav[:exp_len]
        elif hwav.shape[-1] < exp_len:
            import torch
            hwav = torch.nn.functional.pad(hwav, (0, exp_len - hwav.shape[-1]))
        out_path = dest_dir / name
        torchaudio.save(str(out_path), hwav[None], dest_sr)
        done += 1
        if progress_cb:
            progress_cb(done, expected)
        results.append((str(p), str(out_path)))

    return results


def run_enhancer_for(files, device="cuda", profile=True, progress_cb=None, chunk_progress_cb=None):
    # When frozen into an EXE, run in-process for full portability
    if getattr(sys, 'frozen', False):
        return _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb)

    run_id = uuid.uuid4().hex[:8]
    in_dir = INPUT_TMP_ROOT / run_id / "input_audio"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = OUTPUT_ROOT / stamp
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy files into temp input dir
    for f in files:
        dst = in_dir / Path(f).name
        shutil.copy2(f, dst)

    py = _get_console_python()
    cmd = [
        py,
        "-m",
        "resemble_enhance.enhancer",
        str(in_dir),
        str(out_dir),
        "--denoise_only",
        "--device",
        device,
    ]
    if profile:
        cmd += ["--profile", "camera_sync"]
    # Disable alignment at joins to avoid loops
    cmd += ["--align_disable"]

    # Launch process
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = os.environ.copy()
    env["RESEMBLE_PROGRESS"] = "1"
    proc = subprocess.Popen(
        cmd,
        creationflags=creationflags,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Read and parse progress output
    def reader():
        import re
        nonlocal expected
        current_file = None
        cur_n = 0
        start_re = re.compile(r"^PROGRESS START file=(.*) n=(\d+)$")
        chunk_re = re.compile(r"^PROGRESS CHUNK file=(.*) i=(\d+) n=(\d+)$")
        end_re = re.compile(r"^PROGRESS END file=(.*)$")
        for line in proc.stdout:  # type: ignore[attr-defined]
            line = line.rstrip()
            m = start_re.match(line)
            if m:
                current_file = m.group(1)
                try:
                    cur_n = int(m.group(2))
                except Exception:
                    cur_n = 0
                if chunk_progress_cb:
                    chunk_progress_cb(current_file or "", 0, cur_n)
                continue
            m = chunk_re.match(line)
            if m:
                name = m.group(1)
                try:
                    i = int(m.group(2)); n = int(m.group(3))
                except Exception:
                    i, n = 0, 0
                if chunk_progress_cb:
                    chunk_progress_cb(name or current_file or "", i, n)
                continue
            m = end_re.match(line)
            if m:
                name = m.group(1)
                if chunk_progress_cb:
                    chunk_progress_cb(name or current_file or "", cur_n, cur_n)
                continue
            # ignore other logs

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    expected = len(files)
    completed = 0
    # Poll for outputs while the process runs
    while proc.poll() is None:
        completed = sum(1 for f in files if (out_dir / Path(f).name).exists())
        if progress_cb:
            progress_cb(completed, expected)
        time.sleep(0.5)

    # Final progress update
    completed = sum(1 for f in files if (out_dir / Path(f).name).exists())
    if progress_cb:
        progress_cb(completed, expected)

    rc = proc.returncode
    if rc != 0:
        raise RuntimeError(f"Enhancer returned code {rc}")

    # Cleanup inputs if outputs look valid
    # Move outputs next to the original files under Enhanced_<timestamp> (keep originals)
    results: list[tuple[str, str]] = []
    for f in files:
        name = Path(f).name
        tmp_out = out_dir / name
        if not tmp_out.exists() or tmp_out.stat().st_size <= 44:
            continue
        dest_dir = Path(f).parent / f"Enhanced_{stamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_out = dest_dir / name
        try:
            shutil.move(str(tmp_out), str(final_out))
        except Exception:
            # Best-effort fallback to copy
            shutil.copy2(str(tmp_out), str(final_out))
            tmp_out.unlink(missing_ok=True)
        results.append((str(Path(f)), str(final_out)))

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
        self.title("Resemble Enhance")
        self.geometry("980x620")

        self.files = []
        self.history: list[tuple[str, str]] = []
        self.cur_start_time: float | None = None

        # Basic styling
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        # Dark, sleek palette
        self._bg = "#D1D1D1"
        self._panel = "#BBBBBB"
        self._text = "#161616"
        self._muted = "#ffffff"
        self._accent = "#464646"
        self.configure(bg=self._bg)
        style.configure('.', background=self._bg, foreground=self._text)
        style.configure('TFrame', background=self._bg)
        style.configure('Title.TLabel', background=self._bg, foreground=self._muted, font=('Segoe UI', 13, 'bold'))
        style.configure('Info.TLabel', background=self._bg, foreground=self._text)
        style.configure('TButton', padding=8)
        style.map('TButton', background=[('active', "#7a7a7a")])
        style.configure('Drop.TFrame', background=self._panel, bordercolor='#1f2a44', relief='solid')
        style.configure('Horizontal.TProgressbar', troughcolor='#0c0f14', background=self._accent, bordercolor=self._panel, lightcolor=self._accent, darkcolor=self._accent)
        style.configure('Treeview', background=self._panel, fieldbackground=self._panel, foreground=self._text, bordercolor=self._panel)
        style.map('Treeview', background=[('selected', '#1f3d66')], foreground=[('selected', self._text)])
        # Accent button styles
        style.configure('Accent.TButton', background=self._accent, foreground=self._text)
        style.configure('AccentHover.TButton', background="#242424", foreground=self._text)
        # Standard hover style for normal buttons
        style.configure('Hover.TButton', background='#1b2333', foreground=self._text)

        # Helper to apply a simple hover effect to any ttk.Button
        def _bind_hover(b: ttk.Button, base_style: str = 'TButton', hover_style: str = 'Hover.TButton'):
            try:
                b.configure(style=base_style)
                b.bind('<Enter>', lambda e: b.configure(style=hover_style))
                b.bind('<Leave>', lambda e: b.configure(style=base_style))
            except Exception:
                pass
        self._bind_hover = _bind_hover

        # Main paned layout
        pw = ttk.Panedwindow(self, orient='horizontal')
        pw.pack(fill='both', expand=True, padx=12, pady=12)

        left = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(left, weight=1)
        pw.add(right, weight=2)

        # Left column: title, drop zone, buttons, queue list
        title = "Drop or select audio files to enhance" if DND_AVAILABLE else "Select audio files to enhance"
        ttk.Label(left, text=title, style='Title.TLabel').pack(anchor='w')

        dzframe = ttk.Frame(left, style='Drop.TFrame')
        dzframe.pack(fill='x', pady=(8, 10))
        self.drop_zone = tk.Label(dzframe, text='Drop files here', font=('Segoe UI', 13), fg=self._accent, bg=self._panel, bd=2, relief='solid', height=3)
        self.drop_zone.pack(fill='x')
        self.drop_zone.bind('<Button-1>', lambda e: self.add_files())
        self.drop_zone.bind('<Enter>', lambda e: self._dz_hover(True))
        self.drop_zone.bind('<Leave>', lambda e: self._dz_hover(False))
        if DND_AVAILABLE:
            try:
                self.drop_zone.drop_target_register(DND_FILES)
                self.drop_zone.dnd_bind('<<Drop>>', self._on_drop)
            except Exception:
                pass
        else:
            _install_win_dnd(self.drop_zone, lambda files: self._add_paths(files) or self._enable_run())

        btns = ttk.Frame(left)
        btns.pack(fill='x', pady=(0, 8))
        btn_add = ttk.Button(btns, text='Add Files', command=self.add_files)
        btn_add.pack(side='left')
        self._bind_hover(btn_add)
        btn_clear = ttk.Button(btns, text='Clear', command=self.clear_files)
        btn_clear.pack(side='left', padx=6)
        self._bind_hover(btn_clear)
        self.var_profile = tk.BooleanVar(value=True)
        ttk.Checkbutton(btns, text='Camera Sync Profile (48k, safe)', variable=self.var_profile).pack(side='left', padx=12)
        # Optional: sync and export as multichannel via Audalign
        self.var_sync_export = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text='Sync + export multichannel (Audalign)', variable=self.var_sync_export).pack(side='left', padx=12)

        self.listbox = tk.Listbox(left, height=14, selectmode='extended', bg=self._panel, fg=self._text, highlightthickness=0, selectbackground='#1f3d66', selectforeground=self._text)
        self.listbox.pack(fill='both', expand=True)
        dnd_enabled = False
        if DND_AVAILABLE:
            try:
                self.listbox.drop_target_register(DND_FILES)
                self.listbox.dnd_bind('<<Drop>>', self._on_drop)
                dnd_enabled = True
            except Exception:
                dnd_enabled = False
        if not dnd_enabled:
            _install_win_dnd(self.listbox, lambda files: self._add_paths(files) or self._enable_run())

        # Right column: action, progress, log, history
        topbar = ttk.Frame(right)
        topbar.pack(fill='x')
        self.run_btn = ttk.Button(topbar, text='Enhance', command=self.run_task, state='disabled', style='Accent.TButton')
        self.run_btn.pack(side='right')
        # Hover effect for enhance button
        self.run_btn.bind('<Enter>', lambda e: self.run_btn.configure(style='AccentHover.TButton'))
        self.run_btn.bind('<Leave>', lambda e: self.run_btn.configure(style='Accent.TButton'))

        progfrm = ttk.Frame(right)
        progfrm.pack(fill='x', pady=(8, 6))
        self.progress = ttk.Progressbar(progfrm, mode='determinate')
        self.progress.pack(fill='x')
        self.overall_label = ttk.Label(progfrm, text='0 of 0 files')
        self.overall_label.pack(anchor='w', pady=(2, 0))

        curfrm = ttk.Frame(right)
        curfrm.pack(fill='x')
        ttk.Label(curfrm, text='Current file:').pack(side='left')
        self.cur_label = ttk.Label(curfrm, text='-')
        self.cur_label.pack(side='left', padx=6)
        self.cur_bar = ttk.Progressbar(right, mode='determinate')
        self.cur_bar.pack(fill='x', pady=(4, 2))
        self.cur_prog_label = ttk.Label(right, text='0%')
        self.cur_prog_label.pack(anchor='w', pady=(0, 8))

        self.log = tk.Text(right, height=6, state='disabled', bg=self._panel, fg=self._text, insertbackground=self._text, highlightthickness=0)
        self.log.pack(fill='x')

        ttk.Label(right, text='Finished enhancements', style='Title.TLabel').pack(anchor='w', pady=(10, 2))
        histfrm = ttk.Frame(right)
        histfrm.pack(fill='both', expand=True)
        self.hist = ttk.Treeview(histfrm, columns=('src','out'), show='headings')
        self.hist.heading('src', text='Source')
        self.hist.heading('out', text='Output')
        self.hist.column('src', width=360, anchor='w')
        self.hist.column('out', width=360, anchor='w')
        self.hist.pack(side='left', fill='both', expand=True)
        sb = ttk.Scrollbar(histfrm, orient='vertical', command=self.hist.yview)
        self.hist.configure(yscroll=sb.set)
        sb.pack(side='right', fill='y')
        btnhist = ttk.Frame(right)
        btnhist.pack(fill='x', pady=6)
        btn_open = ttk.Button(btnhist, text='Open Selected Output', command=self._open_selected_output)
        btn_open.pack(side='left')
        self._bind_hover(btn_open)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=[("WAV files", ".wav .WAV"), ("All files", "*.*")])
        if not paths:
            return
        self._add_paths(paths)
        self._enable_run()

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, "end")
        self.run_btn["state"] = "disabled"
        self.progress["value"] = 0
        self._log_clear()

    def _add_paths(self, paths):
        for p in paths:
            p = str(p)
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert("end", p)

    def _enable_run(self):
        self.run_btn["state"] = "normal" if self.files else "disabled"

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

    def _log(self, msg):
        self.log["state"] = "normal"
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log["state"] = "disabled"

    def _log_clear(self):
        self.log["state"] = "normal"
        self.log.delete("1.0", "end")
        self.log["state"] = "disabled"

    def _dz_hover(self, on: bool):
        try:
            self.drop_zone.configure(bg=self._accent if on else self._panel, fg=self._text if on else self._accent)
        except Exception:
            pass

    def _clear_queue(self):
        self.files.clear()
        self.listbox.delete(0, "end")
        self._enable_run()
        # Reset progress indicators
        self.progress["value"] = 0
        self.cur_bar["value"] = 0
        self.cur_label["text"] = "-"

    def _append_history(self, results: list[tuple[str,str]]):
        # results: list of (src, out)
        for src, out in results:
            self.history.append((src, out))
            self.hist.insert('', 'end', values=(src, out))

    def _open_selected_output(self):
        sel = self.hist.selection()
        if not sel:
            return
        item = self.hist.item(sel[0])
        vals = item.get('values') or []
        if len(vals) < 2:
            return
        path = vals[1]
        try:
            if os.name == 'nt':
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(['open', path])
        except Exception as e:  # noqa: BLE001
            self._log(f"Open failed: {e}")

    def run_task(self):
        if not self.files:
            return

        self.run_btn["state"] = "disabled"
        self.progress["value"] = 0

        def update_prog(done, total):
            total = max(total, 1)
            pct = int(done * 100 / total)
            self.progress["maximum"] = 100
            self.progress["value"] = pct
            self.overall_label["text"] = f"{done} of {total} files ({pct}%)"

        def update_chunk(name, i, n):
            n = max(n or 0, 1)
            pct = int((i * 100) / n)
            self.cur_bar["maximum"] = 100
            self.cur_bar["value"] = pct
            base = Path(name).name if name else "-"
            self.cur_label["text"] = f"{base} ({i}/{n})"
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
            self.cur_prog_label["text"] = f"{pct}%{eta_txt}"

        def worker():
            try:
                self.after(0, lambda: self._log("Launching enhancer..."))
                results = run_enhancer_for(
                    self.files,
                    device="cuda",
                    profile=self.var_profile.get(),
                    progress_cb=lambda d, t: self.after(0, update_prog, d, t),
                    chunk_progress_cb=lambda name, i, n: self.after(0, update_chunk, name, i, n),
                )
                # Log and show history
                self.after(0, lambda: self._log(f"Done. {len(results)} file(s) enhanced."))
                self.after(0, lambda: self._append_history(results))
                # Optional: sync and export as multichannel using Audalign
                if self.var_sync_export.get() and results:
                    try:
                        self.after(0, lambda: self._log("Syncing with Audalign and exporting multichannel..."))
                        out_path = _sync_and_export_multichannel([out for _, out in results], prefer_48k=self.var_profile.get())
                        if out_path:
                            self.after(0, lambda: self._log(f"Multichannel export written: {out_path}"))
                        else:
                            self.after(0, lambda: self._log("Multichannel export failed: no output produced"))
                    except Exception as e:
                        self.after(0, lambda e=e: self._log(f"Sync/export error: {e}"))
                # Clear the queue after successful enhance
                self.after(0, self._clear_queue)
            except Exception as e:  # noqa: BLE001
                self.after(0, lambda e=e: self._log(f"Error: {e}"))
            finally:
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


# --- Alignment and multichannel export helpers (Audalign-based) ---

def _sync_and_export_multichannel(file_paths: list[str], prefer_48k: bool = True) -> str | None:
    """Align given enhanced files using audalign and export a single multichannel WAV.

    Returns the output multichannel wav path or None on failure.
    """
    import importlib
    from datetime import datetime
    import torchaudio
    import torch

    if not file_paths:
        return None

    # Import audalign lazily
    ad = importlib.import_module('audalign')

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_parent = Path(file_paths[0]).parent
    # Intermediate working dir to avoid leaving per-file outputs
    tmp_dir = (Path.cwd() / ".enhancer_runs_gui" / "tmp_sync" / stamp)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Final folder should contain ONLY the combined multichannel file
    final_dir = first_parent / f"Synced_{stamp}"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Choose recognizers
    try:
        fingerprint_rec = getattr(ad, 'FingerprintRecognizer')()
    except Exception:
        fingerprint_rec = None
    try:
        fine_rec = getattr(ad, 'CorrelationSpectrogramRecognizer')()
    except Exception:
        try:
            fine_rec = getattr(ad, 'CorrelationRecognizer')()
        except Exception:
            fine_rec = None

    # 1) Coarse alignment (prefer fast correlation) and write padded copies if needed
    results = None
    try:
        align_files = getattr(ad, 'align_files')
        # Try correlation-only first for speed
        if hasattr(ad, 'CorrelationRecognizer'):
            corr_rec = getattr(ad, 'CorrelationRecognizer')()
            results = align_files(*file_paths, recognizer=corr_rec)
        # Fallback to fingerprinting if needed
        if not results and fingerprint_rec:
            results = align_files(*file_paths, recognizer=fingerprint_rec)
    except Exception:
        results = None

    # Fallback: correlation-only alignment when fingerprinting doesn't match
    try:
        if (not results) and 'CorrelationRecognizer' in dir(ad):
            corr_rec = getattr(ad, 'CorrelationRecognizer')()
            results = align_files(*file_paths, recognizer=corr_rec)
    except Exception:
        pass

    # 2) Fine align if available
    try:
        if results is not None and fine_rec is not None and hasattr(ad, 'fine_align'):
            results = ad.fine_align(results, recognizer=fine_rec)
    except Exception:
        pass

    # 3) Write aligned, padded mono files to out_dir
    wrote_aligned = False
    # Preferred: write shifts from results
    try:
        if results is not None and hasattr(ad, 'write_shifts_from_results'):
            ad.write_shifts_from_results(results, str(tmp_dir), file_paths)
            wrote_aligned = True
    except Exception:
        wrote_aligned = False
    # Fallback: directly call align_files with destination_path
    if not wrote_aligned:
        try:
            align_files = getattr(ad, 'align_files')
            if fingerprint_rec:
                align_files(*file_paths, destination_path=str(tmp_dir), recognizer=fingerprint_rec)
            else:
                align_files(*file_paths, destination_path=str(tmp_dir))
            wrote_aligned = True
        except Exception:
            wrote_aligned = False

    if not wrote_aligned:
        return None

    # 4) Load aligned files from tmp_dir and build multichannel tensor
    # Try to keep channel order same as input list
    basenames = [Path(p).name for p in file_paths]
    aligned_paths: list[Path] = []
    for base in basenames:
        cand = tmp_dir / base
        if cand.exists():
            aligned_paths.append(cand)
        else:
            # Try to find by stem if audalign changed extension/casing
            stem = Path(base).stem
            matches = list(tmp_dir.glob(f"{stem}*"))
            if matches:
                aligned_paths.append(matches[0])

    if not aligned_paths:
        # As a last resort, take all wavs in out_dir excluding an obvious sum file
        aligned_paths = sorted([p for p in tmp_dir.glob("*.wav") if 'sum' not in p.stem.lower() and 'total' not in p.stem.lower()])

    if not aligned_paths:
        return None

    wavs = []
    srs = []
    max_len = 0
    for p in aligned_paths:
        wav, sr = torchaudio.load(str(p))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # downmix to mono
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        srs.append(int(sr))
        max_len = max(max_len, int(wav.size(-1)))
        wavs.append((wav, int(sr)))

    # Choose target SR
    target_sr = 48000 if prefer_48k else (srs[0] if srs else 48000)
    from torchaudio.functional import resample as ta_resample

    chan_tensors = []
    for wav, sr in wavs:
        mono = wav[0]
        if sr != target_sr:
            mono = ta_resample(mono, orig_freq=sr, new_freq=target_sr)
        cur_len = mono.size(-1)
        if cur_len < max_len:
            mono = torch.nn.functional.pad(mono, (0, max_len - cur_len))
        elif cur_len > max_len:
            mono = mono[:max_len]
        chan_tensors.append(mono.unsqueeze(0))

    if not chan_tensors:
        return None

    multich = torch.cat(chan_tensors, dim=0)
    out_wav = final_dir / f"Synced_Multichannel_{stamp}.wav"
    torchaudio.save(str(out_wav), multich, target_sr)

    # Optional: clear channel mask to avoid L/R stereo interpretation in some NLEs
    # Requires ffmpeg on PATH. This remuxes headers so channels are treated as generic (centered) mono channels.
    try:
        subprocess.run([
            _get_console_python(), '-c', 'import sys'
        ], check=True)
    except Exception:
        pass
    try:
        ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
        if ff:
            tmp_out = final_dir / f"Synced_Multichannel_{stamp}_nomask.wav"
            # -write_channel_mask 0 clears speaker assignment; copy to avoid re-encoding
            cmd = [ff, '-y', '-i', str(out_wav), '-c', 'copy', '-write_channel_mask', '0', str(tmp_out)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if tmp_out.exists() and tmp_out.stat().st_size > 44:
                try:
                    out_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                tmp_out.replace(out_wav)
    except Exception:
        pass

    # Cleanup tmp_dir completely so only the combined file remains
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return str(out_wav)


if __name__ == "__main__":
    app = App()
    app.mainloop()
