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


class _Cancelled(Exception):
    pass


class _Control:
    def __init__(self) -> None:
        import threading as _th
        self.pause = _th.Event()        # when set, pause at next chunk boundary
        self.stop_after_chunk = _th.Event()  # when set, cancel gracefully after current chunk
        self.cancel_now = _th.Event()   # immediate cancel request (treated same at chunk boundary)


def _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb, seam_safe: bool = True, control: _Control | None = None):
    from resemble_enhance.enhancer.inference import denoise
    import torchaudio
    from torchaudio.functional import resample as ta_resample

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expected = len(files)
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
    results: list[tuple[str, str]] = []
    for f in files:
        p = Path(f)
        dest_dir = p.parent / f"Enhanced_{stamp}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.add(dest_dir)
        name = p.name
        if control is not None and (control.cancel_now.is_set() or control.stop_after_chunk.is_set()):
            break
        wav, sr = torchaudio.load(str(p))
        wav = wav.mean(0)
        if seam_safe:
            kwargs = dict(chunk_seconds=60.0, overlap_seconds=4.0, align_max_shift_ratio=0.05, align_disable=False)
        else:
            kwargs = dict(chunk_seconds=31.0, overlap_seconds=1.0, align_max_shift_ratio=0.25, align_disable=True)
        try:
            hwav, model_sr = denoise(
                dwav=wav,
                sr=sr,
                device=device,
                run_dir=None,
                progress_cb=lambda evt, nm, i, n, p=p: on_chunk(evt, str(p), i, n),
                **kwargs,
            )
        except _Cancelled:
            break
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


def run_enhancer_for(files, device="cuda", profile=True, progress_cb=None, chunk_progress_cb=None, seam_safe: bool = True, control: _Control | None = None):
    # When frozen into an EXE, run in-process for full portability
    if getattr(sys, 'frozen', False) or control is not None:
        return _enhance_in_process(files, device, profile, progress_cb, chunk_progress_cb, seam_safe=seam_safe, control=control)

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
    # Seam handling
    if seam_safe:
        cmd += [
            "--chunk_seconds", "60.0",
            "--overlap_seconds", "4.0",
            "--align_max_shift_ratio", "0.05",
        ]
    else:
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
                    i = int(m.group(2))
                    n = int(m.group(3))
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
        self.minsize(1000, 640)

        self.files = []
        self.history: list[tuple[str, str]] = []
        self.cur_start_time: float | None = None
        self.folders: set[str] = set()
        # Per-file status: queued|running|done|failed
        self.file_status: dict[str, str] = {}
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
            self._bg = getattr(cols, 'bg', '#0f1115') if cols is not None else '#0f1115'
            # Prefer input background or a neutral secondary for panels
            self._panel = (
                getattr(cols, 'inputbg', None) or
                getattr(cols, 'secondary', None) or
                '#151923'
            ) if cols is not None else '#151923'
            self._text = getattr(cols, 'fg', '#e5e7eb') if cols is not None else '#e5e7eb'
            self._muted = getattr(cols, 'secondary', '#9aa4b2') if cols is not None else '#9aa4b2'
            self._accent = getattr(cols, 'primary', '#3b82f6') if cols is not None else '#3b82f6'
        else:
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass
            # Fallback dark palette
            self._bg = "#0f1115"
            self._panel = "#151923"
            self._text = "#e5e7eb"
            self._muted = "#9aa4b2"
            self._accent = "#3b82f6"
        self.configure(bg=self._bg)
        style.configure('.', background=self._bg, foreground=self._text)
        style.configure('TFrame', background=self._bg)
        style.configure('Title.TLabel', background=self._bg, foreground=self._text, font=('Segoe UI', 13, 'bold'))
        style.configure('Info.TLabel', background=self._bg, foreground=self._muted)
        style.configure('TButton', padding=8, background=self._panel, foreground=self._text)
        style.map('TButton', background=[('active', "#1f2937")])
        style.configure('Drop.TFrame', background=self._panel, bordercolor='#1f2937', relief='solid')
        style.configure('Horizontal.TProgressbar', troughcolor='#0b0f16', background=self._accent, bordercolor=self._panel, lightcolor=self._accent, darkcolor=self._accent)
        # Tree grid/separators slightly lighter so columns are clearly divided
        style.configure('Treeview', background=self._panel, fieldbackground=self._panel, foreground=self._text, bordercolor='#222833')
        style.map('Treeview', background=[('selected', '#1e293b')], foreground=[('selected', self._text)])
        # Treeview heading styling (neutral grey shade)
        style.configure('Treeview.Heading', background='#1f1f1f', foreground=self._text, bordercolor='#2a2a2a')
        style.map('Treeview.Heading', background=[('active', '#1e293b')])
        # Accent button styles
        style.configure('Accent.TButton', background=self._accent, foreground="#ffffff")
        style.configure('AccentHover.TButton', background="#60a5fa", foreground="#ffffff")
        # Standard hover style for normal buttons
        style.configure('Hover.TButton', background='#1e293b', foreground=self._text)
        # Option checkbuttons
        style.configure('Opt.TCheckbutton', background=self._panel, foreground=self._text, padding=4)

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

        left = ttk.Frame(pw, width=360)
        right = ttk.Frame(pw)
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

        # Left column: title, buttons, options, queue list
        title = "Drop or select audio files to enhance" if DND_AVAILABLE else "Select audio files to enhance"
        ttk.Label(left, text=title, style='Title.TLabel').pack(anchor='w')

        # (Removed dedicated drop zone; drag-and-drop works on the list below.)

        btns = ttk.Frame(left)
        btns.pack(fill='x', pady=(0, 6))
        btn_add = ttk.Button(btns, text='Add Files', command=self.add_files)
        btn_add.pack(side='left')
        self._bind_hover(btn_add)
        btn_add_folder = ttk.Button(btns, text='Add Folder', command=self.add_folder)
        btn_add_folder.pack(side='left', padx=6)
        self._bind_hover(btn_add_folder)
        btn_clear = ttk.Button(btns, text='Clear', command=self.clear_files)
        btn_clear.pack(side='left', padx=6)
        self._bind_hover(btn_clear)

        # Advanced options (collapsed dropdown)
        self.var_profile = tk.BooleanVar(value=True)
        self.var_sync_export = tk.BooleanVar(value=True)
        self.var_postproc = tk.BooleanVar(value=True)
        self.var_fast_wav = tk.BooleanVar(value=False)
        self.var_skip_fine = tk.BooleanVar(value=True)
        self.var_bw64 = tk.BooleanVar(value=True)
        self.var_seam_safe = tk.BooleanVar(value=True)
        self.var_batch_folders = tk.BooleanVar(value=True)

        adv_hdr = ttk.Frame(left)
        adv_hdr.pack(fill='x', pady=(4, 2))
        self._adv_open = False
        self._adv_btn = ttk.Button(adv_hdr, text='Advanced Options â–¸', command=self._toggle_advanced)
        self._adv_btn.pack(anchor='w')

        self._adv_opts = ttk.Frame(left, style='TFrame')
        # build options into the advanced frame (initially hidden)
        ttk.Checkbutton(self._adv_opts, text='Camera Sync (48 kHz)', style='Opt.TCheckbutton', variable=self.var_profile).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Sync + Multichannel', style='Opt.TCheckbutton', variable=self.var_sync_export).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Level + Brighten', style='Opt.TCheckbutton', variable=self.var_postproc).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Fast export (WAV only)', style='Opt.TCheckbutton', variable=self.var_fast_wav).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Skip fine align', style='Opt.TCheckbutton', variable=self.var_skip_fine).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Export BW64 (ADM dual-mono)', style='Opt.TCheckbutton', variable=self.var_bw64).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Seam-safe joins (longer overlap + micro-align)', style='Opt.TCheckbutton', variable=self.var_seam_safe).pack(anchor='w', fill='x', pady=1)
        ttk.Checkbutton(self._adv_opts, text='Batch by folder (group files per folder)', style='Opt.TCheckbutton', variable=self.var_batch_folders).pack(anchor='w', fill='x', pady=1)

        # Queue controls: filter + actions
        ctl = ttk.Frame(left)
        ctl.pack(fill='x', pady=(4, 4))
        ttk.Label(ctl, text='Show:').pack(side='left')
        self.status_filter_var = tk.StringVar(value='All')
        self.status_filter = ttk.Combobox(ctl, textvariable=self.status_filter_var, values=['All','Queued','Running','Done','Failed'], state='readonly', width=10)
        self.status_filter.pack(side='left', padx=(6, 0))
        self.status_filter.bind('<<ComboboxSelected>>', lambda e: self._refresh_queue_tree())

        # Queue tree: grouped folders with child files; multi-select enabled
        self.queue_tree = ttk.Treeview(left, show='tree', selectmode='extended')
        self.queue_tree.pack(fill='both', expand=True)
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
                _drag_cleanup(); return
            dst = self.queue_tree.identify_row(e.y)
            if not dst or dst == src:
                _drag_cleanup(); return
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
                    elif dfolder:  # drop on folder row â€“ before/after the whole block
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

        # Trash can button bottom-right under queue
        queue_footer = ttk.Frame(left)
        queue_footer.pack(fill='x', pady=(4, 0))
        self.trash_btn = ttk.Button(queue_footer, text='ðŸ—‘', width=3, command=self._remove_selected)
        self.trash_btn.pack(side='right')
        self._bind_hover(self.trash_btn)
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

        # Action button at bottom of left column
        bottom_actions = ttk.Frame(left)
        bottom_actions.pack(fill='x', side='bottom', pady=(8, 0))
        self.run_btn = ttk.Button(bottom_actions, text='Enhance', command=self.run_task, state='disabled', style='Accent.TButton')
        self.run_btn.pack(fill='x')
        self.run_btn.bind('<Enter>', lambda e: self.run_btn.configure(style='AccentHover.TButton'))
        self.run_btn.bind('<Leave>', lambda e: self.run_btn.configure(style='Accent.TButton'))
        # Pause/Cancel controls
        ctrlfrm = ttk.Frame(bottom_actions)
        ctrlfrm.pack(fill='x', pady=(6, 0))
        self.pause_btn = ttk.Button(ctrlfrm, text='Pause', command=self._toggle_pause, state='disabled')
        self.pause_btn.pack(side='left')
        self.cancel_btn = ttk.Button(ctrlfrm, text='Cancel', command=self._cancel_graceful, state='disabled')
        self.cancel_btn.pack(side='left', padx=(6, 0))

        progfrm = ttk.Frame(right)
        progfrm.pack(fill='x', pady=(8, 6), padx=(12, 12))
        self.progress = ttk.Progressbar(progfrm, mode='determinate')
        self.progress.pack(fill='x')
        self.overall_label = ttk.Label(progfrm, text='0 of 0 files')
        self.overall_label.pack(anchor='w', pady=(2, 0))
        # Status line below the main progress bar
        self.status_label = ttk.Label(right, text='')
        self.status_label.pack(fill='x', padx=(12, 12), pady=(6, 0))

        ttk.Label(right, text='Enhanced files', style='Title.TLabel').pack(anchor='w', pady=(10, 2))
        histfrm = ttk.Frame(right)
        histfrm.pack(fill='both', expand=True, padx=(12, 12))
        self.hist = ttk.Treeview(histfrm, columns=('src','out'), show='headings', selectmode='browse')
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
        btnhist = ttk.Frame(right)
        btnhist.pack(fill='x', pady=6, padx=(12, 12))
        btn_open = ttk.Button(btnhist, text='Open Selected Output', command=self._open_selected_output)
        btn_open.pack(side='left')
        self._bind_hover(btn_open)

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=[("WAV files", ".wav .WAV"), ("All files", "*.*")])
        if not paths:
            return
        self._add_paths(paths)
        self._enable_run()

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing audio")
        if not folder:
            return
        self._add_paths([folder])
        self._enable_run()

    def clear_files(self):
        self.files.clear()
        try:
            self.folders.clear()
        except Exception:
            self.folders = set()
        try:
            self.file_status.clear()
        except Exception:
            pass
        self.run_btn["state"] = "disabled"
        self.progress["value"] = 0
        self._log_clear()
        self._refresh_queue_tree()

    def _add_paths(self, paths):
        def _is_audio(path: str) -> bool:
            suf = Path(path).suffix.lower()
            return suf in {'.wav', '.wave'}
        for p in paths:
            p = str(p)
            try:
                if Path(p).is_dir():
                    self.folders.add(p)
                    # Collect audio files directly under this folder (non-recursive)
                    for f in sorted(Path(p).iterdir()):
                        if f.is_file() and _is_audio(str(f)):
                            sfp = str(f)
                            if sfp not in self.files:
                                self.files.append(sfp)
                    continue
            except Exception:
                pass
            if _is_audio(p) and p not in self.files:
                self.files.append(p)
                self.file_status[p] = self.file_status.get(p, 'queued')
        self._refresh_queue_tree()

    def _enable_run(self):
        self.run_btn["state"] = "normal" if self.files else "disabled"
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
        groups: dict[str, list[str]] = {}
        for fp in self.files:
            parent = str(Path(fp).parent)
            groups.setdefault(parent, []).append(fp)
        # Include explicitly added empty folders
        for d in self.folders:
            groups.setdefault(d, groups.get(d, []))
        # Determine folder order by first appearance in self.files, then any explicitly added folders
        seen = set()
        folder_order: list[str] = []
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
            node = self.queue_tree.insert('', 'end', text=folder, open=True)
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
                self.queue_tree.insert(node, 'end', text='(no wav files)')
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
        try:
            if self._adv_open:
                self._adv_opts.pack_forget()
                self._adv_open = False
                try:
                    self._adv_btn.configure(text='Advanced Options ▸')
                except Exception:
                    pass
            else:
                self._adv_opts.pack(fill='x', pady=(2, 8))
                self._adv_open = True
                try:
                    self._adv_btn.configure(text='Advanced Options ▾')
                except Exception:
                    pass
        except Exception:
            pass

    def _log(self, msg):
        try:
            self.status_label["text"] = str(msg)
        except Exception:
            pass

    def _log_clear(self):
        try:
            self.status_label["text"] = ''
        except Exception:
            pass

    def _set_status(self, text: str):
        try:
            self.status_label["text"] = text
        except Exception:
            pass

    def _dz_hover(self, on: bool):
        try:
            self.drop_zone.configure(bg=self._accent if on else self._panel, fg=self._text if on else self._accent)
        except Exception:
            pass

    def _clear_queue(self):
        self.files.clear()
        try:
            self.folders.clear()
        except Exception:
            self.folders = set()
        self.file_status.clear()
        self._refresh_queue_tree()
        self._enable_run()
        # Reset progress indicators
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
    def _selected_paths(self) -> list[str]:
        iids = list(self.queue_tree.selection())
        out: list[str] = []
        for iid in iids:
            p = self._iid_to_path.get(iid)
            if p:
                out.append(p)
        return out

    def _remove_selected(self):
        sel = set(self._selected_paths())
        if not sel:
            return
        self.files = [f for f in self.files if f not in sel]
        for f in sel:
            self.file_status.pop(f, None)
        self._refresh_queue_tree()
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
        self._refresh_queue_tree()
        self._enable_run()

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

        # Initialize control flags and UI
        self._control = _Control()
        self._group_done = 0
        self._group_total = 0
        self._group_start_time = None
        self.run_btn["state"] = "disabled"
        self.pause_btn.config(state='normal', text='Pause')
        self.cancel_btn.config(state='normal')
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
            base = Path(name).name if name else "-"
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
                if name in self.file_status:
                    if i >= n:
                        self._set_file_status(name, 'done')
                    else:
                        self._set_file_status(name, 'running')
            except Exception:
                pass
            # Smooth overall: include current file fraction + ETA + throughput
            try:
                done_files = int(getattr(self, '_group_done', 0) or 0)
                total_files = int(getattr(self, '_group_total', 0) or 0) or 1
                frac = min(1.0, max(0.0, (i or 0) / float(n)))
                overall = (done_files + frac) * 100.0 / total_files
                self.progress["maximum"] = 100
                self.progress["value"] = overall
                gst = getattr(self, '_group_start_time', None)
                label = f"File {min(done_files+1, total_files)}/{total_files}: {base} - {pct}%{eta_txt}"
                if gst:
                    gelapsed = max(0.001, now - gst)
                    units = done_files + frac
                    rate = units / gelapsed
                    eta_total = max(0.0, (total_files - units) / max(rate, 1e-9))
                    gmm = int(eta_total // 60)
                    gss = int(eta_total % 60)
                    fpm = rate * 60.0
                    label += f" | Group ETA {gmm:02d}:{gss:02d} | {fpm:.2f} files/min"
                self.overall_label["text"] = label
            except Exception:
                pass

        def worker():
            try:
                # Reset status
                self.after(0, lambda: self._set_status('Ready'))
                self.after(0, lambda: self._log("Launching enhancer..."))
                # Build groups: batch by folder or single group
                files_all = list(self.files)
                groups: list[tuple[str, list[str]]] = []
                if self.var_batch_folders.get():
                    by_parent: dict[str, list[str]] = {}
                    for fp in files_all:
                        parent = str(Path(fp).parent)
                        by_parent.setdefault(parent, []).append(fp)
                    groups = sorted(by_parent.items(), key=lambda kv: kv[0].lower())
                else:
                    groups = [("(all)", files_all)]

                total_groups = len(groups)
                for gi, (gname, gfiles) in enumerate(groups, start=1):
                    if not gfiles:
                        continue
                    self._group_done = 0
                    self._group_total = len(gfiles)
                    self._group_start_time = time.time()
                    self.after(0, lambda gname=gname, gi=gi, total_groups=total_groups: self._log(f"Processing group {gi}/{total_groups}: {gname} ({len(gfiles)} files)"))
                    # group-specific progress wrapper
                    def update_prog_group(done, total, gi=gi, total_groups=total_groups):
                        total = max(total, 1)
                        pct = int(done * 100 / total)
                        self.progress["maximum"] = 100
                        self.progress["value"] = pct
                        self._group_done = done
                        self._group_total = total
                        self.overall_label["text"] = f"Group {gi}/{total_groups}: {done} of {total} files ({pct}%)"

                    # Enhance this group
                    results = run_enhancer_for(
                        gfiles,
                        device="cuda",
                        profile=self.var_profile.get(),
                        progress_cb=lambda d, t: self.after(0, update_prog_group, d, t),
                        chunk_progress_cb=lambda name, i, n: self.after(0, update_chunk, name, i, n),
                        seam_safe=self.var_seam_safe.get(),
                        control=self._control,
                    )
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
                    if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                        break
                    if not self.var_sync_export.get():
                        self.after(0, lambda results=results: self._append_history(results))

                    # Post-processing per group
                    if self.var_postproc.get() and results:
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
                            _postprocess_level_brighten(outs, progress_cb=lambda i, n, m: self.after(0, pp_prog, i, n, m))
                            self.after(0, lambda: self._log("Level + brighten applied."))
                        except Exception as e:
                            if not isinstance(e, _Cancelled):
                                self.after(0, lambda e=e: self._log(f"Post-process error: {e}"))
                        if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                            break

                    # Seam smoothing per group
                    if self.var_seam_safe.get() and results:
                        try:
                            self.after(0, lambda: self._set_status('Scanning seams'))
                            outs = [out for _, out in results]
                            def ss_prog(i, n, msg):
                                n = max(1, n)
                                pct = int(i * 100 / n)
                                self._set_status(f"{msg} - {pct}%")
                                if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                                    raise _Cancelled()
                            _seam_smooth_files(outs, progress_cb=lambda i, n, m: self.after(0, ss_prog, i, n, m))
                            self.after(0, lambda: self._log("Seam smoothing complete."))
                        except Exception as e:
                            if not isinstance(e, _Cancelled):
                                self.after(0, lambda e=e: self._log(f"Seam smoothing warning: {e}"))
                        if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                            break

                    # Sync + export per group
                    if self.var_sync_export.get() and results:
                        try:
                            self.after(0, lambda: self._set_status('Preparing sync'))
                            self.after(0, lambda: self._log("Syncing with Audalign and exporting multichannel..."))
                            outs = [out for _, out in results]
                            def stage_prog(i, n, msg):
                                n = max(1, n)
                                pct = int(i * 100 / n)
                                self._set_status(f"{msg} - {pct}%")
                                if self._control.cancel_now.is_set() or self._control.stop_after_chunk.is_set():
                                    raise _Cancelled()
                            out_path = _sync_and_export_multichannel(
                                outs,
                                prefer_48k=self.var_profile.get(),
                                log=lambda m: self.after(0, self._log, m),
                                progress_cb=lambda i, n, m: self.after(0, stage_prog, i, n, m),
                                wav_only=self.var_fast_wav.get(),
                                skip_fine_align=self.var_skip_fine.get(),
                                use_bw64=self.var_bw64.get(),
                                out_base_dir=gname,
                            )
                            # Cleanup per-file enhanced dirs for this group
                            try:
                                import shutil as _sh
                                parents = {str(Path(p).parent) for p in outs}
                                for d in parents:
                                    if Path(d).name.startswith('Enhanced_'):
                                        _sh.rmtree(d, ignore_errors=True)
                                self.after(0, lambda: self._log("Cleaned per-file enhanced folders."))
                            except Exception as _e:
                                self.after(0, lambda _e=_e: self._log(f"Cleanup warning: {_e}"))
                            if out_path:
                                ch = 0
                                try:
                                    if str(out_path).lower().endswith('.wav'):
                                        import torchaudio as _ta
                                        info = _ta.info(out_path)
                                        ch = getattr(info, 'num_channels', 0) or 0
                                except Exception:
                                    ch = 0
                                label = f"Multichannel ({ch} ch)" if ch else "Multichannel"
                                self.after(0, lambda: self._append_history([(gname, out_path)]))
                                self.after(0, lambda: self._log(f"Group {gi}: multichannel export written: {out_path}"))
                            else:
                                self.after(0, lambda: self._log("Multichannel export failed: no output produced"))
                        except Exception as e:
                            if not isinstance(e, _Cancelled):
                                self.after(0, lambda e=e: self._log(f"Sync/export error: {e}"))
                # Clear the queue after successful enhance
                self.after(0, self._clear_queue)
            except Exception as e:  # noqa: BLE001
                self.after(0, lambda e=e: self._log(f"Error: {e}"))
            finally:
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
            self.pause_btn.configure(text='Pause')
            self._log('Resumed.')
        else:
            self._control.pause.set()
            self.pause_btn.configure(text='Resume')
            self._log('Pausing after current chunk...')

    def _cancel_graceful(self):
        if not hasattr(self, '_control'):
            return
        self._control.stop_after_chunk.set()
        self.cancel_btn.config(state='disabled')
        self._log('Will stop after current chunk...')


# --- Alignment and multichannel export helpers (Audalign-based) ---

def _sync_and_export_multichannel(file_paths: list[str], prefer_48k: bool = True, log=None, progress_cb=None, wav_only: bool = False, skip_fine_align: bool = False, use_bw64: bool = True, out_base_dir: str | None = None) -> str | None:
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
    # Silence tqdm in audalign to prevent blocking on subsequent runs
    os.environ.setdefault('TQDM_DISABLE', '1')
    if log:
        log('Building proxies for alignment...')

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    first_parent = Path(file_paths[0]).parent
    # Intermediate working dir to avoid leaving per-file outputs
    tmp_dir = (Path.cwd() / ".enhancer_runs_gui" / "tmp_sync" / stamp)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Final folder should contain ONLY the combined multichannel file.
    # Place it in specified base dir or alongside originals (not inside Enhanced_*), so cleanup won't remove it.
    if out_base_dir:
        base_dir = Path(out_base_dir)
    else:
        base_dir = first_parent.parent if first_parent.name.startswith('Enhanced_') else first_parent
    final_dir = base_dir / f"Synced_{stamp}"
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

    # Build low-sample-rate proxies for faster alignment
    proxy_sr = 16000
    proxy_dir = tmp_dir / "proxies"
    proxy_dir.mkdir(parents=True, exist_ok=True)

    from torchaudio.functional import resample as ta_resample

    proxies: list[Path] = []
    total_units = 0
    # Proxies count as len(files)
    total_units += len(file_paths)
    completed_units = 0
    def step(i_inc=1, msg=''):
        nonlocal completed_units
        completed_units += i_inc
        if progress_cb:
            progress_cb(completed_units, total_units, msg)

    for idx, p in enumerate(file_paths, start=1):
        wav, sr = torchaudio.load(str(p))
        # downmix to mono for alignment
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0)
        else:
            wav = wav.squeeze(0)
        if sr != proxy_sr:
            wav = ta_resample(wav, orig_freq=sr, new_freq=proxy_sr)
        outp = proxy_dir / Path(p).name
        torchaudio.save(str(outp), wav.unsqueeze(0), proxy_sr)
        proxies.append(outp)
        step(1, f'Proxy {idx}/{len(file_paths)}')

    # 1) Coarse alignment (prefer fast correlation) using proxies
    results = None
    # Coarse align + fine align + write shifts + assemble + convert + cleanup = +6 units (approx)
    total_units += 6
    try:
        align_files = getattr(ad, 'align_files')
        with _suppress_stdout_stderr():
            if hasattr(ad, 'CorrelationRecognizer'):
                corr_rec = getattr(ad, 'CorrelationRecognizer')()
                results = align_files(*[str(p) for p in proxies], recognizer=corr_rec)
            if not results and fingerprint_rec:
                results = align_files(*[str(p) for p in proxies], recognizer=fingerprint_rec)
    except Exception:
        results = None
    if log:
        log('Coarse alignment complete.' if results else 'Coarse alignment failed, attempting fallback...')
    step(1, 'Coarse alignment')

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
            if skip_fine_align:
                if log:
                    log('Skipping fine alignment (user option).')
                step(1, 'Fine alignment (skipped)')
            else:
                # Only fine-align for long content (> 5 min) or many files
                proxy_info = torchaudio.info(str(proxies[0]))
                dur_sec = float(getattr(proxy_info, 'num_frames', 0)) / float(getattr(proxy_info, 'sample_rate', proxy_sr) or proxy_sr)
                if len(file_paths) > 2 or dur_sec > 300:
                    if log:
                        log('Running fine alignment...')
                    with _suppress_stdout_stderr():
                        results = ad.fine_align(results, recognizer=fine_rec)
                    if log:
                        log('Fine alignment complete.')
                step(1, 'Fine alignment')
    except Exception:
        pass

    # 2.5) Estimate and correct drift on originals using windowed correlation (proxies)
    try:
        if len(file_paths) >= 2:
            ref_proxy = proxies[0]
            drift_dir = tmp_dir / "driftcorr"
            drift_dir.mkdir(parents=True, exist_ok=True)
            corrected_files: list[str] = [file_paths[0]]
            for i in range(1, len(file_paths)):
                ratio = _estimate_drift_ratio(str(ref_proxy), str(proxies[i]), proxy_sr)
                # Correct if outside tiny tolerance (> 50 ppm)
                if abs(1.0 - ratio) > 5e-5:
                    src = Path(file_paths[i])
                    dst = drift_dir / src.name
                    _resample_with_ratio(str(src), str(dst), ratio)
                    corrected_files.append(str(dst))
                else:
                    corrected_files.append(file_paths[i])
            # Use corrected list for writing shifts/export
            file_paths = corrected_files
    except Exception:
        pass

    # 3) Write aligned, padded mono files to out_dir
    wrote_aligned = False
    # Preferred: write shifts from results
    try:
        if results is not None and hasattr(ad, 'write_shifts_from_results'):
            if log:
                log('Writing aligned files (temp)...')
            with _suppress_stdout_stderr():
                ad.write_shifts_from_results(results, str(tmp_dir), file_paths)
            wrote_aligned = True
            step(1, 'Write aligned files')
    except Exception:
        wrote_aligned = False
    # Fallback: directly call align_files with destination_path
    if not wrote_aligned:
        try:
            align_files = getattr(ad, 'align_files')
            with _suppress_stdout_stderr():
                if fingerprint_rec:
                    align_files(*file_paths, destination_path=str(tmp_dir), recognizer=fingerprint_rec)
                else:
                    align_files(*file_paths, destination_path=str(tmp_dir))
            wrote_aligned = True
            step(1, 'Write aligned files (fallback)')
        except Exception:
            wrote_aligned = False

    if not wrote_aligned:
        return None

    # 4) Load aligned files from tmp_dir and build multichannel tensor
    if log:
        log('Assembling multichannel file...')
    step(1, 'Assemble multichannel')
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

    # Load, resample to target SR first, then compute max length to avoid truncation
    resampled = []
    srs = []
    for p in aligned_paths:
        wav, sr = torchaudio.load(str(p))
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)  # downmix to mono
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        srs.append(int(sr))
        resampled.append((wav, int(sr)))

    # Choose target SR
    target_sr = 48000 if prefer_48k else (srs[0] if srs else 48000)
    from torchaudio.functional import resample as ta_resample

    # Resample and determine max length in target domain
    monos = []
    max_len = 0
    for wav, sr in resampled:
        mono = wav[0]
        if sr != target_sr:
            mono = ta_resample(mono, orig_freq=sr, new_freq=target_sr)
        monos.append(mono)
        if mono.size(-1) > max_len:
            max_len = int(mono.size(-1))

    # Pad all to max_len (no truncation)
    chan_tensors = []
    for mono in monos:
        cur_len = mono.size(-1)
        if cur_len < max_len:
            mono = torch.nn.functional.pad(mono, (0, max_len - cur_len))
        chan_tensors.append(mono.unsqueeze(0))

    if not chan_tensors:
        return None

    multich = torch.cat(chan_tensors, dim=0)
    n_ch = int(multich.size(0))
    out_wav = final_dir / f"Synced_Multichannel_{stamp}.wav"
    torchaudio.save(str(out_wav), multich, target_sr)

    # Optional: clear channel mask and encode as PCM 24-bit for Premiere dual-mono import behavior
    # Removed redundant python subprocess check to prevent potential stalls
    try:
        ff = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
        if ff and not wav_only:
            # Build a MOV with N mono audio streams (dual-/multi-mono), best for Premiere
            out_mov = final_dir / f"Synced_Multichannel_{stamp}.mov"
            if n_ch == 2:
                filt = "[0:a]channelsplit=channel_layout=stereo[L][R]"
                cmd = [
                    ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(out_wav),
                    '-filter_complex', filt,
                    '-map', '[L]', '-c:a:0', 'pcm_s24le', '-ac:a:0', '1',
                    '-map', '[R]', '-c:a:1', 'pcm_s24le', '-ac:a:1', '1',
                    str(out_mov)
                ]
            else:
                # General case: split each input channel to its own mono stream
                parts = [f"[0:a]pan=mono|c0=c{idx}[ch{idx}]" for idx in range(n_ch)]
                filt = ";".join(parts)
                cmd = [ff, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(out_wav), '-filter_complex', filt]
                for idx in range(n_ch):
                    cmd += ['-map', f'[ch{idx}]', f'-c:a:{idx}', 'pcm_s24le', f'-ac:a:{idx}', '1']
                cmd += [str(out_mov)]
            if log:
                log('Converting to dual-/multi-mono MOV...')
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            except subprocess.TimeoutExpired:
                # Fallback: keep WAV
                out_mov = None
            if out_mov and out_mov.exists() and out_mov.stat().st_size > 44:
                try:
                    out_wav.unlink(missing_ok=True)
                except Exception:
                    pass
                out_wav = out_mov
            step(1, 'Convert container')
        else:
            if wav_only and log:
                log('WAV-only export selected; finalizing WAV metadata...')
            # Prefer BW64 with ADM if requested
            bw_ok = False
            if use_bw64:
                if log:
                    log('Attempting BW64 (ADM) export...')
                try:
                    if _export_bw64_with_adm(str(out_wav), [Path(p).stem for p in aligned_paths], target_sr):
                        bw_ok = True
                        if log:
                            log('BW64 (ADM) export complete.')
                except Exception as _e:
                    bw_ok = False
                    if log:
                        log(f'BW64 export failed, falling back to BWF iXML: {_e}')
            if not bw_ok:
                # Ensure dual-mono behavior by clearing channel mask and embedding iXML names
                try:
                    _finalize_wav_dual_mono(str(out_wav), [Path(p).stem for p in aligned_paths], target_sr)
                except Exception:
                    pass
            step(1, 'Finalize WAV metadata')
    except Exception:
        # Keep WAV if ffmpeg conversion fails
        pass

    # Cleanup tmp_dir completely so only the combined file remains
    try:
        import shutil as _sh
        _sh.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    step(1, 'Cleanup')

    return str(out_wav)


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
    # Fit lag(samples) = a * t(sec) + b
    a, b = np.polyfit(np.asarray(ts), np.asarray(lags), 1)
    # Drift ratio: resample other by (1 - a/sr)
    ratio = 1.0 - (a / float(sr))
    return float(ratio)


def _gcc_phat_lag(x, y) -> int:
    import numpy as np
    n = int(2 ** np.ceil(np.log2(len(x) + len(y))))
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
    import os
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
        from bw64 import write_bw64
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
            torchaudio.save(str(p), fixed, sr)
        except Exception:
            pass
        if progress_cb:
            progress_cb(i, total, f'Seam smoothing {i}/{total}')

    


def _postprocess_level_brighten(paths: list[str], target_lufs: float = -16.0, range_min: float = -20.0, range_max: float = -14.0, progress_cb=None) -> None:
    """Simple, safe loudness normalization with gentle brightening.

    Steps per file:
    - Convert to float32, remove DC (HPF 20 Hz)
    - Apply gentle high-mid/air peaking boosts (+1.0 dB @3.5 kHz, +0.8 dB @8 kHz)
    - Compute single uniform gain based on RMS target and peak ceiling
      gain = min(target_rms / rms, ceiling / peak), capped to +4 dB max
    - Apply gain and save
    """
    import math
    import torchaudio
    import torch
    try:
        import pyloudnorm as pyln
    except Exception:
        pyln = None

    from torchaudio.functional import equalizer_biquad, highpass_biquad

    # Targets
    ceiling_db = -6.0
    ceiling = 10 ** (ceiling_db / 20.0)
    target_rms_db = -16.0
    target_rms = 10 ** (target_rms_db / 20.0)
    max_boost_db = 4.0
    max_boost = 10 ** (max_boost_db / 20.0)

    total = max(1, len(paths))
    for idx, p in enumerate(paths, start=1):
        if progress_cb:
            progress_cb(idx-1, total, f'Post process {idx}/{total}')
        wav, sr = torchaudio.load(str(p))
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        # DC removal
        try:
            wav = highpass_biquad(wav, sr, cutoff_freq=20.0, Q=0.707)
        except Exception:
            pass
        # Gentle brightening
        try:
            wav = equalizer_biquad(wav, sr, center_freq=3500.0, gain=1.0, Q=0.707)
            wav = equalizer_biquad(wav, sr, center_freq=8000.0, gain=0.8, Q=0.707)
        except Exception:
            pass

        # Reference loudness (mono mix)
        mono = wav.mean(0) if wav.dim() == 2 else wav
        rms = float(torch.sqrt(torch.mean(mono * mono) + 1e-12))
        peak = float(torch.max(torch.abs(wav)))

        # Optional LUFS correction for short clips only
        dur = float(wav.size(-1)) / float(sr or 1)
        if pyln is not None and dur <= 60.0:
            try:
                meter = pyln.Meter(sr)
                lufs = float(meter.integrated_loudness(mono.numpy()))
                if not (range_min <= lufs <= range_max):
                    # convert LUFS delta to linear scale and cap
                    gain_db = max(-6.0, min(4.0, target_lufs - lufs))
                    wav = wav * float(10 ** (gain_db / 20.0))
                    mono = wav.mean(0) if wav.dim() == 2 else wav
                    rms = float(torch.sqrt(torch.mean(mono * mono) + 1e-12))
                    peak = float(torch.max(torch.abs(wav)))
            except Exception:
                pass

        # Single safe gain
        gain_rms = target_rms / max(rms, 1e-9)
        gain_peak = ceiling / max(peak, 1e-9)
        gain = min(gain_rms, gain_peak, max_boost)
        if gain <= 0:
            gain = 1.0
        wav = wav * float(gain)

        if progress_cb:
            progress_cb(idx, total, f'Post process {idx}/{total}')

        torchaudio.save(str(p), wav, sr)


if __name__ == "__main__":
    app = App()
    app.mainloop()
