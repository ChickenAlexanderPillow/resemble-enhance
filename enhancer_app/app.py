import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, render_template, request


APP_VERSION = "0.1.1"
app = Flask(__name__, static_folder="static", template_folder="templates")


BASE_DIR = Path.cwd()
INPUT_ROOT = BASE_DIR / "input_audio"
OUTPUT_ROOT = BASE_DIR / "output_audio"
RUNS_DIR = BASE_DIR / ".enhancer_runs"


RUNS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class Job:
    def __init__(self, job_id: str, files: list[Path]):
        self.id = job_id
        self.files = files
        self.input_dir = RUNS_DIR / job_id / "input_audio"
        self.output_base = OUTPUT_ROOT
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.output_base / self.timestamp
        self.state = "queued"  # queued|running|done|failed
        self.error: Optional[str] = None
        self.total = len(files)


jobs: Dict[str, Job] = {}


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _compute_wav_duration_sec(path: Path) -> Optional[float]:
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


def _run_enhance(job: Job) -> None:
    job.state = "running"
    try:
        # Prepare directories
        job.input_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy files into isolated input_dir preserving relative names
        saved_paths: list[Path] = []
        for f in job.files:
            rel = f.name
            dst = job.input_dir / rel
            _ensure_parent(dst)
            data = f.read_bytes()
            dst.write_bytes(data)
            saved_paths.append(dst)
            # Remove temp upload after copying
            try:
                f.unlink(missing_ok=True)  # py310+ on Windows is fine
            except TypeError:
                # Fallback for older Python
                try:
                    if f.exists():
                        f.unlink()
                except Exception:
                    pass

        # Build enhance command mirroring run_enhancer.py behavior (camera_sync friendly)
        py = sys.executable or "python"
        cmd = [
            py,
            "-m",
            "resemble_enhance.enhancer",
            str(job.input_dir),
            str(job.output_dir),
            "--denoise_only",
            "--device",
            "cuda",
            "--profile",
            "camera_sync",
        ]

        import subprocess
        print("[enhancer_app] running:", " ".join(cmd))
        proc = subprocess.run(cmd, env=os.environ)

        if proc.returncode != 0:
            job.state = "failed"
            job.error = f"Enhancer returned code {proc.returncode}"
            return

        # On success, validate outputs and delete inputs
        deleted = 0
        for src in saved_paths:
            dst = job.output_dir / src.name
            if not dst.exists() or dst.stat().st_size <= 44:
                continue
            din = _compute_wav_duration_sec(src)
            dout = _compute_wav_duration_sec(dst)
            if din is not None and dout is not None and abs(din - dout) > 1e-3:
                continue
            if _safe_delete(src):
                deleted += 1

        # Clean up empty dirs
        _prune_empty_dirs(job.input_dir)

        job.state = "done"
    except Exception as e:  # noqa: BLE001
        import traceback
        job.state = "failed"
        msg = str(e)
        if "spawnvpe" in msg or "spawnlp" in msg:
            msg += " | Hint: Please close all previous app windows and relaunch run_enhancer_app.cmd to load the updated build."
        print("[enhancer_app] job failed:", msg)
        print(traceback.format_exc(limit=5))
        job.error = msg


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    saved: list[Path] = []
    for f in files:
        # Temporarily save to runs root; will be copied into job input dir
        tmp = RUNS_DIR / "_incoming" / f.filename
        _ensure_parent(tmp)
        f.save(tmp)
        saved.append(tmp)

    job_id = uuid.uuid4().hex[:8]
    job = Job(job_id, saved)
    jobs[job_id] = job

    t = threading.Thread(target=_run_enhance, args=(job,), daemon=True)
    t.start()

    return jsonify({"job_id": job_id, "app_version": APP_VERSION})


@app.get("/api/status/<job_id>")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    # Compute progress by counting produced outputs
    completed = 0
    try:
        for f in job.files:
            if (job.output_dir / Path(f).name).exists():
                completed += 1
    except Exception:
        completed = 0
    return jsonify(
        {
            "state": job.state,
            "error": job.error,
            "output_dir": str(job.output_dir),
            "app_version": APP_VERSION,
            "total": job.total,
            "completed": completed,
        }
    )


@app.get("/api/ping")
def ping():
    return jsonify({"ok": True, "app_version": APP_VERSION})


@app.post("/api/open/<job_id>")
def open_folder(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    try:
        path = str(job.output_dir)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["open", path])
        return jsonify({"ok": True})
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


def _open_browser(port: int) -> None:
    import webbrowser

    url = f"http://127.0.0.1:{port}"
    # small delay to ensure server starts
    def _later():
        time.sleep(0.8)
        webbrowser.open(url)

    threading.Thread(target=_later, daemon=True).start()


def main() -> None:
    base_port = int(os.environ.get("ENHANCER_APP_PORT", "5137"))
    port = base_port
    _open_browser(port)
    last_err: Optional[Exception] = None
    for _ in range(3):
        try:
            app.run(host="127.0.0.1", port=port, debug=False)
            return
        except OSError as e:  # address in use, try next port
            last_err = e
            port += 1
            continue
    if last_err:
        raise last_err


if __name__ == "__main__":
    main()
