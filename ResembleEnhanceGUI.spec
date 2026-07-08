# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

project_root = Path.cwd()

binaries = []

# Bundle ffmpeg tools so media transcode features work on clean PCs.
ffmpeg_candidates = [
    project_root / "vendor" / "ffmpeg_bin",
    project_root / "ffmpeg_bin",
    Path(r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin"),
]
for ffmpeg_bin_dir in ffmpeg_candidates:
    if all((ffmpeg_bin_dir / tool).exists() for tool in ("ffmpeg.exe", "ffprobe.exe")):
        for tool in ("ffmpeg.exe", "ffprobe.exe"):
            binaries.append((str(ffmpeg_bin_dir / tool), "ffmpeg_bin"))
        break
else:
    print("WARNING: ffmpeg.exe / ffprobe.exe were not found; build will rely on system ffmpeg at runtime.")

# Ensure torchaudio native libraries are collected.
binaries += collect_dynamic_libs("torchaudio")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")

numpy_lib_dir = project_root / ".venv" / "Lib" / "site-packages" / "numpy.libs"
if numpy_lib_dir.exists():
    for dll in numpy_lib_dir.glob("*.dll"):
        binaries.append((str(dll), "numpy.libs"))

source_datas = [
    (str(p), str(p.parent.relative_to(project_root)))
    for p in (project_root / "resemble_enhance").rglob("*.py")
]


a = Analysis(
    ['enhancer_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('resemble_enhance\\model_repo', 'resemble_enhance\\model_repo'),
        ('assets\\icons', 'assets\\icons'),
        *source_datas,
    ],
    hiddenimports=[
        'resemble_enhance.enhancer.__main__',
        'resemble_enhance.enhancer.inference',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_runtime_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


def _keep_collected_item(item):
    src, dest, typ = item
    src_s = str(src).replace("\\", "/").lower()
    dest_s = str(dest).replace("\\", "/").lower()
    blocked_parts = (
        "/.enhancer_runs_gui/",
        "/output_audio/",
        "/__pycache__/",
        "/.pytest_cache/",
        "/torch/include/",
        "/torch/share/cmake/",
    )
    if any(part in src_s or part in dest_s for part in blocked_parts):
        return False
    # PyInstaller sometimes collects static import libraries from Torch.
    # Runtime only needs DLL/PYD binaries; .lib files massively inflate installers.
    if src_s.endswith(".lib") or dest_s.endswith(".lib"):
        return False
    return True


a.binaries = type(a.binaries)([item for item in a.binaries if _keep_collected_item(item)])
a.datas = type(a.datas)([item for item in a.datas if _keep_collected_item(item)])
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ResembleEnhanceGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResembleEnhanceGUI',
)
