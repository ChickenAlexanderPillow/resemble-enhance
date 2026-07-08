# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

project_root = Path.cwd()

binaries = []

# Bundle local ffmpeg tools so media transcode features work on clean PCs.
ffmpeg_bin_dir = Path(r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin")
for tool in ("ffmpeg.exe", "ffprobe.exe"):
    tool_path = ffmpeg_bin_dir / tool
    if tool_path.exists():
        binaries.append((str(tool_path), "ffmpeg_bin"))

# Ensure torchaudio native libraries are collected.
binaries += collect_dynamic_libs("torchaudio")
binaries += collect_dynamic_libs("numpy")
binaries += collect_dynamic_libs("scipy")

numpy_lib_dir = project_root / ".venv" / "Lib" / "site-packages" / "numpy.libs"
if numpy_lib_dir.exists():
    for dll in numpy_lib_dir.glob("*.dll"):
        binaries.append((str(dll), "numpy.libs"))


a = Analysis(
    ['enhancer_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=[
        ('resemble_enhance\\model_repo', 'resemble_enhance\\model_repo'),
        ('assets\\icons', 'assets\\icons'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyinstaller_runtime_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
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
