# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

spec_location = Path(SPECPATH).resolve()
spec_dir = spec_location.parent if spec_location.is_file() else spec_location
project_root = spec_dir.parent
entry_point = (project_root / "app" / "main.py").resolve()
if not entry_point.is_file():
    raise FileNotFoundError(f"FRIDAY entry point not found: {entry_point}")

analysis = Analysis(
    [str(entry_point)],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="FRIDAY",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FRIDAY",
)
