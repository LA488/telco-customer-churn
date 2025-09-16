# TelcoApp.spec
# Запуск: pyinstaller TelcoApp.spec

# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# 🔹 Модули sklearn и ttkbootstrap
hiddenimports = (
    collect_submodules("sklearn") +
    collect_submodules("ttkbootstrap")
)

# 🔹 Пути
project_dir = os.getcwd()
app_dir = os.path.join(project_dir, "app")

# 🔹 Данные
datas = [
    ("models/RandomForest_pipeline.pkl", "models"),
    ("data/processed/telco_churn_processed.csv", "data/processed"),
]

a = Analysis(
    [os.path.join(app_dir, "main.py")],  # app/main.py
    pathex=[app_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    [],
    exclude_binaries=True,
    name="TelcoApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TelcoApp",
    distpath=os.path.abspath(os.path.join(app_dir, "dist")),   # 🔹 абс. путь
    workpath=os.path.abspath(os.path.join(app_dir, "build")), # 🔹 абс. путь
)
