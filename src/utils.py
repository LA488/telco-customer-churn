# src/utils.py
"""
Вспомогательные функции: создание директорий, сохранение/загрузка моделей.
"""

import os
import joblib


def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    return joblib.load(path)
