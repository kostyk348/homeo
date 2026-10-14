"""Общая настройка тестов: корень репозитория и каталог examples в sys.path.

Нужно, чтобы `pytest` работал и без `pip install .` (ядро — чистый Python без
зависимостей) и чтобы тесты могли переиспользовать физическую модель из
examples/driven_oscillator.py, а не дублировать её.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for path in (ROOT, os.path.join(ROOT, "examples")):
    if path not in sys.path:
        sys.path.insert(0, path)
