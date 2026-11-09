"""Прогон ВСЕЙ валидации одной командой: бейзлайны → робастность → абляции → демо.

Складывает частичные результаты (results/baselines.json, robustness.json,
ablation.json, demo.json) в общий results/metrics.json с блоком окружения.

Запуск:  python experiments/run_all.py     (~5–7 минут, CPU-only)
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import environment, RESULTS_DIR  # noqa: E402

PARTS = ("baselines", "robustness", "ablation", "demo")


def main():
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    import experiments.run_baselines as rb
    import experiments.run_robustness as rr
    import experiments.run_ablation as ra
    import experiments.run_demo_tanks as rd
    for mod in (rb, rr, ra, rd):
        name = mod.__name__.rsplit(".", 1)[-1]
        print(f"\n=== {name} ===")
        mod.main()

    metrics = {"environment": environment()}
    for part in PARTS:
        with open(os.path.join(RESULTS_DIR, f"{part}.json"), encoding="utf-8") as f:
            metrics[part] = json.load(f)

    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n=== свод сохранён: {path} (прогон {metrics['runtime_seconds']} с) ===")


if __name__ == "__main__":
    main()
