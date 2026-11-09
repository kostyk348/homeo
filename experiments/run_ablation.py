"""Эксперимент «абляции» homeo на линейном стенде (d = 0.5, чтобы различия
между вариантами были заметны).

Каждая абляция выключает ОДИН реальный механизм контроллера/ядра:

  * full            — полная конфигурация (Adam, шум t0=0.02, annealing k=1,
                      адаптивный h, проекция в ящик, safety-инварианты);
  * no_noise        — t0 = 0: термальный шум выключен;
  * no_anneal       — anneal_k = 0: шум постоянной амплитуды, без затухания;
  * fixed_gain      — вместо Adam градиентный спуск с фиксированным шагом
                      (FixedGainHomeoEngine) — абляция «без адаптивного шага»;
  * no_adaptive_h   — h_doublings = 0: численный градиент всегда с h = grad_eps;
  * no_projection   — project = False: релаксация без обрезки в ящик [−1,1]
                      (привод физически ограничен — как у всех контроллеров);
  * no_safety       — инварианты полос выброшены, остаётся только слежение
                      и экономия управления.

Запуск:  python experiments/run_ablation.py
Результат: results/ablation.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (lin_plant, lin_model, lin_scenario, make_homeo,
                                episode_metrics, save_json, HOMEO_KW)
from homeo.control import FixedGainHomeoEngine, run_episode

D_ABLATION = 0.5


def main():
    model = lin_model()
    scen = lin_scenario(d_amp=D_ABLATION)

    variants = [
        ("full", {}),
        ("no_noise", {"t0": 0.0}),
        ("no_anneal", {"anneal_k": 0.0}),
        ("fixed_gain", {"engine_cls": FixedGainHomeoEngine}),
        ("no_adaptive_h", {"engine_kwargs": {"h_doublings": 0}}),
        ("no_projection", {"engine_kwargs": {"project": False}}),
        ("no_safety", {"use_safety_invariants": False}),
    ]

    out = {"scenario_d": D_ABLATION, "variants": {}}
    print("== Абляции homeo на линейном стенде ==")
    header = f"  {'вариант':14s} viol  depth  overshoot settling  effort  triggers"
    print(header)
    for name, over in variants:
        ctrl = make_homeo(model, scen, **over)
        ep = run_episode(lin_plant(), ctrl, scen, noise_seed=7)
        m = episode_metrics(ep, scen)
        overridden = {k: (v.__name__ if isinstance(v, type) else v) for k, v in over.items()}
        out["variants"][name] = {"overridden": overridden, **m}
        st = "—" if m["settling_time"] is None else f"{m['settling_time']:7.2f}"
        print(f"  {name:14s} {m['violation_count']:5d} {m['max_violation_depth']:6.3f} "
              f"{m['overshoot']:9.3f}  {st} {m['control_effort']:7.1f} {m['filter_triggers']:8d}")

    save_json("ablation.json", out)


if __name__ == "__main__":
    main()
