"""Эксперимент «робастность» на линейном стенде.

Три протокола (все с фиксированными seed, CPU-only):

  1. Развёртка амплитуды возмущения d (ступень потребителя на [40, 80) с):
     violation_count(d) для 5 конфигураций; робастностный запас — максимальная
     проверенная амплитуда с нулём нарушений. Для homeo_no_safety_filter
     safety-фильтр проектируется с d_bound = d (известная граница проекта).
     Конфигурации: homeo (полная), homeo без safety-инвариантов,
     homeo без safety-инвариантов + safety-фильтр, ПИД (сетка), LQR.

  2. Развёртка шума датчиков sigma при номинальном возмущении d = 0.35:
     homeo / ПИД / LQR.

  3. Дрейф параметра объекта: коэффициент стока c2(t) линейно падает на 30 %
     за эпизод (засорение дренажа) при d = 0.35 — homeo / ПИД / LQR.

Запуск:  python experiments/run_robustness.py
Результат: results/robustness.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (lin_plant, lin_model, lin_scenario, make_homeo,
                                make_safety, episode_metrics, save_json,
                                LIN_STEPS, DT)
from experiments.lqr import LQRTankController, pid_gains_from_json
from homeo.control import PIDTankController, run_episode, scenario_constraints
from homeo.metrics import violation_count

D_GRID = [round(0.1 * i, 2) for i in range(1, 13)]        # 0.1 .. 1.2, шаг 0.1
SIGMA_GRID = [0.0, 0.01, 0.02, 0.05]
DRIFT_FRACTION = 0.30


def c2_drift_hook(fraction):
    """Хук дрейфа: c2(t) = c2_0·(1 − fraction·t/T)."""
    def hook(plant, k, t):
        plant.c2 = plant._c2_0 * (1.0 - fraction * t / (LIN_STEPS * DT))
    return hook


def make_drift_plant():
    p = lin_plant()
    p._c2_0 = p.c2
    return p


def main():
    from experiments.lqr import design_linear_stand
    design_linear_stand()  # проверка DARE до замеров
    model = lin_model()
    out = {}

    # ---- 1. развёртка амплитуды возмущения ----
    print("== Робастность: развёртка амплитуды ступени потребителя ==")
    sweep = {}
    for name in ("homeo", "homeo_no_safety", "homeo_no_safety_filter", "pid", "lqr"):
        viol = []
        for d in D_GRID:
            scen = lin_scenario(d_amp=d)
            if name == "homeo":
                ctrl = make_homeo(model, scen)
                safety = None
            elif name == "homeo_no_safety":
                ctrl = make_homeo(model, scen, use_safety_invariants=False)
                safety = None
            elif name == "homeo_no_safety_filter":
                ctrl = make_homeo(model, scen, use_safety_invariants=False)
                safety = make_safety(model, scen, d_bound=d, margin=0.0)
            elif name == "pid":
                ctrl = PIDTankController(scen, *pid_gains_from_json())
                safety = None
            else:
                ctrl = LQRTankController(scen)
                safety = None
            ep = run_episode(lin_plant(), ctrl, scen, noise_seed=7, safety=safety)
            viol.append(violation_count(scenario_constraints(ep, scen)))
        best = 0
        for d, n in zip(D_GRID, viol):
            if n == 0:
                best = d
        sweep[name] = {"amplitudes": D_GRID, "violation_count": viol,
                       "zero_violation_margin": best,
                       "note": "запас = максимальная ПРОВЕРЕННАЯ амплитуда с 0 нарушений "
                               "(сетка шаг 0.1; для homeo_no_safety_filter d_bound = d)"}
        print(f"  {name:24s} запас = {best}")
    out["disturbance_sweep"] = sweep

    # ---- 2. развёртка шума датчиков ----
    print("== Робастность: развёртка СКО шума датчиков (d = 0.35) ==")
    noise = {}
    for name in ("homeo", "pid", "lqr"):
        rows = []
        for sigma in SIGMA_GRID:
            scen = lin_scenario(sigma=sigma)
            if name == "homeo":
                ctrl = make_homeo(model, scen)
            elif name == "pid":
                ctrl = PIDTankController(scen, *pid_gains_from_json())
            else:
                ctrl = LQRTankController(scen)
            ep = run_episode(lin_plant(), ctrl, scen, noise_seed=7)
            rows.append({"sigma": sigma, **episode_metrics(ep, scen)})
        noise[name] = rows
        print(f"  {name:6s}: " + ", ".join(
            f"σ={r['sigma']}: viol={r['violation_count']}" for r in rows))
    out["sensor_noise_sweep"] = noise

    # ---- 3. дрейф параметра объекта ----
    print("== Робастность: дрейф c2 (сток) −30 % за эпизод, d = 0.35 ==")
    drift = {}
    scen = lin_scenario()
    for name in ("homeo", "pid", "lqr"):
        if name == "homeo":
            ctrl = make_homeo(model, scen)
        elif name == "pid":
            ctrl = PIDTankController(scen, *pid_gains_from_json())
        else:
            ctrl = LQRTankController(scen)
        ep = run_episode(make_drift_plant(), ctrl, scen, noise_seed=7,
                         on_step=c2_drift_hook(DRIFT_FRACTION))
        m = episode_metrics(ep, scen)
        drift[name] = m
        print(f"  {name:6s}: viol={m['violation_count']:4d} depth={m['max_violation_depth']:.4f} "
              f"overshoot={m['overshoot']:.3f} effort={m['control_effort']:.1f}")
    out["param_drift_c2"] = {"fraction": DRIFT_FRACTION, "results": drift}

    save_json("robustness.json", out)


if __name__ == "__main__":
    main()
