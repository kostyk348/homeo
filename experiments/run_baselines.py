"""Эксперимент «бейзлайны на линейном стенде»: homeo vs ПИД vs LQR.

Протокол:
  * ПИД настраивается сеткой на ОТДЕЛЬНОМ тюнинговочном сценарии (другой seed
    шума), затем все контроллеры оцениваются на одних и тех же потоках
    возмущений/шума (3 seed'а, метрики усредняются);
  * LQR проектируется один раз (DARE, интегратор ошибки), веса не перенастраиваются;
  * homeo использует параметры из experiments/common.py (HOMEO_KW), сетка не гоняется;
  * метрики: violation_count, max_violation_depth, overshoot, settling_time,
    recovery_time, control_effort (определения — docs/validation.md).

Запуск:  python experiments/run_baselines.py
Результат: results/baselines.json, results/linear_stand_comparison.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import (lin_plant, lin_model, lin_scenario, make_homeo,
                                make_safety, episode_metrics, seed_averaged,
                                save_json, scenario_block, RESULTS_DIR)
from experiments.lqr import LQRTankController, design_linear_stand
from homeo.control import PIDTankController, run_episode

TUNE_SEED = 123          # отдельный поток для настройки ПИД
EVAL_SEEDS = (7, 8, 9)   # потоки оценки, одинаковые для всех контроллеров

GRID_KP = (1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
GRID_KI = (0.05, 0.1, 0.2, 0.4, 0.8)
GRID_KD = (0.0, 0.5, 1.0)


def pid_score(scen, kp, ki, kd, noise_seed):
    ctrl = PIDTankController(scen, kp=kp, ki=ki, kd=kd)
    ep = run_episode(lin_plant(), ctrl, scen, noise_seed=noise_seed)
    m = episode_metrics(ep, scen)
    st = m["settling_time"] if m["settling_time"] is not None else 999.0
    return (m["violation_count"], st, m["control_effort"]), m


def tune_pid():
    scen = lin_scenario()
    best = None
    for kp in GRID_KP:
        for ki in GRID_KI:
            for kd in GRID_KD:
                score, m = pid_score(scen, kp, ki, kd, TUNE_SEED)
                if best is None or score < best[0]:
                    best = (score, (kp, ki, kd), m)
    (score, gains, m) = best
    print(f"  ПИД: сетка {len(GRID_KP) * len(GRID_KI) * len(GRID_KD)} комбинаций, "
          f"лучшие Kp={gains[0]}, Ki={gains[1]}, Kd={gains[2]} "
          f"(на тюн. сценарии: viol={score[0]}, settling={score[1]:.2f}, effort={score[2]:.1f})")
    return gains


def eval_controller(make_ctrl, scen):
    metrics = []
    for seed in EVAL_SEEDS:
        ep = run_episode(lin_plant(), make_ctrl(scen), scen, noise_seed=seed)
        metrics.append(episode_metrics(ep, scen))
    return seed_averaged(metrics)


def main():
    print("== Бейзлайны на линейном стенде (два связанных резервуара, линеаризованные перетоки) ==")
    gains = tune_pid()

    model = lin_model()
    scen = lin_scenario()

    results = {"scenario": scenario_block(
        scen, label="линейный стенд",
        plant_params={"A1": 1.0, "A2": 1.0, "c12": 0.7, "c2": 0.35, "nonlinear": False,
                      "h_start": [0.5, 0.5], "disturbance": "ступень 0.35 м³/с на [40, 80) с"},
    ), "protocol": {"tuning_seed": TUNE_SEED, "eval_seeds": list(EVAL_SEEDS),
                    "note": "ПИД настроен сеткой на отдельном seed; оценка — среднее по 3 seed'ам "
                            "(одни и те же потоки возмущений/шума для всех контроллеров)"}}

    results["homeo"] = {
        "params": {"horizon": 16, "lr": 0.15, "t0": 0.02, "anneal_k": 1.0, "seed": 11,
                   "grad_tol": 1e-4, "max_iter": 80, "w_goal": 1.0, "w_safe": 25.0,
                   "w_eff": 0.05, "safety_margin": 0.03, "d_hat_beta": 0.2},
        "metrics": eval_controller(lambda s: make_homeo(model, s), scen),
    }
    results["pid"] = {
        "gains": {"Kp": gains[0], "Ki": gains[1], "Kd": gains[2],
                  "anti_windup": "условное интегрирование (clamping)"},
        "metrics": eval_controller(lambda s: PIDTankController(s, *gains), scen),
    }
    K, _, _, _ = design_linear_stand()
    results["lqr"] = {
        "design": {"method": "DARE (scipy.linalg.solve_discrete_are), интегратор ошибки",
                   "Q_diag": [2.0, 8.0, 0.4], "R": 0.05, "K": [round(float(v), 4) for v in K.ravel()]},
        "metrics": eval_controller(lambda s: LQRTankController(s), scen),
    }

    save_json("baselines.json", results)
    plot_comparison(gains)
    _print_table(results)


def _fmt(v):
    """Метрика для таблицы: скаляр — число; список (по seed'ам, когда хотя бы один
    None) — среднее по завершённым seed'ам с пометкой «*»; None/пусто — «—»."""
    if v is None:
        return "      —"
    if isinstance(v, list):
        vals = [x for x in v if x is not None]
        if not vals:
            return "      —"
        return f"{sum(vals) / len(vals):7.2f}*"
    return f"{v:7.2f}"


def _print_table(res):
    print("\n  контроллер   viol  depth   overshoot settling recovery effort")
    for name in ("homeo", "pid", "lqr"):
        m = res[name]["metrics"]
        st = _fmt(m["settling_time"])
        rec = _fmt(m["recovery_time"])
        viol = m["violation_count"]
        viol_s = f"{viol:5.1f}" if not isinstance(viol, list) else _fmt(viol)
        print(f"  {name:11s} {viol_s} {m['max_violation_depth']:6.3f} "
              f"{m['overshoot']:9.3f}  {st}  {rec} {m['control_effort']:7.1f}")
    print("  (* — среднее по seed'ам, где метрика определена; «—» — не определена)")


def plot_comparison(gains):
    """Траектории h2 (и h1) трёх контроллеров на одном потоке (seed 7)."""
    scen = lin_scenario()
    model = lin_model()
    runs = [("homeo", make_homeo(model, scen), None),
            ("ПИД", PIDTankController(scen, *gains), None),
            ("LQR", LQRTankController(scen), None)]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, constrained_layout=True)
    lo1, hi1 = scen.h1_band
    lo2, hi2 = scen.h2_band
    for ax, (lo, hi, key, lbl) in zip(axes, [(lo1, hi1, "h1", "уровень бака 1, м"),
                                             (lo2, hi2, "h2", "уровень бака 2, м")]):
        ax.axhspan(lo, hi, color="#d9f0d9", zorder=0, label="допустимая полоса Ω")
        ax.axhline(scen.ref, color="#888", lw=0.8, ls="--", label="уставка")
    for name, ctrl, _ in runs:
        ep = run_episode(lin_plant(), ctrl, scen, noise_seed=7)
        axes[0].plot(ep["t"], ep["h1"], lw=1.4, label=name)
        axes[1].plot(ep["t"], ep["h2"], lw=1.4, label=name)
    axes[1].set_xlabel("время, с")
    for ax, (lo, hi, key, lbl) in zip(axes, [(lo1, hi1, "h1", "уровень бака 1, м"),
                                             (lo2, hi2, "h2", "уровень бака 2, м")]):
        ax.set_ylabel(lbl)
        ax.axvspan(40, 80, color="#f6d3d3", alpha=0.6, zorder=0)
    axes[0].set_title("Линейный стенд: homeo vs ПИД vs LQR (ступень расхода потребителя 40–80 с)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    fig.savefig(os.path.join(RESULTS_DIR, "linear_stand_comparison.png"), dpi=140)
    plt.close(fig)
    print(f"  -> {os.path.join(RESULTS_DIR, 'linear_stand_comparison.png')}")


if __name__ == "__main__":
    main()
