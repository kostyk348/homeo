"""Демо-кейс аудита: два связанных резервуара с квадратичными перетоками.

Объект (нелинейный):
    A1·h1' = u − k12·sign(h1−h2)·√|h1−h2|
    A2·h2' = k12·sign(h1−h2)·√|h1−h2| − k2·√h2 − d(t)

Допустимое множество Ω — полосы уровней обоих баков. Контроллер получает
номинальную модель со СМЕЩЁННЫМИ параметрами (k12: 0.85 vs 0.9, k2: 0.40 vs
0.45) и не знает профиль потребителя d(t) — только его границу (для фильтра).

Конфигурации: homeo + safety-фильтр (герой демо), homeo без фильтра,
homeo без safety-инвариантов, ПИД и LQR, спроектированные на линейном стенде
и развёрнутые без перенастройки.

Плоты (results/):
  * demo_trajectory.png      — траектории уровней против допустимой области,
                               возмущение, срабатывания фильтра;
  * demo_constraints.png     — запасы до границ полос (активные ограничения),
                               срабатывания фильтра;
  * demo_disturbance_control.png — возмущение и управления всех конфигураций.

Запуск:  python experiments/run_demo_tanks.py
Результат: results/demo.json + 3 плота
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import (demo_plant, demo_model, demo_scenario, make_homeo,
                                make_safety, episode_metrics, save_json,
                                scenario_block, RESULTS_DIR, DEMO_D)
from experiments.lqr import LQRTankController, pid_gains_from_json
from homeo.control import PIDTankController, run_episode, scenario_constraints
from homeo.metrics import violation_count, max_violation_depth, control_effort

DEMO_KW = dict(safety_margin=0.02, d_hat_beta=0.2)


def build_controllers(model, scen):
    safety = make_safety(model, scen, d_bound=DEMO_D, horizon=8, n_grid=41, margin=0.0)
    gains = pid_gains_from_json()
    return [
        ("homeo+фильтр", make_homeo(model, scen, **DEMO_KW), safety),
        ("homeo", make_homeo(model, scen, **DEMO_KW), None),
        ("homeo без safety", make_homeo(model, scen, use_safety_invariants=False), None),
        ("ПИД (со стенда)", PIDTankController(scen, *gains), None),
        ("LQR (со стенда)", LQRTankController(scen), None),
    ]


def main():
    scen = demo_scenario()
    model = demo_model()
    out = {"scenario": scenario_block(
        scen, label="демо: два связанных резервуара (квадратичные перетоки)",
        plant_params={"A1": 1.0, "A2": 1.0, "k12": 0.9, "k2": 0.45, "nonlinear": True,
                      "h_start": [0.6, 0.6],
                      "disturbance": "потребитель 0.55 м³/с, меандр период 60 с (скважность 50 %)"},
        model_params={"k12": model.c12, "k2": model.c2,
                      "note": "номинальная модель контроллера смещена относительно объекта"},
    )}

    runs = []
    print("== Демо: два связанных резервуара (нелинейные перетоки, смещённая модель) ==")
    print(f"  {'конфигурация':18s} viol  depth   overshoot settling recovery  effort  triggers")
    for name, ctrl, safety in build_controllers(model, scen):
        ep = run_episode(demo_plant(), ctrl, scen, noise_seed=7, safety=safety)
        m = episode_metrics(ep, scen, d_start=30.0)
        m["max_violation_depth"] = round(max_violation_depth(scenario_constraints(ep, scen)), 6)
        m["control_effort"] = round(control_effort(ep["u"], scen.dt), 3)
        m["violation_count"] = violation_count(scenario_constraints(ep, scen))
        infos = [i for i in ep["filter_infos"] if i is not None]
        m["filter_blocked_best_effort"] = sum(1 for i in infos if not i["feasible_found"])
        m["max_candidate_violation"] = round(max((i["cand_violation"] for i in infos), default=0.0), 6)
        out.setdefault("controllers", {})[name] = m
        runs.append((name, ep))
        st = "—" if m["settling_time"] is None else f"{m['settling_time']:7.2f}"
        rec = "—" if m["recovery_time"] is None else f"{m['recovery_time']:7.2f}"
        print(f"  {name:18s} {m['violation_count']:5d} {m['max_violation_depth']:6.3f} "
              f"{m['overshoot']:9.3f}  {st}  {rec} {m['control_effort']:7.1f} {m['filter_triggers']:8d}")

    save_json("demo.json", out)
    plot_trajectory(runs, scen)
    plot_constraints(runs, scen)
    plot_disturbance_control(runs, scen)


def _bands_axes(fig, axes, scen):
    lo1, hi1 = scen.h1_band
    lo2, hi2 = scen.h2_band
    for ax, (lo, hi) in zip(axes, [(lo1, hi1), (lo2, hi2)]):
        ax.axhspan(lo, hi, color="#d9f0d9", zorder=0)


def _disturbance_spans(axes, scen, color="#f6d3d3"):
    for ax in axes:
        t0 = 0.0
        prev = 0.0
        for k, t in enumerate([k * scen.dt for k in range(scen.steps)]):
            cur = scen.d_profile(t)
            if cur > 0 and prev <= 0:
                t0 = t
            if cur <= 0 and prev > 0:
                ax.axvspan(t0, t, color=color, alpha=0.6, zorder=0)
            prev = cur
        if prev > 0:
            ax.axvspan(t0, scen.steps * scen.dt, color=color, alpha=0.6, zorder=0)


def plot_trajectory(runs, scen):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True, constrained_layout=True)
    _bands_axes(fig, axes, scen)
    _disturbance_spans(axes, scen)
    show = {"homeo+фильтр": ("#1b6e3c", 1.9), "homeo": ("#4a90d9", 1.1),
            "ПИД (со стенда)": ("#c46a1f", 1.1), "LQR (со стенда)": ("#7d3ac1", 1.1),
            "homeo без safety": ("#b32424", 1.1)}
    for name, ep in runs:
        color, lw = show.get(name, ("#555", 1.0))
        axes[0].plot(ep["t"], ep["h1"], color=color, lw=lw, label=name)
        axes[1].plot(ep["t"], ep["h2"], color=color, lw=lw, label=name)
    hero = dict(runs)["homeo+фильтр"]
    trig_t = [hero["t"][k] for k, tr in enumerate(hero["triggers"]) if tr]
    for ax in axes:
        for t in trig_t:
            ax.axvline(t, color="#1b6e3c", lw=0.5, alpha=0.35, zorder=1)
    axes[1].set_xlabel("время, с")
    axes[0].set_ylabel("уровень бака 1, м")
    axes[1].set_ylabel("уровень бака 2, м")
    axes[0].set_title("Демо: два связанных резервуара — траектории против допустимой области Ω\n"
                      "(зелёное — Ω, красные полосы — работа потребителя, вертикали — срабатывания safety-фильтра)")
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    fig.savefig(os.path.join(RESULTS_DIR, "demo_trajectory.png"), dpi=140)
    plt.close(fig)
    print(f"  -> {os.path.join(RESULTS_DIR, 'demo_trajectory.png')}")


def plot_constraints(runs, scen):
    """Запасы до границ полос для homeo+фильтр: margin_j = −g_j (≤ 0 — нарушено)."""
    hero = dict(runs)["homeo+фильтр"]
    lo1, hi1 = scen.h1_band
    lo2, hi2 = scen.h2_band
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    for label, series, color in [
            ("запас бака 1 (верх)", [hi1 - v for v in hero["h1"]], "#2a7fb8"),
            ("запас бака 1 (низ)", [v - lo1 for v in hero["h1"]], "#7fb8d4"),
            ("запас бака 2 (верх)", [hi2 - v for v in hero["h2"]], "#1b6e3c"),
            ("запас бака 2 (низ)", [v - lo2 for v in hero["h2"]], "#8fce9f")]:
        ax.plot(hero["t"], series, lw=1.3, color=color, label=label)
    ax.axhline(0.0, color="#b32424", lw=1.4, label="граница допустимости (0)")
    for k, tr in enumerate(hero["triggers"]):
        if tr:
            ax.axvline(hero["t"][k], color="#555", lw=0.5, alpha=0.5)
    ax.set_xlabel("время, с")
    ax.set_ylabel("запас до границы, м")
    ax.set_title("Активные ограничения и срабатывания safety-фильтра (homeo+фильтр):\n"
                 "кривая касается нуля — ограничение активно, уходит ниже — нарушение")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(os.path.join(RESULTS_DIR, "demo_constraints.png"), dpi=140)
    plt.close(fig)
    print(f"  -> {os.path.join(RESULTS_DIR, 'demo_constraints.png')}")


def plot_disturbance_control(runs, scen):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True, constrained_layout=True)
    t = runs[0][1]["t"]
    d = runs[0][1]["d"]
    axes[0].fill_between(t, d, step="mid", color="#c46a1f", alpha=0.55)
    axes[0].set_ylabel("расход потребителя d, м³/с")
    axes[0].set_title("Возмущение (потребитель) и управления конфигураций")
    show = {"homeo+фильтр": ("#1b6e3c", 1.9), "homeo": ("#4a90d9", 1.1),
            "ПИД (со стенда)": ("#c46a1f", 1.1), "LQR (со стенда)": ("#7d3ac1", 1.1),
            "homeo без safety": ("#b32424", 1.1)}
    for name, ep in runs:
        color, lw = show.get(name, ("#555", 0.8))
        axes[1].plot(ep["t"], ep["u"], color=color, lw=lw, label=name)
    axes[1].set_xlabel("время, с")
    axes[1].set_ylabel("управление u (насос), м³/с")
    axes[1].set_ylim(-0.05, scen.umax * 1.08)
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    fig.savefig(os.path.join(RESULTS_DIR, "demo_disturbance_control.png"), dpi=140)
    plt.close(fig)
    print(f"  -> {os.path.join(RESULTS_DIR, 'demo_disturbance_control.png')}")


if __name__ == "__main__":
    main()
