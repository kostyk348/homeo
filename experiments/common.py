"""Общий код экспериментов валидации: сценарии, конфигурации контроллеров, метрики.

Все эксперименты детерминированы: фиксированные seed'ы (потоки возмущений и шума
генерируются заранее в run_episode), CPU-only, никаких внешних данных.
"""

import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from homeo.control import (TwoTankPlant, TwoTankModel, Scenario,          # noqa: E402
                           HomeoTankController,
                           run_episode, scenario_constraints, make_safety_filter)
from homeo.metrics import (violation_count, max_violation_depth,          # noqa: E402
                           overshoot, settling_time, recovery_time,
                           control_effort, zero_violation_margin)

RESULTS_DIR = os.path.join(ROOT, "results")

# --------------------------------------------------------------------------
# Сценарии
# --------------------------------------------------------------------------
DT = 0.25

# линейный стенд: 120 с, ступенька расхода потребителя на [40, 80) с
LIN_STEPS = 480
LIN_BANDS = ((0.2, 1.8), (0.3, 1.4))
LIN_REF = 1.0
LIN_SIGMA = 0.005
LIN_UMAX = 1.5


def lin_plant():
    return TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=LIN_UMAX,
                        h1=0.5, h2=0.5, nonlinear=False, dt=DT)


def lin_model():
    return TwoTankModel(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=LIN_UMAX,
                        nonlinear=False, dt=DT)


def lin_scenario(d_amp=0.35, sigma=LIN_SIGMA, d_start=40.0, d_end=80.0):
    return Scenario(steps=LIN_STEPS, dt=DT, h1_band=LIN_BANDS[0], h2_band=LIN_BANDS[1],
                    ref=LIN_REF, sigma=sigma, umax=LIN_UMAX,
                    d_profile=lambda t: d_amp if d_start <= t < d_end else 0.0)


# демо: нелинейные квадратичные перетоки, периодический потребитель, смещённая
# номинальная модель у контроллера (честная неопределённость)
DEMO_STEPS = 720  # 180 с
DEMO_BANDS = ((0.2, 1.5), (0.25, 1.2))
DEMO_REF = 0.9
DEMO_UMAX = 1.2
DEMO_D = 0.55
DEMO_SIGMA = 0.02
DEMO_MODEL_C12 = 0.85   # объект: 0.9
DEMO_MODEL_C2 = 0.40    # объект: 0.45


def demo_plant():
    return TwoTankPlant(A1=1.0, A2=1.0, c12=0.9, c2=0.45, umax=DEMO_UMAX,
                        h1=0.6, h2=0.6, nonlinear=True, dt=DT)


def demo_model():
    return TwoTankModel(A1=1.0, A2=1.0, c12=DEMO_MODEL_C12, c2=DEMO_MODEL_C2,
                        umax=DEMO_UMAX, nonlinear=True, dt=DT)


def demo_scenario():
    return Scenario(steps=DEMO_STEPS, dt=DT, h1_band=DEMO_BANDS[0], h2_band=DEMO_BANDS[1],
                    ref=DEMO_REF, sigma=DEMO_SIGMA, umax=DEMO_UMAX,
                    d_profile=lambda t: DEMO_D if (t % 60.0) >= 30.0 else 0.0)


# --------------------------------------------------------------------------
# Контроллеры
# --------------------------------------------------------------------------
HOMEO_KW = dict(horizon=16, lr=0.15, t0=0.02, anneal_k=1.0, seed=11,
                grad_tol=1e-4, max_iter=80, w_goal=1.0, w_safe=25.0, w_eff=0.05,
                safety_margin=0.03, d_hat_beta=0.2)


def make_homeo(model, scen, **over):
    kw = dict(HOMEO_KW)
    kw.update(over)
    return HomeoTankController(model, scen, **kw)


def make_safety(model, scen, d_bound, margin=0.0, horizon=8, n_grid=41):
    return make_safety_filter(model, scen, d_bound=d_bound, horizon=horizon,
                              n_grid=n_grid, margin=margin)


# --------------------------------------------------------------------------
# Метрики эпизода
# --------------------------------------------------------------------------
def episode_metrics(ep, scen, d_start=40.0, d_end=80.0, settle_window=None):
    """Свод метрик по эпизоду (определения — homeo/metrics.py и docs/validation.md).

    settling_time измеряется на окне ДО возмущения [0, d_start): во время
    возмущения контроллер по дизайну жертвует слежением ради допустимости,
    и это не «незатухший переходный процесс».
    """
    cons = scenario_constraints(ep, scen)
    band2 = scen.h2_band[1] - scen.h2_band[0]
    settle_tol = 0.02 * band2
    k_to = int((settle_window if settle_window is not None else d_start) / scen.dt)
    k_onset = int(d_start / scen.dt)
    return {
        "violation_count": violation_count(cons),
        "max_violation_depth": round(max_violation_depth(cons), 6),
        "overshoot": round(overshoot(ep["h2"], scen.ref), 6),
        "settling_time": _round_none(settling_time(ep["h2"], scen.ref, settle_tol,
                                                   scen.dt, k_to=k_to)),
        "recovery_time": _round_none(recovery_time(cons, k_onset, scen.dt)),
        "control_effort": round(control_effort(ep["u"], scen.dt), 3),
        "filter_triggers": int(sum(ep["triggers"])),
    }


def _round_none(v, nd=3):
    return None if v is None else round(v, nd)


def seed_averaged(metrics_list):
    """Среднее по списку метрик (None остаётся None, если встречается у всех)."""
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        if any(v is None for v in vals):
            out[k] = None if all(v is None for v in vals) else vals
        elif isinstance(vals[0], float):
            out[k] = round(sum(vals) / len(vals), 6)
        else:
            out[k] = round(sum(vals) / len(vals), 3) if k != "filter_triggers" else \
                round(sum(vals) / len(vals), 1)
    return out


def run_episode_on(plant_factory, model, scen, controller_factory, *, noise_seed=7,
                   safety=None):
    """Прогон: контроллер строится фабрикой, объект — фабрикой plant_factory."""
    return run_episode(plant_factory(), controller_factory(model, scen), scen,
                       noise_seed=noise_seed, safety=safety)


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------
def save_json(name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> {path}")
    return path


def environment():
    import homeo
    import numpy
    import scipy
    import matplotlib
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_only": True,
        "homeo_version": homeo.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "seeds": {"controller": HOMEO_KW["seed"], "noise": 7,
                  "pid_grid": "tuning on noise_seed=7"},
        "note": "все прогоны детерминированы фиксированными seed; ускорение GPU не используется",
    }


def scenario_block(scen, *, plant_params, model_params=None, label):
    return {
        "label": label,
        "dt": scen.dt,
        "steps": scen.steps,
        "T": scen.steps * scen.dt,
        "h1_band": list(scen.h1_band),
        "h2_band": list(scen.h2_band),
        "ref": scen.ref,
        "sigma_sensor": scen.sigma,
        "umax": scen.umax,
        "plant_params": plant_params,
        "controller_model_params": model_params,
    }
