"""Тесты валидационного слоя (аудит 2026-11): метрики, safety-фильтр, контур,
детерминизм, адаптивный шаг, термальный шум/annealing, абляционные флаги ядра.

Только stdlib: тесты не требуют numpy/scipy (CI ставит лишь ".[test]").
"""

import math
import random

import pytest

from homeo.engine import HomeoEngine, make_invariant, make_dynamic_invariant
from homeo.safety import max_violation, filter_control
from homeo.metrics import (violation_count, max_violation_depth, overshoot,
                           settling_time, recovery_time, control_effort,
                           violation_depth, zero_violation_margin)
from homeo.control import (TwoTankPlant, TwoTankModel, Scenario,
                           HomeoTankController, PIDTankController,
                           run_episode, scenario_constraints, make_safety_filter)


# --------------------------------------------------------------------------
# метрики на синтетических траекториях
# --------------------------------------------------------------------------
def test_violation_count_and_depth_on_synthetic():
    s = [0.5, 1.0, 1.25, 1.4, 1.5, 1.1, 0.9]          # полоса [0.3, 1.4]
    cons = [(s, (0.3, 1.4))]
    assert violation_count(cons) == 1                  # только 1.5 вне полосы (1.4 — граница)
    assert max_violation_depth(cons) == pytest.approx(0.1)
    assert violation_depth(1.25, 0.3, 1.4) == 0.0
    assert violation_depth(0.2, 0.3, 1.4) == pytest.approx(0.1)


def test_violation_count_any_of_several_constraints():
    s1 = [0.9, 0.9, 0.9]
    s2 = [0.1, 0.5, 2.0]
    cons = [(s1, (0.3, 1.4)), (s2, (0.3, 1.4))]
    assert violation_count(cons) == 2                  # шаг 0: s2=0.1; шаг 2: s2=2.0
    assert max_violation_depth(cons) == pytest.approx(0.6)


def test_overshoot_settling_effort():
    s = [0.0, 0.5, 1.05, 0.99, 1.01, 1.0, 1.0]
    assert overshoot(s, 1.0) == pytest.approx(0.05)
    st = settling_time(s, 1.0, tol=0.02, dt=0.5)
    assert st == pytest.approx(1.5)                    # индекс 3 (0.99) — далее в трубке до конца
    # так и не осел: серия ЗАКАНЧИВАЕТСЯ вне трубки → None (определение —
    # «первое t, начиная с которого |v−ref| ≤ tol до конца окна»)
    assert settling_time([1.0, 1.2, 1.2], 1.0, 0.02, 1.0) is None
    # граничный случай того же определения: последний отсчёт в трубке формально
    # считается оседанием на последнем шаге окна (k=2 — конец окна)
    assert settling_time([1.0, 1.2, 1.0], 1.0, 0.02, 1.0) == pytest.approx(2.0)
    assert control_effort([0.5, -0.5, 1.0], dt=0.25) == pytest.approx(0.5)


def test_recovery_time_definitions():
    s = [0.5, 0.9, 1.0, 1.6, 1.7, 1.3, 1.0, 1.0]       # полоса [0.3, 1.4]
    cons = [(s, (0.3, 1.4))]
    # нарушений после onset (индекс 5) нет
    assert recovery_time(cons, 5, dt=1.0) == 0.0
    # onset в момент нарушения (индекс 3): вернулось на индексе 5 → 2.0
    assert recovery_time(cons, 3, dt=1.0) == pytest.approx(2.0)
    # никогда не возвращается
    bad = [( [0.5, 1.6, 1.7, 1.8], (0.3, 1.4) )]
    assert recovery_time(bad, 1, dt=1.0) is None


def test_zero_violation_margin():
    m = zero_violation_margin([0.1, 0.2, 0.3, 0.4], [0, 0, 5, 7])
    assert m["margin"] == pytest.approx(0.2)
    assert m["first_violating"] == pytest.approx(0.3)
    m2 = zero_violation_margin([0.1, 0.2], [3, 1])
    assert m2["margin"] is None and m2["first_violating"] == pytest.approx(0.1)


# --------------------------------------------------------------------------
# safety-слой
# --------------------------------------------------------------------------
def test_max_violation_inequalities_and_equalities():
    g = [lambda x: 1.0 - x[0], lambda x: x[0] - 2.0]
    h = [lambda x: x[1]]
    assert max_violation([1.5, 0.0], g, h) == 0.0
    assert max_violation([2.5, 0.0], g, h) == pytest.approx(0.5)
    assert max_violation([1.5, 0.3], g, h) == pytest.approx(0.3)   # |h| > 0
    assert max_violation([1.5, 0.05], g, h, tol=0.1) == 0.0        # допуск


def test_filter_passes_feasible_candidate_unchanged():
    model = TwoTankModel(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, dt=0.25)
    scen = Scenario(steps=10, dt=0.25, h1_band=(0.2, 1.8), h2_band=(0.3, 1.4),
                    ref=1.0, d_profile=lambda t: 0.0, umax=1.5)
    g = scen.g_funcs()
    u, info = filter_control(model.step, [1.4, 1.0], 0.35, 0.0, 1.5,
                             g_funcs=g, horizon=4, n_grid=21)
    assert u == pytest.approx(0.35)
    assert info["triggered"] is False and info["feasible_found"]


def test_filter_projects_infeasible_candidate():
    """Кандидат u=1.5 из состояния у верхней границы бака 1 выведет h1 за полосу →
    фильтр обязан заменить кандидата и доложить о срабатывании."""
    model = TwoTankModel(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, dt=0.25)
    scen = Scenario(steps=10, dt=0.25, h1_band=(0.2, 1.8), h2_band=(0.3, 1.4),
                    ref=1.0, d_profile=lambda t: 0.0, umax=1.5)
    g = scen.g_funcs()
    u, info = filter_control(model.step, [1.79, 1.0], 1.5, 0.0, 1.5,
                             g_funcs=g, horizon=4, n_grid=41)
    assert u < 1.5
    assert info["triggered"] is True and info["feasible_found"]
    assert info["out_violation"] == pytest.approx(0.0, abs=1e-9)


def test_filter_best_effort_when_nothing_feasible():
    """Полоса бака 2 физически недостижима для любого u → фильтр возвращает
    управление с минимальным нарушением и feasible_found=False."""
    model = TwoTankModel(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, dt=0.25)
    scen = Scenario(steps=10, dt=0.25, h1_band=(0.2, 1.8), h2_band=(1.9, 2.0),
                    ref=1.0, d_profile=lambda t: 0.0, umax=1.5)
    g = scen.g_funcs()
    u, info = filter_control(model.step, [1.0, 0.5], 0.35, 0.0, 1.5,
                             g_funcs=g, horizon=2, n_grid=21)
    assert info["feasible_found"] is False
    assert info["out_violation"] > 0.0


# --------------------------------------------------------------------------
# объект: санитарные проверки физики
# --------------------------------------------------------------------------
def test_plant_mass_balance_directions():
    p = TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, h1=1.2, h2=1.0, dt=0.25)
    p.step(0.0, 0.0)                                   # без притока переток тянет оба вниз
    assert p.h1 < 1.2 and p.h2 < 1.0
    p2 = TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, h1=1.2, h2=1.0, dt=0.25)
    p2.step(1.5, 0.0)                                  # полный насос — уровень бака 1 растёт
    assert p2.h1 > 1.2
    p3 = TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, h1=1.2, h2=1.0, dt=0.25)
    p3.step(0.35, 0.2)                                 # потребитель из бака 2
    assert p3.h2 < 1.0
    p4 = TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.0, h1=1.2, h2=1.0, dt=0.25)
    p4.step(5.0, 0.0)                                  # насыщение привода: u → umax = 1.0
    assert p4.h1 == pytest.approx(1.2 + 0.25 * (1.0 - 0.7 * 0.2))


def test_sqrt_model_flows_are_zero_at_equal_levels():
    m = TwoTankModel(c12=0.9, c2=0.45, nonlinear=True, dt=0.25)
    q12, qout = m.flows(1.0, 1.0)
    assert q12 == 0.0 and qout == pytest.approx(0.45)


# --------------------------------------------------------------------------
# замкнутый контур: детерминизм и безопасность
# --------------------------------------------------------------------------
def _stand_scen(d_amp=0.35):
    return Scenario(steps=480, dt=0.25, h1_band=(0.2, 1.8), h2_band=(0.3, 1.4),
                    ref=1.0, sigma=0.005, umax=1.5,
                    d_profile=lambda t: d_amp if 40.0 <= t < 80.0 else 0.0)


def _stand_model():
    return TwoTankModel(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5, nonlinear=False, dt=0.25)


def _stand_plant():
    return TwoTankPlant(A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5,
                        h1=0.5, h2=0.5, nonlinear=False, dt=0.25)


def test_closed_loop_determinism_fixed_seed():
    scen = _stand_scen()
    model = _stand_model()
    ep1 = run_episode(_stand_plant(), HomeoTankController(model, scen, seed=11), scen, noise_seed=7)
    ep2 = run_episode(_stand_plant(), HomeoTankController(model, scen, seed=11), scen, noise_seed=7)
    for key in ("t", "h1", "h2", "u", "d", "y1", "y2"):
        assert ep1[key] == ep2[key], f"поток {key} не воспроизводится"


def test_closed_loop_noise_seed_changes_trajectory():
    scen = _stand_scen()
    model = _stand_model()
    ep1 = run_episode(_stand_plant(), HomeoTankController(model, scen, seed=11), scen, noise_seed=7)
    ep2 = run_episode(_stand_plant(), HomeoTankController(model, scen, seed=11), scen, noise_seed=8)
    assert ep1["h2"] != ep2["h2"]


def test_homeo_zero_violations_pid_violates_on_linear_stand():
    """Ключевой контраст стенда: homeo с safety-инвариантами удерживает Ω,
    ПИД (слепой к баку 1) выталкивает h1 из полосы при ступени потребителя."""
    scen = _stand_scen()
    model = _stand_model()
    ep = run_episode(_stand_plant(), HomeoTankController(model, scen, seed=11), scen, noise_seed=7)
    m = violation_count(scenario_constraints(ep, scen))
    assert m == 0, f"homeo нарушил Ω {m} раз"
    assert abs(ep["h2"][-1] - scen.ref) <= 0.03         # слежение восстановлено
    pid = PIDTankController(scen, kp=6.0, ki=0.8, kd=0.5)
    ep_pid = run_episode(_stand_plant(), pid, scen, noise_seed=7)
    assert violation_count(scenario_constraints(ep_pid, scen)) > 50


def test_safety_filter_rescues_controller_without_safety_invariants():
    """Архитектурный принцип аудита: кандидат от оптимизатора проходит фильтр.
    Даже контроллер БЕЗ барьерных инвариантов с фильтром даёт 0 нарушений."""
    scen = _stand_scen(d_amp=0.5)
    model = _stand_model()
    safety = make_safety_filter(model, scen, d_bound=0.5, horizon=8, n_grid=41)
    ctrl = HomeoTankController(model, scen, seed=11, use_safety_invariants=False)
    ep = run_episode(_stand_plant(), ctrl, scen, noise_seed=7, safety=safety)
    assert violation_count(scenario_constraints(ep, scen)) == 0
    assert sum(ep["triggers"]) > 50                     # фильтр реально работал


# --------------------------------------------------------------------------
# ядро: адаптивный шаг, шум/annealing, абляционные флаги
# --------------------------------------------------------------------------
def test_adaptive_h_finds_gradient_across_flat_region():
    """E = 0 при x > 0.2 и (x−0.2)² при x ≤ 0.2: со старта x=0.25 (в плоской зоне)
    обычный h=1e-4 даёт нулевой градиент; удвоения обязаны дотянуться до излома
    и вытолкнуть точку правее (в сторону нулевой энергии). Без удвоений — ноль
    градиента, точка стоит на месте."""
    def err(X):
        v = X[0]
        return 0.0 if v > 0.2 else v - 0.2
    for h_dbl, moved in ((12, True), (0, False)):
        eng = HomeoEngine(center=[0.0], range_=[1.0],
                          invariants=[make_invariant(1.0, 2, err)],
                          lr=0.2, max_iter=300, grad_tol=1e-6, t0=0.0, seed=1,
                          h_doublings=h_dbl)
        eng.set_raw([0.25])
        eng.tick()
        x = eng.norm(eng.raw())[0]
        if moved:
            assert x > 0.26, f"адаптивный h не нашёл излом: x={x}"
        else:
            assert x == pytest.approx(0.25), "без удвоений градиент не должен найтись"


def test_thermal_noise_anneals_to_zero():
    """При t0>0 и anneal_k=1 шум затухает: 500-й шаг почти не шевелит сошедшуюся
    точку, а при anneal_k=0 шум постоянной амплитуды — точка гуляет."""
    invs = [make_invariant(1.0, 2, lambda X: X[0] - 0.1)]
    eng = HomeoEngine(center=[0.0], range_=[1.0], invariants=invs,
                      lr=0.1, max_iter=1000, grad_tol=0.0, t0=0.02, anneal_k=1.0, seed=5)
    eng.set_raw([0.0])
    xs_anneal = []
    for _ in range(40):
        eng._step()
        xs_anneal.append(eng._Xnorm[0])
    spread_anneal = max(xs_anneal[-10:]) - min(xs_anneal[-10:])
    eng2 = HomeoEngine(center=[0.0], range_=[1.0], invariants=invs,
                       lr=0.1, max_iter=1000, grad_tol=0.0, t0=0.02, anneal_k=0.0, seed=5)
    eng2.set_raw([0.0])
    xs_const = []
    for _ in range(40):
        eng2._step()
        xs_const.append(eng2._Xnorm[0])
    spread_const = max(xs_const[-10:]) - min(xs_const[-10:])
    assert spread_const > spread_anneal * 5.0, (
        f"annealing не гасит шум: {spread_const} vs {spread_anneal}")


def test_t0_zero_is_seed_independent():
    outs = []
    for seed in (1, 2, 3):
        eng = HomeoEngine(center=[0.0], range_=[1.0],
                          invariants=[make_invariant(1.0, 2, lambda X: X[0] - 0.3)],
                          lr=0.1, max_iter=200, grad_tol=1e-4, t0=0.0, seed=seed)
        eng.set_raw([0.9])
        eng.tick()
        outs.append(eng.raw()[0])
    assert outs[0] == outs[1] == outs[2]


def test_project_false_allows_leaving_box():
    """Абляция проекции: с проекцией итерация НИКОГДА не покидает [−1,1]; без проекции
    оптимизатор волен выйти за ящик (у E=(x+5)² минимум при x=−5, далеко за границей).
    Замечание: пробные точки _grad обрезаются границей ящика (задокументированное
    поведение ядра), поэтому глубоко за границей оценка градиента разворачивается
    назад и точка может вернуться — абляционное СВОЙСТВО не «конечная точка вне
    ящика», а «точке разрешено быть вне ящика» (у E с минимумом внутри ящика
    обрезка не видна вовсе)."""
    invs = [make_invariant(1.0, 2, lambda X: X[0] + 5.0)]   # минимум на нижней границе
    common = dict(center=[0.0], range_=[1.0], invariants=invs,
                  lr=0.5, max_iter=5, grad_tol=0.0, t0=0.0, seed=1)
    eng_on = HomeoEngine(**common, project=True)
    eng_on.set_raw([0.0])
    traj_on = [eng_on._Xnorm[0]]
    for _ in range(common["max_iter"]):
        eng_on._step()
        traj_on.append(eng_on._Xnorm[0])
    assert all(abs(x) <= 1.0 for x in traj_on), traj_on
    eng_off = HomeoEngine(**common, project=False)
    eng_off.set_raw([0.0])
    traj_off = [eng_off._Xnorm[0]]
    for _ in range(common["max_iter"]):
        eng_off._step()
        traj_off.append(eng_off._Xnorm[0])
    assert any(abs(x) > 1.0 for x in traj_off), (
        f"без проекции итерат обязан выходить из ящика: {traj_off}")


def test_fixed_gain_engine_matches_plain_gradient_descent():
    """FixedGainHomeoEngine = градиентный спуск с фиксированным шагом: на квадратной
    чаше E=(x−0.5)² траектория совпадает с ручным GD (шум выключен, dx/dX = 1)."""
    from homeo.control import FixedGainHomeoEngine
    eng = FixedGainHomeoEngine(center=[0.0], range_=[1.0],
                               invariants=[make_invariant(1.0, 2, lambda X: X[0] - 0.5)],
                               lr=0.1, max_iter=50, grad_tol=0.0, t0=0.0, seed=1)
    eng.set_raw([0.0])
    hist = []
    for _ in range(10):
        eng._step()
        hist.append(eng._Xnorm[0])
    x = 0.0
    for got in hist:
        x = x - 0.1 * (2.0 * (x - 0.5))                  # dE/dX = 2(x−0.5)
        assert got == pytest.approx(x, abs=1e-9), f"движок {got} vs ручной GD {x}"
