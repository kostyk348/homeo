"""Тесты на РЕАЛЬНОЙ системе: демпфированный осциллятор под гармонической силой.

    m·x'' + c·x' + k·x = F0·cos(ω·t)   →   x(t) = A·cos(ω·t − φ)

Физический эталон двойной (обе половины считает один и тот же модуль
examples/driven_oscillator.py, поэтому тесты проверяют ещё и его):

  1) замкнутая формула  A = F0/sqrt((k−mω²)² + (cω)²),  φ = atan2(cω, k−mω²);
  2) независимое численное интегрирование того же ОДУ (RK4) от покоя до
     установившегося режима.

homeo решает систему по двум невязкам в ньютонах (r1, r2 из подстановки анзаца),
не зная ни одной из этих формул. Тесты требуют: сошлось за < max_iter,
относительная ошибка амплитуды ≤ 1e-3, ошибка фазы ≤ 1e-3 рад, невязки ≤ 1e-2 Н.
"""

import math
import random

import pytest

import driven_oscillator as osc


CASES = list(osc.CASES)


def _random_cases(n=8, seed=2026):
    """Физически осмысленные конфигурации: 0.15 ≤ ζ ≤ 0.6 (переходный процесс
    успевает затухнуть — иначе RK4-эталон сам не установится)."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        m = rnd.uniform(0.5, 3.0)
        k = rnd.uniform(5.0, 120.0)
        zeta = rnd.uniform(0.15, 0.6)
        c = zeta * 2.0 * math.sqrt(k * m)
        F0 = rnd.uniform(0.5, 20.0)
        w = rnd.uniform(0.3, 1.5) * math.sqrt(k / m)
        out.append((m, c, k, F0, w))
    return out


RANDOM_CASES = _random_cases()


# --------------------------------------------------------------------------
# эталон
# --------------------------------------------------------------------------
@pytest.mark.parametrize("params", CASES + RANDOM_CASES,
                         ids=[f"c{i}" for i in range(len(CASES) + len(RANDOM_CASES))])
def test_reference_is_self_consistent_closed_form_vs_rk4(params):
    """Прежде чем что-то проверять, убеждаемся, что эталон согласован:
    замкнутая формула и прямое интегрирование ОДУ дают одно и то же число."""
    m, c, k, F0, w = params
    A_f, phi_f = osc.closed_form(m, c, k, F0, w)
    A_n, phi_n = osc.rk4_reference(m, c, k, F0, w)
    assert abs(A_n - A_f) / A_f <= 1e-5, f"A: {A_n} vs {A_f}"
    assert abs(phi_n - phi_f) <= 1e-5, f"φ: {phi_n} vs {phi_f}"


# --------------------------------------------------------------------------
# homeo на реальной системе
# --------------------------------------------------------------------------
@pytest.mark.parametrize("idx", range(len(RANDOM_CASES)))
def test_homeo_finds_operating_point_of_oscillator(idx):
    m, c, k, F0, w = RANDOM_CASES[idx]
    box_factor = 1.05 if idx % 2 == 0 else 2.0
    A_ref, _ = osc.closed_form(m, c, k, F0, w)
    r = osc.solve(m, c, k, F0, w, A_box=box_factor * A_ref)
    assert r["converged"], f"не сошлось: iters={r['iters']}/{r['max_iter']}"
    assert r["rel_err_A"] <= 1e-3, f"A={r['A']} против эталона {r['A_ref']}"
    assert r["abs_err_phi"] <= 1e-3, f"φ={r['phi']} против эталона {r['phi_ref']}"


@pytest.mark.parametrize("params", RANDOM_CASES,
                         ids=[f"r{i}" for i in range(len(RANDOM_CASES))])
def test_homeo_residuals_are_small_in_newtons(params):
    """Физическая точность: невязки уравнений движения в ньютонах после сходимости."""
    m, c, k, F0, w = params
    r = osc.solve(m, c, k, F0, w)
    assert r["converged"]
    assert max(abs(r["r1"]), abs(r["r2"])) <= 1e-2, f"r1={r['r1']}, r2={r['r2']} Н"
    assert r["energy"] <= 1e-4, f"E={r['energy']:.3e}"


def test_fixed_cases_reported_by_the_example():
    """Четыре случая из examples/driven_oscillator.py — с теми же числами."""
    worst_a = worst_p = 0.0
    for m, c, k, F0, w in CASES:
        r = osc.solve(m, c, k, F0, w)
        assert r["converged"], f"не сошлось: {(m, c, k, F0, w)}"
        worst_a = max(worst_a, r["rel_err_A"])
        worst_p = max(worst_p, r["abs_err_phi"])
    assert worst_a <= 1e-3, f"макс. относительная ошибка амплитуды {worst_a:.3e}"
    assert worst_p <= 1e-3, f"макс. ошибка фазы {worst_p:.3e} рад"


def test_box_narrower_than_solution_reports_non_convergence():
    """Область применимости: ящик по амплитуде уже, чем решение → признак
    несходимости (iters == max_iter), точка на границе, большая остаточная ошибка."""
    m, c, k, F0, w = CASES[0]
    A_ref, _ = osc.closed_form(m, c, k, F0, w)
    r = osc.solve(m, c, k, F0, w, A_box=0.3 * A_ref, max_iter=300)
    assert not r["converged"], "движок заявил сходимость вне ящика"
    assert r["iters"] == r["max_iter"] == 300
    assert abs(r["A"] - 2.0 * r["A_box"]) <= 1e-9, "точка должна упереться в границу ящика"
    assert r["rel_err_A"] > 0.05, f"ошибка {r['rel_err_A']:.3f} должна быть заметной"
    assert r["energy"] > 1e-2, f"E={r['energy']:.3e}: невязка не занулилась"


def test_amplitude_and_phase_are_measurable_quantities():
    """Санитарная проверка физики: амплитуда положительна и не превышает
    статический прогиб × добротность; фаза лежит в (0, π) — сила опережает
    перемещение, как и должно быть у осциллятора с демпфированием."""
    for m, c, k, F0, w in CASES + RANDOM_CASES:
        A, phi = osc.closed_form(m, c, k, F0, w)
        assert 0.0 < A < 100.0 * F0 / k, f"A={A}"
        assert 0.0 < phi < math.pi, f"φ={phi}"
