"""Пример на реальной системе: демпфированный осциллятор под гармонической силой.

    m·x'' + c·x' + k·x = F0·cos(ω·t)

Установившееся решение:  x(t) = A·cos(ω·t − φ),  где

    A   = F0 / sqrt((k − m·ω²)² + (c·ω)²)      [м]
    φ   = atan2(c·ω, k − m·ω²)                [рад]

A и φ — измеряемые величины (амплитуда по датчику перемещения и сдвиг фаз между
силой и перемещением), то есть это система с физическим эталоном, а не демо
«функция вернула число».

Как это делает homeo. Ядро не получает ни формулу для A, ни формулу для φ.
Оно получает ДВА УРАВНЕНИЯ НЕВЯЗКИ в ньютонах — подстановку анзаца
x = A·cos(ωt − φ) в уравнение движения с приравниванием коэффициентов при cos и sin:

    r1(A, φ) = (k − m·ω²)·A − F0·cos φ      [Н]      ← коэффициент при cos
    r2(A, φ) = −c·ω·A + F0·sin φ            [Н]      ← коэффициент при sin

Пространство политики X = (A, φ), нормированное к [−1, 1]. Ядро минимизирует
E = Σ |rᵢ|²/F0² (веса O(1), чтобы абсолютный grad_tol был осмысленным) и само
находит точку, где обе невязки нулевые. Никакой подгонки: если физика решения
неверна, невязки не занулятся.

Эталон двойной, чтобы проверка не опиралась на одну и ту же выкладку:
  1) замкнутая формула выше;
  2) независимое численное интегрирование того же ОДУ методом Рунге—Кутты 4
     (rk4_reference): интегрируем от покоя, ждём затухания переходного процесса
     и измеряем установившиеся A и φ проекцией на cos/sin.

Запуск:  python examples/driven_oscillator.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homeo import HomeoEngine, make_invariant  # noqa: E402


# --------------------------------------------------------------------------
# эталон 1: замкнутая формула
# --------------------------------------------------------------------------
def closed_form(m, c, k, F0, w):
    """A [м], φ [рад] установившегося отклика на F0·cos(ωt)."""
    d = k - m * w * w
    A = F0 / math.sqrt(d * d + (c * w) ** 2)
    phi = math.atan2(c * w, d)
    return A, phi


# --------------------------------------------------------------------------
# эталон 2: независимое интегрирование ОДУ (RK4, только stdlib)
# --------------------------------------------------------------------------
def rk4_reference(m, c, k, F0, w, steps_per_period=600, min_periods=8,
                  decay_periods=12.0, meas_periods=4):
    """Установившиеся A и φ прямым интегрированием m x'' + c x' + k x = F0 cos(ωt).

    Длина прогона выбирается из времени затухания τ = 2m/c: интегрируем
    decay_periods·τ, затем измеряем амплитуду и фазу проекцией x(t) на cos/sin
    за последние measure_periods периодов (в установившемся режиме x(t) = A·cos(ωt − φ)
    даёт коэффициенты A·cos φ и A·sin φ).
    """
    T = 2.0 * math.pi / w
    tau = 2.0 * m / c
    periods = max(min_periods, int(math.ceil(decay_periods * tau / T)) + meas_periods)
    dt = T / steps_per_period
    nsteps = periods * steps_per_period
    keep_from = nsteps - meas_periods * steps_per_period

    def deriv(t, x, v):
        return v, (F0 * math.cos(w * t) - c * v - k * x) / m

    x = v = 0.0
    t = 0.0
    sc = ss = taken = 0.0
    for i in range(nsteps):
        k1 = deriv(t, x, v)
        k2 = deriv(t + dt / 2, x + dt / 2 * k1[0], v + dt / 2 * k1[1])
        k3 = deriv(t + dt / 2, x + dt / 2 * k2[0], v + dt / 2 * k2[1])
        k4 = deriv(t + dt, x + dt * k3[0], v + dt * k3[1])
        x += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += dt
        if i >= keep_from:
            sc += x * math.cos(w * t)
            ss += x * math.sin(w * t)
            taken += dt
    a = 2.0 * sc * dt / taken
    b = 2.0 * ss * dt / taken
    return math.hypot(a, b), math.atan2(b, a)


# --------------------------------------------------------------------------
# homeo: решаем систему по невязкам
# --------------------------------------------------------------------------
def make_engine(m, c, k, F0, w, A_box, lr=0.1, max_iter=400, grad_tol=1e-4,
                t0=0.0, seed=3):
    """Движок с X = (A, φ): A ∈ (0, 2·A_box), φ ∈ (−π, π). Веса невязок = 1/F0²."""
    wgt = 1.0 / (F0 * F0)

    def _A(X):
        return X[0] * A_box + A_box

    def r1(X):  # невязка при cos, [Н]
        return (k - m * w * w) * _A(X) - F0 * math.cos(X[1] * math.pi)

    def r2(X):  # невязка при sin, [Н]
        return -c * w * _A(X) + F0 * math.sin(X[1] * math.pi)

    return HomeoEngine(center=[A_box, 0.0], range_=[A_box, math.pi],
                       invariants=[make_invariant(wgt, 2, r1), make_invariant(wgt, 2, r2)],
                       lr=lr, max_iter=max_iter, grad_tol=grad_tol, t0=t0, seed=seed)


def solve(m, c, k, F0, w, A_box=None, start_frac=0.4, **kw):
    """Возвращает dict: найденные A, φ, число шагов, невязки [Н], энергию.

    A_box — половина ширины ящика по амплитуде (A ∈ (0, 2·A_box)); по умолчанию
    берётся 1.05·A* — так ящик содержит эталонное решение.
    """
    A_ref, phi_ref = closed_form(m, c, k, F0, w)
    if A_box is None:
        A_box = 1.05 * A_ref
    eng = make_engine(m, c, k, F0, w, A_box, **kw)
    eng.set_raw([A_box * start_frac, 0.0])
    iters = eng.tick()
    Xn = eng.norm(eng.raw())
    A = Xn[0] * A_box + A_box
    phi = Xn[1] * math.pi
    r1 = (k - m * w * w) * A - F0 * math.cos(phi)
    r2 = -c * w * A + F0 * math.sin(phi)
    return {
        "A": A, "phi": phi, "iters": iters, "A_ref": A_ref, "phi_ref": phi_ref,
        "A_box": A_box, "max_iter": eng.max_iter,
        "converged": iters < eng.max_iter, "energy": eng.energy(),
        "r1": r1, "r2": r2,
        "rel_err_A": abs(A - A_ref) / A_ref, "abs_err_phi": abs(phi - phi_ref),
    }


CASES = [
    # m,    c,    k,    F0,   w
    (1.0, 2.0, 20.0, 5.0, 3.0),
    (2.0, 2.4, 50.0, 10.0, 4.0),
    (0.5, 1.5, 8.0, 2.0, 5.0),
    (1.2, 0.9, 100.0, 1.0, 9.0),
]


def main():
    print("Демпфированный осциллятор под гармонической силой: m·x'' + c·x' + k·x = F0·cos(ω·t)")
    print("Эталон: A = F0/sqrt((k−mω²)² + (cω)²) [м], φ = atan2(cω, k−mω²) [рад]")
    print("Плюс независимая проверка эталона интегрированием ОДУ (RK4).")
    print()

    print("== эталон сверен двумя независимыми способами ==")
    for m, c, k, F0, w in CASES:
        Ac, pc = closed_form(m, c, k, F0, w)
        Ar, pr = rk4_reference(m, c, k, F0, w)
        print(f"  m={m:<4} c={c:<4} k={k:<6} F0={F0:<5} w={w:<4}"
              f" | формула A={Ac:.6f} φ={pc:+.6f}"
              f" | RK4 A={Ar:.6f} φ={pr:+.6f}"
              f" | ΔA/A={abs(Ar - Ac) / Ac:.1e} Δφ={abs(pr - pc):.1e}")

    print()
    print("== homeo решает систему по невязкам (никаких формул для A и φ внутри) ==")
    worst_a = worst_p = 0.0
    worst_res = 0.0
    n_conv = 0
    for i, (m, c, k, F0, w) in enumerate(CASES, 1):
        r = solve(m, c, k, F0, w)
        n_conv += int(r["converged"])
        worst_a = max(worst_a, r["rel_err_A"])
        worst_p = max(worst_p, r["abs_err_phi"])
        worst_res = max(worst_res, abs(r["r1"]), abs(r["r2"]))
        print(f"  случай {i}: m={m} c={c} k={k} F0={F0} w={w} (A* = {r['A_ref']:.6f} м, "
              f"φ* = {r['phi_ref']:+.6f} рад)")
        print(f"    шагов: {r['iters']}/{r['max_iter']} "
              f"({'сошлось' if r['converged'] else 'НЕ СОШЛОСЬ (упёрлось в границу ящика)'}), "
              f"E = {r['energy']:.2e}")
        print(f"    найдено: A = {r['A']:.6f} м, φ = {r['phi']:+.6f} рад")
        print(f"    ошибка против эталона: ΔA/A = {r['rel_err_A']:.2e}, "
              f"|Δφ| = {r['abs_err_phi']:.2e} рад")
        print(f"    невязки уравнений движения: r1 = {r['r1']:+.2e} Н, r2 = {r['r2']:+.2e} Н")

    print()
    print(f"  ИТОГ: сошлось {n_conv}/{len(CASES)} систем; макс. относительная ошибка "
          f"амплитуды = {worst_a:.2e}; макс. ошибка фазы = {worst_p:.2e} рад; "
          f"макс. невязка = {worst_res:.2e} Н")

    print()
    print("== честное поведение вне области применимости: ящик уже, чем решение ==")
    m, c, k, F0, w = CASES[0]
    r = solve(m, c, k, F0, w, A_box=0.3 * closed_form(m, c, k, F0, w)[0], max_iter=300)
    print(f"  A* = {r['A_ref']:.6f} м, а ящик допускает A ≤ {2 * r['A_box']:.6f} м")
    print(f"    шагов: {r['iters']}/{r['max_iter']} → "
          f"tick() вернул max_iter: сходимости НЕТ (это и есть признак)")
    print(f"    найдено A = {r['A']:.6f} м (упёрлось в границу ящика), "
          f"относительная ошибка {r['rel_err_A']:.3f}")
    print(f"    невязки не занулились: r1 = {r['r1']:+.3f} Н, r2 = {r['r2']:+.3f} Н, "
          f"E = {r['energy']:.3f}")

    ok = (n_conv == len(CASES) and worst_a < 1e-3 and worst_p < 1e-3
          and not r["converged"])
    print()
    print("ПРОВЕРКА ПРИМЕРА:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
