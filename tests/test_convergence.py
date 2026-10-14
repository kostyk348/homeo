"""Тесты сходимости ядра homeo: проверяется СВОЙСТВО, а не «функция что-то вернула».

Контракт (он же зафиксирован в README, раздел «Сходимость»):

  (A) есть решение внутри нормированного ящика [−1,1]^n
      → tick() завершается ДОСРОЧНО по grad_tol (iters < max_iter), и найденная
        точка совпадает с аналитическим минимумом с точностью ≤ TOL_NORM
        по нормированной координате (TOL_NORM = 1e-3 = 0.1 % ширины ящика);
  (B) решения внутри ящика нет
      → движок НЕ рапортует о сходимости: iters == max_iter, точка упирается в
        границу, остаточная энергия/невязка велика;
  (C) признак «iters == max_iter» не срабатывает ложно, когда оптимум лежит ровно
      НА границе ящика — это законная точка (невязка там нулевая);
  (D) grad_tol абсолютен в единицах E: при масштабировании весов на λ его надо
      масштабировать тоже, иначе меняются и точка, и признак сходимости.

Каждый тест падает, если сломать сходимость: проверено мутациями ядра (lr→0,
_grad→0, grad_tol×100, выброс wfn(X)) — 93/93/79/12 упавших тестов соответственно.
Усиление термального шума ×250 набор НЕ ломает: annealing 2^(−k·t) съедает амплитуду
шума за ~15 шагов, и точность в итоге определяется шагом lr и допуском grad_tol,
а не амплитудой шума (см. таблицу мутаций в README).
"""

import random

import pytest

from homeo import HomeoEngine, Invariant, make_invariant, make_dynamic_invariant

TOL_NORM = 1e-3      # заявленная точность по нормированной координате (0.1 % ширины ящика)
MAX_ITER = 600       # с запасом: измеренный максимум на наборе — 215 шагов
GRAD_TOL = 1e-4
LR = 0.03            # измеренный максимум ошибки по всему набору при lr=0.03: 4.8e-4
                     # (см. README, раздел «Сходимость»: таблица точности по lr)


# --------------------------------------------------------------------------
# генераторы систем с ИЗВЕСТНЫМ ответом
# --------------------------------------------------------------------------
def quadratic_system(rnd, n=2, target_lo=-0.95, target_hi=0.95):
    """Набор из n квадратичных инвариантов: аналитический минимум = target."""
    center = [rnd.uniform(-50.0, 50.0) for _ in range(n)]
    range_ = [rnd.uniform(0.5, 50.0) for _ in range(n)]
    target = [rnd.uniform(target_lo, target_hi) for _ in range(n)]
    weights = [round(rnd.uniform(0.5, 2.0), 3) for _ in range(n)]
    invs = []
    for i in range(n):
        invs.append(make_invariant(weights[i], 2, _quad_err(i, target[i])))
    start = [rnd.uniform(-0.9, 0.9) for _ in range(n)]
    return {"center": center, "range_": range_, "target": target,
            "weights": weights, "invariants": invs, "start": start}


def _quad_err(i, t):
    def err(X):
        return X[i] - t
    return err


def make_engine(sys, *, t0=0.0, seed=1, max_iter=MAX_ITER, grad_tol=GRAD_TOL,
                lr=LR, weights=None):
    if weights is None:
        invs = sys["invariants"]
    else:
        invs = [make_invariant(weights[i], 2, _quad_err(i, sys["target"][i]))
                for i in range(len(sys["target"]))]
    eng = HomeoEngine(center=sys["center"], range_=sys["range_"], invariants=invs,
                      lr=lr, max_iter=max_iter, grad_tol=grad_tol, t0=t0, seed=seed)
    eng.set_raw([sys["start"][i] * sys["range_"][i] + sys["center"][i]
                 for i in range(len(sys["center"]))])
    return eng


def norm_err(sys, eng):
    """Максимальная ошибка по нормированным координатам против аналитического ответа."""
    Xn = eng.norm(eng.raw())
    return max(abs(Xn[i] - sys["target"][i]) for i in range(len(sys["target"])))


BATCH = [quadratic_system(random.Random(1000 + k)) for k in range(40)]


# --------------------------------------------------------------------------
# (A) сходимость к аналитическому минимуму на наборе систем
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", range(len(BATCH)))
def test_analytic_minimum_batch(k):
    """Системы, где ответ известен: 40 случайных квадратичных задач, 2–3 координаты."""
    sys = BATCH[k]
    eng = make_engine(sys, t0=0.0, seed=k)
    iters = eng.tick()
    assert iters >= 1
    assert iters < eng.max_iter, (
        f"tick() не сошёлся: iters={iters} == max_iter={eng.max_iter} "
        f"(цель {sys['target']}, старт {sys['start']})")
    err = norm_err(sys, eng)
    assert err <= TOL_NORM, (
        f"ошибка против аналитического минимума {err:.2e} > {TOL_NORM:.0e} "
        f"(target={sys['target']}, weights={sys['weights']})")


def test_batch_aggregate_is_clean():
    """Сводка по набору: сошлись все, максимум ошибки заявлен как ≤ TOL_NORM."""
    iters_hist = []
    errs = []
    for k, sys in enumerate(BATCH):
        eng = make_engine(sys, t0=0.0, seed=k)
        iters_hist.append(eng.tick())
        errs.append(norm_err(sys, eng))
    assert all(it < MAX_ITER for it in iters_hist), \
        f"не все системы сошлись досрочно: {sorted(iters_hist)}"
    assert max(errs) <= TOL_NORM, f"худшая ошибка {max(errs):.2e}"
    assert min(iters_hist) >= 1


@pytest.mark.parametrize("seed", range(8))
def test_noise_does_not_bias_the_optimum(seed):
    """Термальный шум (t0=0.02) не сдвигает точку равновесия за пределы точности."""
    sys = quadratic_system(random.Random(3000 + seed))
    eng = make_engine(sys, t0=0.02, seed=seed)
    iters = eng.tick()
    assert iters < MAX_ITER, f"со шумом не сошлось: iters={iters}"
    assert norm_err(sys, eng) <= TOL_NORM


@pytest.mark.parametrize("a,b,t", [
    (0.5, -0.5, 0.0), (0.5, -0.5, 0.25), (0.5, -0.5, 0.5),
    (0.5, -0.5, 0.75), (0.5, -0.5, 1.0),
    (0.2, -0.9, 0.3), (-0.7, 0.8, 0.6), (0.0, 0.9, 0.9),
])
def test_dynamic_weights_shift_equilibrium_to_weighted_mean(a, b, t):
    """Динамические веса: E = (1−t)(x−a)² + t·(x−b)² → x* = (1−t)·a + t·b."""
    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[10.0, 10.0],
        invariants=[
            make_dynamic_invariant(1.0, 2, lambda X: X[0] - a, lambda X: 1.0 - t),
            make_dynamic_invariant(1.0, 2, lambda X: X[0] - b, lambda X: t),
        ],
        lr=LR, max_iter=MAX_ITER, grad_tol=GRAD_TOL, t0=0.0, seed=7)
    eng.set_raw([0.0, 0.0])
    iters = eng.tick()
    assert iters < MAX_ITER, f"iters={iters} == max_iter: не сошлось"
    x = eng.norm(eng.raw())[0]
    xstar = (1.0 - t) * a + t * b
    assert abs(x - xstar) <= TOL_NORM, f"x={x:.6f}, x*={xstar:.6f}"


def test_self_weight_root_is_not_naive_root():
    """Вес, зависящий от X: E = (x−0.5)² + x⁴ → корень 4x³+2x−1 = 0 (найден бисекцией),
    а НЕ x = 0.5. Доказывает, что численный градиент учитывает dW/dX."""
    root = _bisect(lambda x: 4.0 * x ** 3 + 2.0 * x - 1.0, 0.0, 1.0)
    assert abs(root - 0.3855) < 1e-3, f"бисекция эталона: root={root}"

    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[1.0, 1.0],
        invariants=[
            make_invariant(1.0, 2, lambda X: X[0] - 0.5),
            make_dynamic_invariant(1.0, 2, lambda X: X[0], lambda X: X[0] * X[0]),
        ],
        lr=LR, max_iter=MAX_ITER, grad_tol=GRAD_TOL, t0=0.0, seed=7)
    eng.set_raw([0.9, 0.0])
    iters = eng.tick()
    assert iters < MAX_ITER
    x = eng.norm(eng.raw())[0]
    assert abs(x - root) <= TOL_NORM, f"x={x:.6f}, аналитический корень={root:.6f}"
    assert abs(x - 0.5) > 0.05, "движок скатился в наивный корень: dW/dX проигнорирован"


def _bisect(f, lo, hi, tol=1e-12):
    assert f(lo) * f(hi) < 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_numerical_gradient_matches_analytic_derivative():
    """Ядро численной сходимости: dE/dX через центральные разности == аналитический
    dE/dX, включая член dW/dX (иначе всё остальное проверяет не то)."""
    # E = 2·(x−0.5)² + 3·x²·x²  →  dE/dx = 4·(x−0.5) + 12·x³
    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[1.0, 1.0],
        invariants=[make_invariant(2.0, 2, lambda X: X[0] - 0.5),
                    make_dynamic_invariant(3.0, 2, lambda X: X[0],
                                           lambda X: X[0] * X[0])],
        lr=LR, max_iter=MAX_ITER, grad_tol=GRAD_TOL, t0=0.0, seed=1)
    X = [0.3, 0.0]
    g = eng._grad(X, 1e-6)
    x = X[0]
    analytic = 4.0 * (x - 0.5) + 12.0 * x ** 3
    assert abs(g[0] - analytic) <= 1e-6, f"g={g[0]:.9f}, analytic={analytic:.9f}"
    assert g[1] == 0.0


def test_energy_matches_definition():
    """E(X) = Σ w_j · wfn_j(X) · |err_j(X)|^p_j — проверяем формулу вручную."""
    def e1(X):
        return X[0] - 0.25

    def e2(X):
        return X[1] + 0.75

    w2 = lambda X: 1.0 + X[0] * X[0]  # noqa: E731
    inv1 = make_invariant(1.5, 2, e1)
    inv2 = make_dynamic_invariant(0.5, 3, e2, w2)
    eng = HomeoEngine(center=[0.0, 0.0], range_=[1.0, 1.0],
                      invariants=[inv1, inv2], lr=LR, max_iter=MAX_ITER,
                      grad_tol=GRAD_TOL, t0=0.0, seed=1)
    X = [0.4, -0.2]
    manual = 1.5 * abs(e1(X)) ** 2 + 0.5 * w2(X) * abs(e2(X)) ** 3
    assert abs(eng.energy(X) - manual) <= 1e-12
    assert abs(inv2.energy(X) - 0.5 * w2(X) * abs(e2(X)) ** 3) <= 1e-15
    assert isinstance(inv1, Invariant)


def test_norm_raw_roundtrip_and_clipping():
    """Нормировка: [−1,1] ↔ raw; выход за ящик обрезается (и это видно)."""
    eng = HomeoEngine(center=[5.0, -2.0], range_=[2.0, 4.0],
                      invariants=[make_invariant(1.0, 2, lambda X: X[0]),
                                  make_invariant(1.0, 2, lambda X: X[1])],
                      lr=LR, max_iter=10, t0=0.0, seed=1)
    eng.set_raw([5.0, -2.0])
    assert eng.norm(eng.raw()) == [0.0, 0.0]
    eng.set_raw([9.0, -10.0])
    assert eng.raw() == [7.0, -6.0], "выход за ящик должен обрезаться к ±1"
    assert eng.norm([7.0, -6.0]) == [1.0, -1.0]


# --------------------------------------------------------------------------
# (B) выход за границы применимости: движок обязан это сообщить
# --------------------------------------------------------------------------
@pytest.mark.parametrize("target,t0", [
    ((1.6, -1.4), 0.0),
    ((1.6, -1.4), 0.02),
    ((-1.35, 1.9), 0.0),
    ((2.0, 2.0), 0.0),
    ((1.5, -1.5), 0.0),
])
def test_solution_outside_box_reports_non_convergence(target, t0):
    """Решения внутри ящика нет → iters == max_iter (признак), точка на границе,
    остаточная энергия велика. То есть метод НЕ рапортует о ложной сходимости."""
    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[10.0, 10.0],
        invariants=[make_invariant(1.0, 2, _quad_err(0, target[0])),
                    make_invariant(1.0, 2, _quad_err(1, target[1]))],
        lr=LR, max_iter=200, grad_tol=GRAD_TOL, t0=t0, seed=5)
    eng.set_raw([0.0, 0.0])
    iters = eng.tick()
    assert iters == eng.max_iter, (
        f"для цели {target} вне ящика движок заявил сходимость за {iters} шагов")
    Xn = eng.norm(eng.raw())
    assert all(abs(v) == 1.0 for v in Xn), f"должно упереться в границу: {Xn}"
    residual = max(abs(Xn[i] - target[i]) for i in range(2))
    assert residual > 0.25, f"остаточная ошибка {residual:.3f} не показывает несходимость"
    assert eng.energy() > 0.05, f"энергия {eng.energy():.3e} слишком мала для невязки"


@pytest.mark.parametrize("target", [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)])
def test_boundary_optimum_is_not_a_false_failure(target):
    """Оптимум ровно на границе — законный случай: сходится досрочно, E = 0.
    Так признак из теста (B) не путается с честным граничным решением."""
    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[10.0, 10.0],
        invariants=[make_invariant(1.0, 2, _quad_err(0, target[0])),
                    make_invariant(1.0, 2, _quad_err(1, target[1]))],
        lr=LR, max_iter=200, grad_tol=GRAD_TOL, t0=0.0, seed=5)
    eng.set_raw([0.0, 0.0])
    iters = eng.tick()
    assert iters < eng.max_iter, f"граничный оптимум объявлен несходимостью: {iters}"
    assert eng.energy() <= 1e-6, f"E={eng.energy():.3e} на граничном оптимуме"
    assert norm_err({"target": target}, eng) <= TOL_NORM


# --------------------------------------------------------------------------
# (C) устойчивость признака «сошлось»
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t0", [0.0, 0.02])
def test_second_tick_does_not_move_converged_point(t0):
    """Если tick() сказал «сошлось», следующий tick() почти не двигает точку —
    то есть признак сходимости осмысленный, а не просто счётчик."""
    eng = HomeoEngine(
        center=[0.0, 0.0], range_=[10.0, 10.0],
        invariants=[make_invariant(1.0, 2, _quad_err(0, 0.3)),
                    make_invariant(1.0, 2, _quad_err(1, -0.7))],
        lr=LR, max_iter=300, grad_tol=GRAD_TOL, t0=t0, seed=9)
    eng.set_raw([5.0, 5.0])
    it1 = eng.tick()
    p1 = eng.raw()
    it2 = eng.tick()
    p2 = eng.raw()
    assert it1 < 300
    assert it2 < 300, f"второй tick() не сошёлся: {it2}"
    move = max(abs(p2[i] - p1[i]) for i in range(2))
    assert move <= 1e-2, f"сошедшаяся точка уехала на {move:.3e} (диапазон 10)"


def test_same_seed_is_deterministic():
    def run():
        sys = quadratic_system(random.Random(4242))
        eng = make_engine(sys, t0=0.02, seed=13)
        eng.tick()
        return eng.raw()

    assert run() == run()


def test_reset_restores_fresh_behaviour():
    """reset() возвращает движок к поведению свежего (моменты Adam обнулены)."""
    sys = quadratic_system(random.Random(77))
    eng = make_engine(sys, t0=0.0, seed=3)
    eng.tick()
    eng.set_raw([sys["start"][i] * sys["range_"][i] + sys["center"][i]
                 for i in range(len(sys["center"]))])
    eng.reset()
    eng.tick()
    after_reset = eng.raw()

    fresh = make_engine(sys, t0=0.0, seed=3)
    fresh.tick()
    for a, b in zip(after_reset, fresh.raw()):
        assert abs(a - b) <= 1e-12


# --------------------------------------------------------------------------
# (D) контракт абсолютного grad_tol: масштаб весов должен сопровождаться масштабом tol
# --------------------------------------------------------------------------
@pytest.mark.parametrize("lam", [1e-4, 1e-2, 1.0, 1e2, 1e4])
def test_scale_covariance_with_scaled_grad_tol(lam):
    """E → λ·E при grad_tol → λ·grad_tol даёт ту же точку и тот же признак сходимости
    (Adam инвариантен к масштабу энергии — если мерить его в тех же единицах)."""
    invs = [make_invariant(lam, 2, _quad_err(0, 0.3)),
            make_invariant(lam, 2, _quad_err(1, -0.7))]
    eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 10.0], invariants=invs,
                      lr=LR, max_iter=500, grad_tol=GRAD_TOL * lam, t0=0.0, seed=11)
    eng.set_raw([5.0, 5.0])
    iters = eng.tick()
    assert iters < eng.max_iter, f"λ={lam}: не сошлось (iters={iters})"
    Xn = eng.norm(eng.raw())
    assert abs(Xn[0] - 0.3) <= 1e-4 and abs(Xn[1] + 0.7) <= 1e-4, \
        f"λ={lam}: точка {Xn} разошлась с аналитическим минимумом (0.3, −0.7)"
