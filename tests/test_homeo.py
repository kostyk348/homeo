"""Тесты ядра homeo (перенесены из C++ test_bowl.cpp).

1. bowl         — двухмерная чаша: сходимость к точке.
2. surrender    — динамические веса: равновесие x* = 5 − 10·threat.
3. self_weight  — вес зависит от X: минимум E=(x−0.5)² + x²·x² в ~3.855,
                  доказывает учёт dW/dX в численном градиенте.
"""

from homeo import HomeoEngine, make_invariant, make_dynamic_invariant


def test_bowl():
    eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 10.0],
                      invariants=[
                          make_invariant(1.0, 2, lambda X: X[0] - 0.15),   # цель x = 1.5
                          make_invariant(1.0, 2, lambda X: X[1] + 0.20),   # цель y = −2.0
                      ],
                      lr=0.1, max_iter=400, grad_tol=1e-4, t0=0.02, seed=42)
    eng.set_raw([8.0, -8.0])
    eng.tick()
    x, y = eng.raw()
    assert abs(x - 1.5) < 0.01, f"x={x:.4f}, exp 1.5"
    assert abs(y + 2.0) < 0.01, f"y={y:.4f}, exp -2.0"


def test_surrender():
    """Смена приоритетов: w_goal = 1−threat, w_retreat = threat → x* = 5 − 10·threat."""
    threat = [0.0]
    invs = [
        make_dynamic_invariant(1.0, 2,
                               lambda X: X[0] - 0.5,
                               lambda X: 1.0 - threat[0]),
        make_dynamic_invariant(1.0, 2,
                               lambda X: X[0] + 0.5,
                               lambda X: threat[0]),
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        threat[0] = t
        eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 1.0], invariants=invs,
                          lr=0.1, max_iter=400, grad_tol=1e-4, t0=0.02, seed=7)
        eng.set_raw([0.0, t])
        eng.tick()
        x = eng.raw()[0]
        assert abs(x - (5.0 - 10.0 * t)) < 0.05, \
            f"threat={t}: x={x:.4f}, exp {5.0 - 10.0 * t:.4f}"


def test_self_weight():
    """E = (x−0.5)² + x²·x² → минимум ~3.855, НЕ 5.0.
    Если градиент игнорировал бы dW/dX, равновесие было бы в 5.0."""
    invs = [
        make_invariant(1.0, 2, lambda X: X[0] - 0.5),
        make_dynamic_invariant(1.0, 2, lambda X: X[0], lambda X: X[0] * X[0]),
    ]
    eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 10.0], invariants=invs,
                      lr=0.1, max_iter=400, grad_tol=1e-4, t0=0.02, seed=7)
    eng.set_raw([0.0, 0.0])
    eng.tick()
    x = eng.raw()[0]
    assert 2.5 < x < 4.5, f"x={x:.4f}, exp ~3.855"
