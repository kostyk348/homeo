"""Пример: двухмерная "чаша" — сходимость к заданной точке из стартовой.

X = [x, y]; инварианты тянут x → 1.5, y → −2.0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homeo import HomeoEngine, make_invariant

eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 10.0],
                  invariants=[
                      make_invariant(1.0, 2, lambda X: X[0] - 0.15),   # x → 1.5
                      make_invariant(1.0, 2, lambda X: X[1] + 0.20),   # y → −2.0
                  ],
                  lr=0.1, max_iter=400, grad_tol=1e-4, t0=0.02, seed=42)

eng.set_raw([8.0, -8.0])
iters = eng.tick()
x, y = eng.raw()
print(f"[bowl] start (8, -8) -> ({x:.3f}, {y:.3f}), iters={iters}")
print("expect (1.5, -2.0) —", "PASS" if abs(x - 1.5) < 0.01 and abs(y + 2.0) < 0.01 else "FAIL")
