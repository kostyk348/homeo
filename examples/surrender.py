"""Пример: surrender — переключение приоритетов под растущей угрозой.

Два инварианта с динамическими весами:
  I_goal    : X[0] − 0.5,  вес = 1 − threat   ("наступать")
  I_retreat : X[0] + 0.5,  вес = threat       ("отступать")
Равновесие смещается x* = 5 − 10·threat: пока угрозы нет — цель, с ростом
угрозы — отступление.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homeo import HomeoEngine, make_dynamic_invariant

print("threat  x*          x        ok")
for t in (0.0, 0.25, 0.5, 0.75, 1.0):
    eng = HomeoEngine(center=[0.0, 0.0], range_=[10.0, 1.0],
                      invariants=[
                          make_dynamic_invariant(1.0, 2,
                                                 lambda X: X[0] - 0.5,
                                                 lambda X: 1.0 - t),
                          make_dynamic_invariant(1.0, 2,
                                                 lambda X: X[0] + 0.5,
                                                 lambda X: t),
                      ],
                      lr=0.1, max_iter=400, grad_tol=1e-4, t0=0.02, seed=7)
    eng.set_raw([0.0, t])
    eng.tick()
    x = eng.raw()[0]
    xstar = 5.0 - 10.0 * t
    ok = abs(x - xstar) < 0.05
    print(f"{t:5.2f}   {xstar:8.2f}  {x:8.3f}  {'PASS' if ok else 'FAIL'}")
