"""LQR-базовый контроллер для линейного стенда (единственное место с scipy).

Дизайн на линейном стенде:
  * непрерывная модель  A = [[−c12/A1, c12/A1], [c12/A2, −(c12+c2)/A2]],  B = [1/A1, 0];
  * дискретизация c2d (zero-order hold) с шагом dt;
  * интегратор ошибки слежения xi_{k+1} = xi_k + dt·(ref − h2_k) — иначе LQR не
    отбрасывает постоянное возмущение (расход потребителя);
  * DARE через scipy.linalg.solve_discrete_are, K = (R + BᵀPB)⁻¹BᵀPA;
  * управление u = u_ss − K·(xa − xa_ss), насыщение [0, umax] + anti-windup
    (интегратор замирает в насыщении, как у ПИД).

Веса Q, R подобраны вручную один раз и НЕ перенастраиваются под сценарии.
На демо (нелинейный объект) LQR применяется БЕЗ перенастройки — это честная
проверка «спроектирован на линейном стенде, развёрнут на нелинейном объекте».
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from homeo.control import PIDTankController  # noqa: F401  (переиспользуем интерфейс)

# единственная точка правки весов; документированы в docs/validation.md
Q_DIAG = (2.0, 8.0, 0.4)
R_COEF = 0.05


def design_linear_stand(c12=0.7, c2=0.35, A1=1.0, A2=1.0, dt=0.25):
    """Возвращает (K, Ad, Bd, dt) для линейного стенда."""
    Mc = np.array([[-c12 / A1, c12 / A1],
                   [c12 / A2, -(c12 + c2) / A2]])
    Bc = np.array([[1.0 / A1], [0.0]])
    Ad, Bd, *_ = cont2discrete((Mc, Bc, np.eye(2), np.zeros((2, 1))), dt)
    Aa = np.vstack([np.hstack([Ad, np.zeros((2, 1))]),
                    np.array([[-dt, 0.0, 1.0]])])
    Ba = np.vstack([Bd, [[0.0]]])
    Q = np.diag(Q_DIAG)
    R = np.array([[R_COEF]])
    P = solve_discrete_are(Aa, Ba, Q, R)
    K = np.linalg.solve(R + Ba.T @ P @ Ba, Ba.T @ P @ Aa)
    return K, Ad, Bd, dt


class LQRTankController:
    """u = u_ss − K·([h1, h2, xi] − xa_ss); xa_ss — равновесие при ref и d=0."""

    def __init__(self, scen, K=None, h1_ss=None, u_ss=None):
        from experiments.common import lin_model  # параметры дизайна — стенда
        m = lin_model()
        self.K, _, _, _ = design_linear_stand(c12=m.c12, c2=m.c2, A1=m.A1, A2=m.A2,
                                              dt=scen.dt) if K is None else (K, None, None, scen.dt)
        self.scen = scen
        c2, c12 = m.c2, m.c12
        self.h1_ss = h1_ss if h1_ss is not None else scen.ref + c2 * scen.ref / c12
        self.u_ss = u_ss if u_ss is not None else c2 * scen.ref
        self.reset()

    def reset(self):
        self.xi = 0.0

    def act(self, y, u_prev, k):
        dt = self.scen.dt
        e = self.scen.ref - y[1]
        dx = np.array([y[0] - self.h1_ss, y[1] - self.scen.ref, self.xi])
        u_unsat = self.u_ss - float(self.K @ dx)
        u = max(0.0, min(self.scen.umax, u_unsat))
        # anti-windup: интегратор замирает в насыщении в сторону ошибки
        if (u_unsat < self.scen.umax or e < 0.0) and (u_unsat > 0.0 or e > 0.0):
            self.xi += e * dt
        return u


def pid_gains_from_json():
    """Настроенные сеткой коэффициенты ПИД из results/baselines.json.

    Если файла ещё нет (эксперименты запускаются по отдельности в другом
    порядке) — сетка настройки прогоняется на месте (детерминирована, ~15 с).
    """
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "baselines.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)["pid"]["gains"]
        return g["Kp"], g["Ki"], g["Kd"]
    from experiments.run_baselines import tune_pid
    return tune_pid()
