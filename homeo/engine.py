"""Ядро гомеостатического контроллера: инварианты, Adam-релаксация, термальный шум. Порт с C++ CIDHC."""


import math
import random


class Invariant:
    """Инвариант: (weight, p, err_fn, wfn). err: X->float, wfn: X->float (аналитический вес)."""

    __slots__ = ("weight", "p", "err", "wfn")

    def __init__(self, weight, p, err, wfn=None):
        self.weight = float(weight)
        self.p = int(p)
        self.err = err
        self.wfn = wfn if wfn is not None else (lambda X: 1.0)

    def energy(self, X):
        return self.weight * self.wfn(X) * abs(self.err(X)) ** self.p


def make_invariant(weight, p, err):
    """Статический вес."""
    return Invariant(weight, p, err)


def make_dynamic_invariant(weight, p, err, wfn):
    """Динамический вес: итоговый вклад = weight · wfn(X) · |err|^p."""
    return Invariant(weight, p, err, wfn)


class HomeoEngine:
    """
    Релаксация: X_norm -> минимум E. Один тик = один проход Adam по всем dims.
    """

    def __init__(self, center, range_, invariants,
                 lr=0.1, max_iter=100, grad_tol=1e-5,
                 t0=0.02, anneal_k=1.0, seed=42, grad_eps=1e-4):
        self.center = list(map(float, center))
        self.range = list(map(float, range_))
        self.n = len(center)
        self.invariants = list(invariants)
        self.lr = lr
        self.max_iter = max_iter
        self.grad_tol = grad_tol
        self.t0 = t0
        self.anneal_k = anneal_k
        self.grad_eps = grad_eps
        self.rng = random.Random(seed)
        # Adam-моменты
        self.m = [0.0] * self.n
        self.v = [0.0] * self.n
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        self._Xnorm = [0.0] * self.n
        self._iter = 0

    # ---- координаты ----
    def norm(self, raw):
        """raw -> [−1,1]"""
        return [max(-1.0, min(1.0, (raw[i] - self.center[i]) / self.range[i]))
                for i in range(self.n)]

    def raw(self):
        """[−1,1] -> raw (развернуть)"""
        return [self._Xnorm[i] * self.range[i] + self.center[i]
                for i in range(self.n)]

    def set_raw(self, raw):
        self._Xnorm = self.norm(raw)

    # ---- энергия ----
    def energy(self, Xnorm=None):
        if Xnorm is None:
            Xnorm = self._Xnorm
        return sum(inv.energy(Xnorm) for inv in self.invariants)

    # ---- градиент (центральные разности, учитывает dW/dX) ----
    def _grad(self, X, h):
        g = [0.0] * self.n
        for i in range(self.n):
            Xp = X[:]
            Xm = X[:]
            Xp[i] = min(1.0, Xp[i] + h)
            Xm[i] = max(-1.0, Xm[i] - h)
            g[i] = (self.energy(Xp) - self.energy(Xm)) / (2.0 * h)
        return g

    # ---- один шаг релаксации ----
    def _step(self):
        X = self._Xnorm
        # адаптивный h: удваиваем, пока градиент не станет ненулевым
        h = self.grad_eps
        g = self._grad(X, h)
        for _ in range(12):
            if any(abs(gi) > 0.0 for gi in g):
                break
            h *= 2.0
            g = self._grad(X, h)

        self._iter += 1
        for i in range(self.n):
            gi = g[i]
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * gi
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * gi * gi
            mhat = self.m[i] / (1.0 - self.b1 ** self._iter)
            vhat = self.v[i] / (1.0 - self.b2 ** self._iter)
            step = self.lr * mhat / (math.sqrt(vhat) + self.eps)
            # термальный шум CLT: (12·U−6) ~ N(0,1), anneal T = T0·2^(−k·t)
            T = self.t0 * (2.0 ** (-self.anneal_k * (self._iter - 1)))
            noise = 0.0
            for _ in range(12):
                noise += self.rng.random()
            noise = (noise - 6.0) * T
            X[i] = max(-1.0, min(1.0, X[i] - step + noise))

        return g

    # ---- тик: релаксация до сходимости ----
    def tick(self):
        g = self._step()
        iters = 1
        while iters < self.max_iter:
            g = self._step()
            iters += 1
            gm = max(abs(gi) for gi in g)
            if gm < self.grad_tol:
                break
        return iters

    def reset(self):
        self.m = [0.0] * self.n
        self.v = [0.0] * self.n
        self._iter = 0
