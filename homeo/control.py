"""Валидационный стенд «два связанных резервуара» + контроллеры + прогон эпизода.

Модуль добавлен при валидационном апгрейде (аудит 2026-11). Он НЕ меняет ядро
homeo/engine.py, а подключает его как контроллер в замкнутый контур:

  * объект — два связанных резервуара (линейный стенд и демо-вариант с
    квадратичными перетоками);
  * допустимое множество Ω = {x : g_j(x) ≤ 0} — полосы уровней обоих баков
    (см. docs/formal-setup.md);
  * HomeoTankController — инварианты оценивают ПОСЛЕДСТВИЯ политики
    (предсказанную траекторию уровней при постоянном u на горизонте), а не
    только текущее измерение: так численный градиент по управлению отличен
    от нуля и релаксация реально управляет;
  * PIDTankController — честный базовый ПИД с anti-windup (условное
    интегрирование) и производной по измерению с фильтром;
  * FixedGainHomeoEngine — абляция «без адаптивного шага» (Adam → GD с
    фиксированным шагом);
  * run_episode — прогон с ЗАРАНЕЕ сгенерированными потоками возмущений и шума
    (одинаковые потоки для всех контроллеров — честное сравнение; фиксированный
    seed — полная воспроизводимость).

Только stdlib (math, random): нулевых рантайм-зависимостей это не ломает.
"""

import math
import random

from .engine import HomeoEngine
from .safety import filter_control


# --------------------------------------------------------------------------
# Объект: два связанных резервуара
# --------------------------------------------------------------------------
class TwoTankPlant:
    """A1·h1' = u − q12(h1,h2);  A2·h2' = q12(h1,h2) − qout(h2) − d

    linear (стенд):  q12 = c12·(h1 − h2),       qout = c2·h2
    sqrt (демо):     q12 = k12·sign(Δ)·√|Δ|,   qout = k2·√h2

    u — насос в бак 1 (ограничен [0, umax] — это физика привода), d — расход
    потребителя из бака 2 (неизвестное контроллеру возмущение). Интегрирование —
    явный Эйлер с шагом dt. Параметры — изменяемые атрибуты: эксперименты
    дрейфуют их для проверки робастности.
    """

    def __init__(self, A1=1.0, A2=1.0, c12=0.7, c2=0.35, umax=1.5,
                 h1=0.5, h2=0.5, nonlinear=False, dt=0.25):
        self.A1 = float(A1)
        self.A2 = float(A2)
        self.c12 = float(c12)
        self.c2 = float(c2)
        self.umax = float(umax)
        self.h1 = float(h1)
        self.h2 = float(h2)
        self.nonlinear = bool(nonlinear)
        self.dt = float(dt)

    def flows(self, h1, h2):
        """Переток между баками и расход потребителя при уровнях (h1, h2)."""
        if self.nonlinear:
            dh = h1 - h2
            q12 = self.c12 * math.copysign(math.sqrt(abs(dh)), dh)
            qout = self.c2 * math.sqrt(max(0.0, h2))
        else:
            q12 = self.c12 * (h1 - h2)
            qout = self.c2 * h2
        return q12, qout

    def step(self, u, d=0.0):
        """Один шаг Эйлера; возвращает новые уровни (h1, h2)."""
        u = max(0.0, min(self.umax, u))          # физика привода (насыщение)
        q12, qout = self.flows(self.h1, self.h2)
        self.h1 += self.dt * (u - q12) / self.A1
        self.h2 += self.dt * (q12 - qout - d) / self.A2
        self.h1 = max(0.0, self.h1)
        self.h2 = max(0.0, self.h2)
        return self.h1, self.h2


class TwoTankModel(TwoTankPlant):
    """Номинальная модель объекта для предсказания (контроллер и safety-фильтр).

    Отдельный экземпляр: возмущение НЕ известно контроллеру, а параметры модели
    могут быть смещены относительно объекта (честная неопределённость в демо).
    step(x, u, d, k) совместим с safety.filter_control.
    """

    def step(self, x, u=0.0, d=0.0, k=0):
        h1, h2 = x
        u = max(0.0, min(self.umax, u))
        q12, qout = self.flows(h1, h2)
        d1 = d[0] if isinstance(d, (list, tuple)) else 0.0
        d2 = d[1] if isinstance(d, (list, tuple)) else (d or 0.0)
        h1n = max(0.0, h1 + self.dt * (u - q12) / self.A1)
        h2n = max(0.0, h2 + self.dt * (q12 - qout - d2) / self.A2)
        return [h1n, h2n]

    def predict(self, h1, h2, u, horizon, d=0.0):
        """Траектория уровней при ПОСТОЯННОМ u (и опционально постоянном
        возмущении d — оценка расхода потребителя) на горизонте шагов."""
        x = [h1, h2]
        trajs = ([], [])
        for _ in range(int(horizon)):
            x = self.step(x, u, d)
            trajs[0].append(x[0])
            trajs[1].append(x[1])
        return trajs


# --------------------------------------------------------------------------
# Сценарий эпизода
# --------------------------------------------------------------------------
class Scenario:
    """Эпизод валидации: полосы допустимости (Ω), уставка, возмущение, шум.

    d_profile(t) — детерминированный профиль расхода потребителя [м³/с];
    sigma — СКО шума датчиков уровней [м]; шум генерируется в run_episode
    по фиксированному seed (одинаково для всех контроллеров).
    """

    def __init__(self, steps, dt, h1_band, h2_band, ref, d_profile, sigma=0.0, umax=1.5):
        self.steps = int(steps)
        self.dt = float(dt)
        self.h1_band = (float(h1_band[0]), float(h1_band[1]))
        self.h2_band = (float(h2_band[0]), float(h2_band[1]))
        self.ref = float(ref)
        self.d_profile = d_profile
        self.sigma = float(sigma)
        self.umax = float(umax)

    def g_funcs(self):
        """Ω = {x : g_j(x) ≤ 0} — четыре неравенства-полосы (формализм аудита)."""
        lo1, hi1 = self.h1_band
        lo2, hi2 = self.h2_band
        return [
            lambda x: lo1 - x[0],
            lambda x: x[0] - hi1,
            lambda x: lo2 - x[1],
            lambda x: x[1] - hi2,
        ]


def make_safety_filter(model, scen, *, d_bound, horizon=8, n_grid=41, margin=0.0):
    """Проекционный safety-фильтр на номинальной модели с worst-case возмущением.

    d_bound — известная верхняя граница расхода потребителя: фильтр проверяет
    предсказанную траекторию при d = d_bound (консервативно) и проецирует
    кандидата на множество управлений, удерживающих траекторию в Ω с запасом
    margin. Возвращает callable (x, u_cand, k) -> (u, info).
    """
    g_funcs = scen.g_funcs()

    def d_check(j):
        return d_bound

    def fn(x, u_cand, k):
        u, info = filter_control(model.step, x, u_cand, 0.0, scen.umax,
                                 g_funcs=g_funcs, horizon=horizon,
                                 d_check=d_check, n_grid=n_grid, tol=margin, k0=k)
        return u, info

    return fn


# --------------------------------------------------------------------------
# Контроллер homeo в контуре
# --------------------------------------------------------------------------
class HomeoTankController:
    """homeo как контроллер: X = [u] (нормированный расход насоса).

    Инварианты оценивают последствия: предсказанную траекторию уровней при
    постоянном u на горизонте (номинальная модель, без возмущений):
      * цель      : (h2_bar − ref)/scale2,              вес w_goal;
      * полосы    : max по траектории глубины выхода за каждую полосу,  вес w_safe
                    (односторонние шарнирные штрафы — barrier-подобные);
      * экономия  : (u − u_prev)/umax,                  вес w_eff.
    Каждый тик — свежая релаксация (engine.reset() + тёплый старт от u_prev).
    """

    def __init__(self, model, scen, *, horizon=16, lr=0.15, t0=0.02, anneal_k=1.0,
                 seed=11, grad_tol=1e-4, max_iter=80,
                 w_goal=1.0, w_safe=25.0, w_eff=0.05,
                 use_safety_invariants=True, engine_cls=HomeoEngine, d_hat_beta=0.2,
                 safety_margin=0.03, engine_kwargs=None):
        self.model = model
        self.scen = scen
        self.horizon = int(horizon)
        self.w_goal = float(w_goal)
        self.w_safe = float(w_safe)
        self.w_eff = float(w_eff)
        self.use_safety_invariants = bool(use_safety_invariants)
        self.d_hat_beta = float(d_hat_beta)
        self.safety_margin = float(safety_margin)  # барьер держит уровень на margin глубже границы
        self.d_hat = 0.0          # оценка расхода потребителя по измерениям
        self._y2_prev = None
        half = scen.umax / 2.0
        self.half = half
        self.diag = {}
        ekw = dict(lr=lr, max_iter=max_iter, grad_tol=grad_tol, t0=t0,
                   anneal_k=anneal_k, seed=seed)
        if engine_kwargs:
            ekw.update(engine_kwargs)   # напр. h_doublings=0, project=False — для абляций
        self.eng = engine_cls(center=[half], range_=[half], invariants=[], **ekw)

    def reset(self):
        self.diag = {}
        self.d_hat = 0.0
        self._y2_prev = None

    def act(self, y, u_prev, k):
        h1m, h2m = y
        # оценка возмущения: d = q12 − qout − dh2/dt (из измерений и номинальной
        # модели потоков), сглаживание 1-го порядка. Оценка включает и ошибку
        # модели параметров — это честная неопределённость, она же даёт контроллеру
        # интегралоподобное поведение при постоянном возмущении.
        if self._y2_prev is not None and self.scen.dt > 0.0:
            q12, qout = self.model.flows(h1m, h2m)
            d_inst = (q12 - qout) - (h2m - self._y2_prev) / self.scen.dt
            b = self.d_hat_beta
            self.d_hat = (1.0 - b) * self.d_hat + b * d_inst
        self._y2_prev = h2m
        d_hat = self.d_hat
        lo1, hi1 = self.scen.h1_band
        lo2, hi2 = self.scen.h2_band
        s1 = hi1 - lo1
        s2 = hi2 - lo2
        ref = self.scen.ref
        umax = self.scen.umax
        half = self.half
        model = self.model
        H = self.horizon

        cache_u = [None]
        cache_traj = [None]

        def traj(u):
            if cache_u[0] != u:
                cache_u[0] = u
                cache_traj[0] = model.predict(h1m, h2m, u, H, d=d_hat)
            return cache_traj[0]

        def goal_err(X):
            u = X[0] * half + half
            _, t2 = traj(u)
            bar = sum(t2[H // 2:]) / max(1, len(t2[H // 2:]))
            return (bar - ref) / s2

        def hinge(idx, hi, lo, scale):
            def err(X):
                u = X[0] * half + half
                t = traj(u)[idx]
                depth = 0.0
                for v in t:
                    if v > hi:
                        depth = max(depth, v - hi)
                    elif v < lo:
                        depth = max(depth, lo - v)
                return depth / scale
            return err

        def effort_err(X):
            u = X[0] * half + half
            return (u - u_prev) / umax

        invs = [_make_inv(self.w_goal, 2, goal_err)]
        if self.use_safety_invariants:
            # барьер-подобные шарниры с ЗАПАСОМ safety_margin внутрь полосы:
            # минимизатор шарнира (у которого градиент нулевой строго внутри
            # допустимого множества) дребезжит вокруг границы — запас уводит
            # дребезг от настоящей границы и гасит ошибку оценки возмущения
            m = self.safety_margin
            invs.append(_make_inv(self.w_safe, 2, hinge(0, hi1 - m, -math.inf, s1)))
            invs.append(_make_inv(self.w_safe, 2, hinge(0, math.inf, lo1 + m, s1)))
            invs.append(_make_inv(self.w_safe, 2, hinge(1, hi2 - m, -math.inf, s2)))
            invs.append(_make_inv(self.w_safe, 2, hinge(1, math.inf, lo2 + m, s2)))
        invs.append(_make_inv(self.w_eff, 2, effort_err))

        self.eng.invariants = invs
        self.eng.reset()
        self.eng.set_raw([u_prev])
        iters = self.eng.tick()
        u = self.eng.raw()[0]
        self.diag[k] = {"iters": iters, "energy": self.eng.energy()}
        return u


def _make_inv(weight, p, err):
    from .engine import make_invariant
    return make_invariant(weight, p, err)


class FixedGainHomeoEngine(HomeoEngine):
    """Абляция «без адаптивного шага»: вместо Adam — градиентный спуск с
    фиксированным шагом lr (моменты m/v и их нормировка отключены)."""

    def _step(self):
        X = self._Xnorm
        h = self.grad_eps
        g = self._grad(X, h)
        for _ in range(self.h_doublings):
            if any(abs(gi) > 0.0 for gi in g):
                break
            h *= 2.0
            g = self._grad(X, h)
        self._iter += 1
        T = self.t0 * (2.0 ** (-self.anneal_k * (self._iter - 1)))
        for i in range(self.n):
            noise = 0.0
            for _ in range(12):
                noise += self.rng.random()
            noise = (noise - 6.0) * T
            if self.project:
                X[i] = max(-1.0, min(1.0, X[i] - self.lr * g[i] + noise))
            else:
                X[i] = X[i] - self.lr * g[i] + noise
        return g


# --------------------------------------------------------------------------
# Базовый ПИД с anti-windup
# --------------------------------------------------------------------------
class PIDTankController:
    """ПИД по уровню бака 2: u = Kp·e + Ki·∫e + Kd·de/dt, e = ref − h2.

    anti-windup — условное интегрирование: интегратор замирает, когда привод
    насыщен и ошибка толкает дальше в насыщение. Производная — по измерению,
    с фильтром 1-го порядка (τ = d_tau). Ограничение выхода [0, umax].
    """

    def __init__(self, scen, kp, ki, kd=0.0, d_tau=0.5):
        self.scen = scen
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.d_tau = float(d_tau)
        self.reset()

    def reset(self):
        self._I = 0.0
        self._df = 0.0
        self._h2_prev = None

    def act(self, y, u_prev, k):
        dt = self.scen.dt
        e = self.scen.ref - y[1]
        if self._h2_prev is None or dt <= 0.0:
            raw_d = 0.0
        else:
            raw_d = (y[1] - self._h2_prev) / dt
        alpha = dt / (self.d_tau + dt) if (self.d_tau + dt) > 0 else 1.0
        self._df = alpha * raw_d + (1.0 - alpha) * self._df
        self._h2_prev = y[1]

        u_unsat = self.kp * e + self._I - self.kd * self._df
        u = max(0.0, min(self.scen.umax, u_unsat))
        # условное интегрирование (clamping)
        if (u_unsat < self.scen.umax or e < 0.0) and (u_unsat > 0.0 or e > 0.0):
            self._I += e * dt
        return u


# --------------------------------------------------------------------------
# Прогон эпизода
# --------------------------------------------------------------------------
def run_episode(plant, controller, scen, *, safety=None, noise_seed=0,
                d_scale=1.0, on_step=None):
    """Прогон замкнутого контура с заранее сгенерированными потоками.

    Возмущение d[k] = scen.d_profile(t_k)·d_scale и шум датчиков (σ = scen.sigma,
    seed = noise_seed) генерируются ДО старта: все контроллеры видят одинаковые
    потоки (честное сравнение), фиксированный seed даёт точную воспроизводимость.

    safety — callable (x, u_cand, k) -> (u, info) из make_safety_filter (или None).
    on_step(plant, k, t) — хук для дрейфа параметров объекта (робастность).

    Возвращает dict со списками: t, h1, h2, u, u_cand, d, y1, y2, triggers,
    filter_infos.
    """
    rnd = random.Random(noise_seed)
    noise1 = [rnd.gauss(0.0, scen.sigma) for _ in range(scen.steps)]
    noise2 = [rnd.gauss(0.0, scen.sigma) for _ in range(scen.steps)]
    d = [scen.d_profile(k * scen.dt) * d_scale for k in range(scen.steps)]

    plant.h1 = float(plant.h1)
    plant.h2 = float(plant.h2)
    if hasattr(controller, "reset"):
        controller.reset()

    t_s, h1_s, h2_s, u_s, u_c_s, d_s, y1_s, y2_s = ([] for _ in range(8))
    triggers, filter_infos = [], []
    u_prev = 0.0
    for k in range(scen.steps):
        t = k * scen.dt
        y1 = plant.h1 + noise1[k]
        y2 = plant.h2 + noise2[k]
        u_cand = controller.act([y1, y2], u_prev, k)
        u_cand = max(0.0, min(plant.umax, u_cand))
        if safety is not None:
            u, info = safety([plant.h1, plant.h2], u_cand, k)
            triggers.append(1 if info["triggered"] else 0)
            filter_infos.append(info)
        else:
            u, info = u_cand, None
            triggers.append(0)
            filter_infos.append(None)
        if on_step is not None:
            on_step(plant, k, t)
        h1, h2 = plant.step(u, d[k])
        u_prev = u
        t_s.append(t)
        h1_s.append(plant.h1)
        h2_s.append(plant.h2)
        u_s.append(u)
        u_c_s.append(u_cand)
        d_s.append(d[k])
        y1_s.append(y1)
        y2_s.append(y2)

    return {"t": t_s, "h1": h1_s, "h2": h2_s, "u": u_s, "u_cand": u_c_s,
            "d": d_s, "y1": y1_s, "y2": y2_s, "triggers": triggers,
            "filter_infos": filter_infos}


def scenario_constraints(ep, scen):
    """Ограничения эпизода в формате homeo.metrics: [(series, (lo, hi)), ...]."""
    return [(ep["h1"], scen.h1_band), (ep["h2"], scen.h2_band)]
