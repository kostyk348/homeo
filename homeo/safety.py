"""Safety-слой: измерение нарушения допустимости и проекционный фильтр управления.

Модуль добавлен при валидационном апгрейде (аудит 2026-11): ядро homeo само по себе
умеет только обрезку в ящик нормированных координат (проекция на бокс) и мягкие
штрафы в энергии E. Здесь — то, чего в ядре нет:

  * `max_violation(x, g, h)` — численная мера нарушения допустимого множества
    Ω = {x : g_j(x) ≤ 0, h_i(x) = 0}:  max( max_j g_j(x), max_i |h_i(x)| ), 0 внутри Ω;
  * `filter_control(...)` — однокоординатный проекционный safety-фильтр: кандидат
    от оптимизатора проходит проверку на номинальной модели с консервативным
    профилем возмущений; если предсказанная траектория покидает Ω, кандидат
    заменяется ближайшим допустимым управлением (проекция кандидата на множество
    допустимых управлений). Все срабатывания логируются.

Только stdlib (math): нулевых рантайм-зависимостей это не ломает.
"""

import math


def max_violation(x, g_funcs=(), h_funcs=(), tol=0.0):
    """Мера нарушения Ω = {g_j(x) ≤ 0} ∩ {h_i(x) = 0} в точке x.

    Возвращает max( max_j g_j(x), max_i |h_i(x)| ) − tol, обрезанное к 0:
    0.0 означает «внутри Ω с запасом tol», положительное число — глубину нарушения.
    """
    v = -math.inf
    for g in g_funcs:
        v = max(v, g(x))
    for h in h_funcs:
        v = max(v, abs(h(x)) - 2.0 * tol)
    if v == -math.inf:
        return 0.0
    return max(0.0, v - tol)


def filter_control(step_fn, x0, u_cand, u_lo, u_hi, g_funcs=(), h_funcs=(),
                   horizon=1, d_check=None, n_grid=41, tol=0.0, k0=0):
    """Проекционный safety-фильтр для скалярного управления.

    step_fn(x, u, d, k) -> x_next — номинальная модель (один шаг);
    x0            — текущее измеренное состояние (список);
    u_cand        — кандидат от контроллера;
    [u_lo, u_hi]  — физические пределы привода;
    g_funcs/h_funcs — ограничения допустимости Ω (см. max_violation);
    horizon       — сколько шагов вперёд проверяем;
    d_check(j)    — консервативный (worst-case) профиль возмущения для проверки;
    n_grid        — плотность сетки по управлению;
    tol           — допуск: допустимо всё с нарушением ≤ tol.

    Возвращает (u, info): u — фильтрованное управление; info = {"triggered": bool,
    "cand_violation": float, "out_violation": float, "feasible_found": bool}.
    `triggered` — кандидат был изменён фильтром (событие для лога/графика).
    Если допустимых управлений на сетке нет, возвращается управление с
    минимальным предсказанным нарушением (best effort) и feasible_found=False.
    """
    def violation_of(u):
        x = list(x0)
        worst = 0.0
        for j in range(int(horizon)):
            d = d_check(j) if d_check is not None else None
            x = step_fn(x, u, d, k0 + j)
            worst = max(worst, max_violation(x, g_funcs, h_funcs))
        return worst

    cands = [u_lo + (u_hi - u_lo) * i / (n_grid - 1) for i in range(n_grid)]
    if u_cand not in cands:
        cands.append(min(max(u_cand, u_lo), u_hi))
    viol = [violation_of(u) for u in cands]

    feas = [i for i, v in enumerate(viol) if v <= tol]
    if feas:
        best = min(feas, key=lambda i: abs(cands[i] - u_cand))
        u_out, out_v, found = cands[best], viol[best], True
    else:
        best = min(range(len(cands)), key=lambda i: viol[i])
        u_out, out_v, found = cands[best], viol[best], False

    return u_out, {
        "triggered": u_out != u_cand,
        "cand_violation": violation_of(min(max(u_cand, u_lo), u_hi)),
        "out_violation": out_v,
        "feasible_found": found,
    }
