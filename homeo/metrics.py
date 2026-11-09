"""Метрики валидации контроллеров на синтетических траекториях.

Все функции чистые (списки float, только stdlib) — используются экспериментами
в experiments/ и напрямую тестируются в tests/test_metrics.py.

Определения (зафиксированы в docs/validation.md):

  * нарушение      : depth = max(0, lo − v, v − hi); ограничение нарушено,
                     если depth > tol (по умолчанию tol = 0 — любое);
  * violation_count: число шагов k, на которых нарушено ХОТЯ БЫ одно ограничение;
  * max_violation_depth: максимальная глубина нарушения по эпизоду;
  * overshoot      : max(0, max_k v_k − ref) — выброс над уставкой;
  * settling_time  : первое t = k·dt, начиная с которого |v − ref| ≤ tol ДО КОНЦА
                     окна (окно — весь эпизод или его часть); None, если ни разу;
  * recovery_time  : время от начала возмущения (k_onset) до первого момента,
                     когда состояние вернулось в допустимое множество и остаётся
                     в нём до конца эпизода; 0.0, если нарушений после onset не
                     было; None, если так и не вернулось;
  * control_effort : Σ_k |u_k|·dt — интеграл модуля управления.
"""

import math


def violation_depth(v, lo, hi):
    """Глубина выхода значения v за полосу [lo, hi]: 0 внутри полосы."""
    return max(0.0, lo - v, v - hi)


def violation_count(constraints, tol=0.0):
    """constraints: список пар (series, (lo, hi)). Шаг считается нарушенным,
    если хотя бы одна полоса нарушена с глубиной > tol."""
    if not constraints:
        return 0
    n = min(len(s) for s, _ in constraints)
    cnt = 0
    for k in range(n):
        for s, (lo, hi) in constraints:
            if violation_depth(s[k], lo, hi) > tol:
                cnt += 1
                break
    return cnt


def max_violation_depth(constraints):
    """Максимальная глубина нарушения по всем ограничениям и всем шагам."""
    worst = 0.0
    for s, (lo, hi) in constraints:
        for v in s:
            worst = max(worst, violation_depth(v, lo, hi))
    return worst


def overshoot(series, ref):
    """Максимальный выброс над уставкой ref (только сверху): max(0, max_k v_k − ref)."""
    return max(0.0, max(series) - ref)


def settling_time(series, ref, tol, dt, k_from=0, k_to=None):
    """Первое время входа в трубку |v − ref| ≤ tol и staying до k_to (по умолчанию — до конца).

    Возвращает время в тех же единицах, что dt (t = k·dt), или None.
    """
    n = len(series) if k_to is None else min(k_to, len(series))
    for k in range(int(k_from), n):
        if all(abs(series[j] - ref) <= tol for j in range(k, n)):
            return k * dt
    return None


def recovery_time(constraints, k_onset, dt, k_end=None):
    """Время восстановления допустимости после возмущения в шаге k_onset.

    0.0 — если после onset нарушений нет; None — если до конца эпизода
    допустимость не восстановлена. Иначе время от onset до первого момента,
    после которого нарушений больше нет до k_end (по умолчанию — конец).
    """
    n = min(len(s) for s, _ in constraints) if constraints else 0
    if k_end is None:
        k_end = n
    else:
        k_end = min(k_end, n)
    k_onset = max(0, min(int(k_onset), k_end))

    def violated(k):
        return any(violation_depth(s[k], lo, hi) > 0.0 for s, (lo, hi) in constraints)

    if not any(violated(k) for k in range(k_onset, k_end)):
        return 0.0
    for k in range(k_onset, k_end):
        if not any(violated(j) for j in range(k, k_end)):
            return (k - k_onset) * dt
    return None


def control_effort(u, dt):
    """Σ |u_k|·dt — интеграл модуля управления (управляющее усилие)."""
    return sum(abs(v) for v in u) * dt


def zero_violation_margin(amplitudes, violations):
    """Робастностный запас по развёртке амплитуд возмущения.

    amplitudes — проверенные амплитуды, violations — violation_count для каждой.
    Возвращает {"margin": a* | None, "first_violating": a | None}:
    margin = max{a : viol(a) == 0}; first_violating = min{a : viol(a) > 0}.
    margin — не монотонная оценка: это максимальная ПРОВЕРЕННАЯ амплитуда без
    нарушений (сетка задаётся экспериментом).
    """
    margin = None
    first_violating = None
    for a, c in zip(amplitudes, violations):
        if c == 0:
            margin = a if margin is None else max(margin, a)
        else:
            first_violating = a if first_violating is None else min(a, first_violating)
    return {"margin": margin, "first_violating": first_violating}
