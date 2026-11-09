# Changelog

Все заметные изменения в этом проекте документируются в этом файле.

Формат — [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версионирование — [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Тегов пока нет: записи ниже собраны из `git log` (первый коммит `f6e37e6`, 2026-08-11).
Владелец может превратить раздел `Unreleased` в `v0.1.0` при постановке первого тега.

## [Unreleased]

### Added

- **Валидационный слой** (`homeo/control.py`, `homeo/metrics.py`, `homeo/safety.py`,
  `experiments/`, `tests/test_validation.py` — всего 132 теста):
  - объект «два связанных резервуара» (`TwoTankPlant`): линейный стенд и демо‑кейс
    с квадратичными перетоками; номинальная модель контроллера может быть смещена
    относительно объекта;
  - `HomeoTankController`: инварианты оценивают последствия политики
    (предсказанную траекторию уровней на горизонте при постоянном `u`),
    барьерные шарниры полос с запасом `safety_margin`, оценка расхода
    потребителя `d_hat`; `FixedGainHomeoEngine` — абляция «без Adam»;
  - `PIDTankController` — честный ПИД с anti‑windup (clamping);
  - `run_episode` — заранее сгенерированные потоки возмущений и шума
    (фиксированные seed, одинаковые для всех контроллеров — детерминизм и
    честное сравнение);
  - `max_violation` — мера нарушения Ω = {g_j(x) ≤ 0, h_i(x) = 0};
    `filter_control` — сеточный проекционный safety‑фильтр одномерного
    управления с worst‑case возмущением и логированием срабатываний;
  - метрики: `violation_count`, `max_violation_depth`, `overshoot`,
    `settling_time`, `recovery_time`, `control_effort`,
    `zero_violation_margin` (определения — `docs/validation.md`).
- **Эксперименты** (`experiments/`, extra `[experiment]` = numpy/scipy/matplotlib):
  `run_baselines.py` (homeo vs ПИД, настроенный сеткой 90 комбинаций на отдельном
  seed, vs LQR (DARE с интегратором ошибки)), `run_robustness.py` (развёртка
  амплитуды ступени потребителя, шума датчиков, дрейф c2), `run_ablation.py`
  (7 абляций ядра и контроллера), `run_demo_tanks.py` (нелинейный объект со
  смещённой моделью), `run_all.py`; результаты — `results/*.json` + 4 плота.
- **Документация**: `docs/formal-setup.md` (формальная постановка: x, Ω, ядро,
  барьерные шарниры, safety‑фильтр, исследовательский вопрос, ограничения) и
  `docs/validation.md` (протокол, таблицы и плоты строго из `results/*.json`);
  `README.en.md` — полное английское зеркало README с переключателем языка.
- **Ядро `homeo`** (`homeo/engine.py`, 137 строк, зависимости — только `math` и `random`) (`f6e37e6`):
  - инварианты `I_j(X)` с весами `w_j(X)`, минимизация `E(X) = Σ_j w_j(X)·|I_j(X)|^p_j`;
  - градиентный спуск Adam + термальный шум CLT `(12·U−6)` с annealing `T_t = T0·2^(−k·t)`
    (порт `relaxer.hpp` из C++ CIDHC);
  - численный градиент центральными разностями, поэтому `dW/dX` учитывается честно
    (минимум при весе, зависящем от `X`, не обязан совпадать с нулём инварианта).
- **Примитивы** (`f6e37e6`): `HomeoEngine`, `Invariant`, `make_invariant`,
  `make_dynamic_invariant`, `tick()`, `set_raw()/raw()`, `reset()`; нормализация `[−1, 1]` ↔
  развёртка в осмысленные коэффициенты политики.
- **Пакет `homeo-controller`**: `pyproject.toml`, extra `[test]`, без рантайм-зависимостей (`f6e37e6`).
- **Примеры** `examples/simple_bowl.py`, `examples/surrender.py` (`f6e37e6`) и
  `examples/driven_oscillator.py` — реальная система: демпфированный осциллятор под гармонической
  силой с эталоном из замкнутой формулы и независимого интегрирования RK4 (`b72d31c`).
- **112 тестов** на трёх уровнях (`b72d31c`, `tests/conftest.py`, `tests/test_convergence.py`,
  `tests/test_real_system.py`, расширенный `tests/test_homeo.py`):
  - примитивы, перенесённые из C++ (`test_bowl.cpp`): `bowl`, `surrender`, `self_weight`;
  - сходимость как **свойство**: 40 случайных квадратичных задач с аналитическим минимумом,
    динамические веса, `w(X)=X²`, границы и запредельные цели, 5 масштабов весов;
  - выход за границы применимости: нет корня внутри ящика → движок не рапортует о сходимости
    (`iters == max_iter`).
- **CI** `.github/workflows/tests.yml` (`b72d31c`): Python 3.9 / 3.11 / 3.13,
  `pip install ".[test]"` → `pytest tests/ -v` → прогон трёх примеров.
- **README** (`b72d31c`): контракт сходимости, таблица измеренных ошибок по `lr`, раздел про
  границы применимости, раздел про CI.
- `LICENSE` (MIT) и `.gitignore` (`f6e37e6`).
