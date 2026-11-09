English | [Русский](README.md)

[![tests](https://github.com/kostyk348/homeo/actions/workflows/tests.yml/badge.svg)](https://github.com/kostyk348/homeo/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![dependencies](https://img.shields.io/badge/runtime_deps-none-brightgreen.svg)](homeo/engine.py)
[![Release](https://img.shields.io/github/v/release/kostyk348/homeo?sort=semver&color=blue)](https://github.com/kostyk348/homeo/releases)
[![Last commit](https://img.shields.io/github/last-commit/kostyk348/homeo?branch=main)](https://github.com/kostyk348/homeo/commits/main)

# homeo — a homeostatic invariant-based controller

> **Status: research prototype, not a production controller.** This repository
> is a compact research core (balancing competing invariants) plus a
> **validation harness** (audit 2026‑11): a closed loop on a model plant, PID/LQR
> baselines, robustness sweeps, ablations. Invariance and stability are NOT
> proven here: there is no Lyapunov function, no projection onto the admissible
> set Ω (the safety filter is grid-based and scalar-input only), and every
> number is a simulation with fixed seeds. What exactly was and was not
> verified: `docs/formal-setup.md` (problem statement and limitations),
> `docs/validation.md` (protocol and results).

A compact core (137 lines, zero dependencies) for balancing **competing
invariants** in policy space. Ported from C++ (CIDHC, `include/homeo/`).

Instead of "learning behavior", the controller re-balances a set of invariants
for the current environment state on every tick and returns policy coefficients
(all coordinates are normalized to `[−1, 1]` and unfolded into meaningful
quantities: planting shares, sales aggressiveness, hiring pace, etc.).

## Quickstart

```bash
pip install -e .                # core: homeo-controller, stdlib only
pip install -e ".[test]"        # + pytest
python -m pytest tests/ -q      # 132 passed (~30 s)

# validation harness (needs numpy/scipy/matplotlib):
pip install -e ".[experiment]"
python experiments/run_all.py   # ~4–7 min: results/*.json + 4 plots + metrics.json summary
```

Individually: `python experiments/run_baselines.py` (baselines),
`run_robustness.py`, `run_ablation.py`, `run_demo_tanks.py`. Without
installation — just put `homeo/` on your `PYTHONPATH` (the core is pure Python,
only `math` and `random`).

## Two layers: the core and the validation harness

**The core** (`homeo/engine.py`) is a stochastic relaxation in policy space: it
minimizes the energy of competing invariants `E(X) = Σ w_j(X)·|I_j(X)|^p` via
Adam + CLT thermal noise with annealing. On its own it provides no safety
guarantees.

**The validation layer** (`homeo/control.py`, `homeo/safety.py`,
`homeo/metrics.py`, `experiments/`) plugs the core into a closed loop as a
controller:

* plant — two coupled tanks (a linear stand and a nonlinear demo case with
  square-root orifice flows); the admissible set Ω is a pair of level bands,
  Ω = {g_j(x) ≤ 0} (equality constraints are supported by the formalism but
  empty in the scenarios);
* controller invariants evaluate the **consequences of a policy** — the
  predicted level trajectory over a horizon at constant `u`: reference
  tracking, barrier hinges on the bands with a `safety_margin`, control effort;
* an optional **safety filter**: the control candidate is projected onto a grid
  of admissible controls using a worst-case disturbance prediction (every
  activation is logged);
* baselines — an honest PID with anti-windup (tuned on a 90-combination grid
  using a separate seed) and an LQR (DARE + error integrator, never re-tuned);
* all runs are deterministic: fixed seeds, identical disturbance/noise streams
  for every controller.

Headline numbers (strictly from `results/*.json`; protocol and full tables in
`docs/validation.md`):

| scenario | homeo | PID (grid-tuned) | LQR |
|---|---|---|---|
| stand: steps outside Ω out of 480 (mean over 3 seeds) | **0** | 121 | 152.7 |
| stand: margin on the consumer step amplitude | 0.7 | 0 | 0.2 |
| demo (nonlinear plant, biased model): steps outside Ω out of 720 | **0** (11 filter activations) | 363 | 366 |

Honest negative result: in the ablation study on the stand, safety is broken
only by removing the safety invariants (163 violations vs 0 for the full
configuration); thermal noise, annealing, the adaptive gradient step and the
box projection all yield identical metrics — on this class of problems safety
comes from the Ω invariants and/or the filter, not from the stochastic part of
the relaxation. Details and limitations: `docs/validation.md` (§5) and
`docs/formal-setup.md` (§6–7).

## How the core works

Behavior is defined by invariants `I_j(X)` with weights `w_j(X)`:

```
E(X) = Σ_j w_j(X) · |I_j(X)|^p_j      →  min
```

- **The weight may depend on the state or on X itself** (an analytic function:
  sigmoid, clamp, piecewise, power — no neural networks). This is how
  "surrender" transitions work: while the threat is small the goal invariant
  dominates; once it grows, the retreat invariant outweighs it and the
  equilibrium shifts smoothly.
- **Minimization** — gradient descent (Adam) + CLT thermal noise `(12·U−6)`
  with annealing `T_t = T0·2^(−k·t)` (as in `relaxer.hpp`).
- **The numerical gradient** via central differences automatically accounts
  for `dW/dX` — self-weighting: if an invariant's weight depends on X, the
  minimum need not coincide with the invariant's zero, and the gradient finds
  that honestly (see `test_self_weight`).

## API

```python
from homeo import HomeoEngine, make_invariant, make_dynamic_invariant

eng = HomeoEngine(
    center=[0.5, 0.5], range_=[0.5, 0.5],          # each X0 ∈ [0, 1]
    invariants=[
        make_invariant(1.0, 2, lambda X: X[0] - 0.4),          # target 0.4
        make_dynamic_invariant(1.0, 2,                          # surrender:
            lambda X: X[1] - 1.0,                              #   "advance",
            lambda X: 1.0 - threat),                           #   weight falls with threat
    ],
    lr=0.2, max_iter=40, grad_tol=1e-4, t0=0.01, seed=42)

eng.set_raw([0.1, 0.1])   # warm start from the current policy
eng.tick()                # relax until convergence (≤ max_iter steps)
x = eng.raw()             # unfolded policy coefficients
```

### Primitives

| Object | Purpose |
|---|---|
| `Invariant(weight, p, err, wfn)` | invariant: `err: X→float`, `wfn: X→float` (default 1.0) |
| `make_invariant(weight, p, err)` | static weight |
| `make_dynamic_invariant(weight, p, err, wfn)` | weight depends on X / state |
| `HomeoEngine(center, range_, invariants, ...)` | policy relaxation |
| `HomeoEngine.tick()` | one Adam pass to convergence (returns the iteration count) |
| `HomeoEngine.set_raw(raw)` / `.raw()` | normalization `[−1,1]` ↔ unfolding |
| `HomeoEngine.reset()` | resets the Adam moments |

Core ablation flags (audit 2026‑11): `h_doublings` — maximum doublings of the
adaptive step `h` of the numerical gradient (0 disables adaptation); `project`
— clipping the point into the box `[−1,1]^n` after every Adam step (`False` is
for ablations only).

## Tests

```bash
python -m pytest tests/ -v     # 132 tests; the only dependency is pytest
```

Four levels:

* `tests/test_homeo.py` — primitives ported from C++ (`test_bowl.cpp`):
  `bowl` (convergence to a given point), `surrender` (dynamic weights shift the
  equilibrium), `self_weight` (X-dependent weight: the minimum of
  `E = (x−0.5)² + x²·x²` lies at `~3.855`, not at 5.0 — proving `dW/dX` is
  accounted for);
* `tests/test_convergence.py` — convergence as a PROPERTY on a set of systems
  with known answers + explicit out-of-scope checks (see below);
* `tests/test_real_system.py` — a real system: a damped oscillator under a
  harmonic force (reference = closed-form formula AND independent RK4
  integration);
* `tests/test_validation.py` — the validation layer (audit 2026‑11, +20 tests):
  metrics on synthetic trajectories, the safety filter (feasible candidate
  passes unchanged / infeasible one is projected / best-effort when nothing is
  feasible), plant physics (flow directions, actuator saturation), closed-loop
  determinism by seed, the key contrast "homeo — 0 violations vs PID > 50",
  the filter rescuing a controller without safety invariants, the adaptive
  step `h`, thermal noise decay, the core ablation flags, and
  `FixedGainHomeoEngine` = plain gradient descent. Layer tests are stdlib-only
  (CI installs just `.[test]`).

Run: `132 passed in 29.38s` (Python 3.12.16); the core without the validation
layer — `112 passed in 0.26–0.54s` (Python 3.14.7 / 3.9.25).

CI: `.github/workflows/tests.yml` — on every `push` and `pull_request`, matrix
3.9 / 3.11 / 3.12 / 3.13: `pip install ".[test]"` → `pytest tests/ -v` → run
the three examples.

### Convergence: contract and measured numbers

The tests verify a property, not "the function returned something". The
contract:

1. **Root inside the box** `[−1,1]^n` → `tick()` terminates early
   (`iters < max_iter`), and the found point matches the analytic minimum to
   ≤ `1e-3` in the normalized coordinate (0.1% of the box width).
2. **No root inside the box** → the engine does not report convergence:
   `iters == max_iter`, the point rests on the boundary, the residual energy
   is large.
3. **Optimum exactly on the boundary** — a legitimate case: converges early,
   `E = 0` (the boundary is not confused with non-convergence).
4. **`grad_tol` is absolute in units of `E`**: if you scale the weights by λ,
   scale the tolerance too; then neither the point nor the flag changes
   (verified for λ from `1e-4` to `1e4`).

The set this was measured on: 40 random quadratic problems with a known
analytic minimum (2 coordinates, weights 0.5…2, start within 90% of the box),
the same set with thermal noise `t0=0.02`, 8 dynamic-weight problems
(`x* = (1−t)·a + t·b`), a nonlinear problem with weight `w(X)=X²` (the root of
`4x³+2x−1=0` is computed by bisection inside the test), boundary and
out-of-box targets, 5 weight scales.

Worst error in the normalized coordinate (weights O(1), `grad_tol=1e-4`):

| `lr` | quadratics, `t0=0` | same, `t0=0.02` | dynamic weights | `w(X)=X²` | weight scale | worst | max steps |
|---|---|---|---|---|---|---|---|
| 0.010 | 4.1e-04 | 1.0e-04 | 1.6e-04 | 6.8e-05 | 4.7e-05 | 4.1e-04 | 600* |
| 0.015 | 1.1e-04 | 1.1e-04 | 2.2e-04 | 8.5e-05 | 1.4e-05 | 2.2e-04 | 386 |
| 0.020 | 4.9e-04 | 7.8e-04 | 4.6e-04 | 9.4e-05 | 1.8e-05 | 7.8e-04 | 237 |
| **0.030** | **3.5e-04** | **1.5e-04** | **4.8e-04** | **6.5e-05** | **3.5e-05** | **4.8e-04** | **215** |
| 0.050 | 2.6e-04 | 4.5e-04 | 3.7e-03 | 6.1e-04 | 5.5e-05 | 3.7e-03 | 196 |
| 0.100 | 7.9e-04 | 1.5e-04 | 5.7e-03 | 1.0e-03 | 2.5e-05 | 5.7e-03 | 189 |

\* at `lr=0.01` one problem out of 40 fails to reach `grad_tol` within 600
steps: too small a step hurts the convergence FLAG (and time), not accuracy.

The tests fix `lr=0.03`, `grad_tol=1e-4`, `max_iter=600` and a claim of
≤ `1e-3` (the measured maximum over the whole set is `4.8e-4`, ~2× margin).
`lr` is the main accuracy knob — tune it to your requirement (the `lr=0.1`
from the original examples yields up to `5.7e-03`).

**The set was verified by mutations** (breaking the core in a copy of the
repository — the tests must fail; the core set, 112 tests before the
validation upgrade):

| core mutation | run result |
|---|---|
| `lr` zeroed (no learning) | 93 failed / 19 passed |
| `_grad` returns zeros | 93 failed / 19 passed |
| `grad_tol` loosened 100× | 79 failed / 33 passed |
| `wfn(X)` factor dropped from `E` | 12 failed / 100 passed |
| thermal noise amplified 250× | 0 failed / 112 passed — and that is the correct outcome: annealing `2^(−k·t)` eats the noise amplitude within ~15 steps, so the final accuracy is set by `lr` and `grad_tol`, not by the noise amplitude |

## Examples

```bash
python examples/simple_bowl.py      # convergence to a point
python examples/surrender.py        # priority switching under threat
python examples/driven_oscillator.py  # a real system with a reference (below)
```

## A real-system example: the damped oscillator

`examples/driven_oscillator.py` is not a "bowl" demo but a measurable system
with a reference:

```
m·x'' + c·x' + k·x = F0·cos(ω·t)   →   x(t) = A·cos(ω·t − φ)

A = F0 / sqrt((k − m·ω²)² + (c·ω)²)      [m]
φ = atan2(c·ω, k − m·ω²)                 [rad]
```

`A` and `φ` are measurable quantities (amplitude from a displacement sensor and
the phase shift between force and displacement). The core receives neither of
the two formulas: it receives two RESIDUAL EQUATIONS in newtons — substituting
the ansatz `x = A·cos(ωt − φ)` into the equation of motion and equating the
`cos` and `sin` coefficients:

```
r1(A, φ) = (k − m·ω²)·A − F0·cos φ      [N]
r2(A, φ) = −c·ω·A + F0·sin φ            [N]
```

and finds the point where both residuals vanish. The example's reference is
double so that the check does not lean on the same derivation twice: (1) the
closed-form formula above; (2) independent numerical integration of the same
ODE with Runge–Kutta 4 (stdlib only) — from rest to steady state, measuring
`A` and `φ` by projecting `x(t)` onto `cos`/`sin`.

Output of `python examples/driven_oscillator.py` (abridged):

```
== эталон сверен двумя независимыми способами ==  (reference verified two independent ways)
  m=1.0 c=2.0 k=20.0 F0=5.0 w=3.0 | формула A=0.399043 φ=+0.499347 | RK4 A=0.399043 φ=+0.499347 | ΔA/A=1.7e-07 Δφ=1.4e-07
  m=1.2 c=0.9 k=100.0 F0=1.0 w=9.0 | формула A=0.116682 φ=+1.237976 | RK4 A=0.116682 φ=+1.237973 | ΔA/A=2.4e-06 Δφ=2.8e-06
== homeo решает систему по невязкам (никаких формул для A и φ внутри) ==  (homeo solves the system from residuals; no A/φ formulas inside)
  случай 1: m=1.0 c=2.0 k=20.0 F0=5.0 w=3.0 (A* = 0.399043 м, φ* = +0.499347 рад)
    шагов: 168/400 (сошлось), E = 2.60e-09      (steps: 168/400, converged)
    найдено: A = 0.399057 м, φ = +0.499384 рад  (found)
    ошибка против эталона: ΔA/A = 3.51e-05, |Δφ| = 3.70e-05 рад  (error vs reference)
    невязки уравнений движения: r1 = +2.43e-04 Н, r2 = +7.83e-05 Н  (equation residuals)
  ИТОГ: сошлось 4/4 систем; макс. относительная ошибка амплитуды = 1.65e-04;  (TOTAL: 4/4 systems converged; max relative amplitude error; max phase error; max residual)
        макс. ошибка фазы = 2.25e-04 рад; макс. невязка = 2.76e-03 Н
== честное поведение вне области применимости: ящик уже, чем решение ==  (honest out-of-scope behavior: the box is narrower than the solution)
  A* = 0.399043 м, а ящик допускает A ≤ 0.239426 м
    шагов: 300/300 → tick() вернул max_iter: сходимости НЕТ (это и есть признак)  (tick() returned max_iter: NO convergence — that is the flag)
    невязки не занулились: r1 = -1.756 Н, r2 = +0.958 Н, E = 0.160
ПРОВЕРКА ПРИМЕРА: PASS  (EXAMPLE CHECK: PASS)
```

(The example's console output is in Russian; the numbers are what matters here.)
The same numbers are checked by tests (`tests/test_real_system.py`): the
reference agrees with RK4 (≤1e-5), on 8 random configurations (0.15 ≤ ζ ≤ 0.6)
the amplitude is found with relative error ≤ 1e-3, the phase ≤ 1e-3 rad,
residuals ≤ 1e-2 N, and a box narrower than the solution yields an honest
"did not converge".

## Scope of applicability (honest limitations)

1. **The solution must lie inside the box** `center ± range`. If it does not,
   the engine does not converge and reports this in exactly one way: `tick()`
   returns `max_iter`. Telling it from a legitimate boundary optimum is simple:
   look at `energy()` — on the boundary it is zero for an attainable optimum
   and large for an unattainable one (for the oscillator with a box
   `A ≤ 0.3·A*`: `iters = 300/300`, `E = 0.16`, 40% error).
2. **`grad_tol` is absolute in units of `E`** — a trap when working with
   weights. Measurement: weights `1e-3` with `grad_tol = 1e-4` "converge" in 36
   steps but 2.3% away from the analytic minimum (`raw = 2.773` instead of
   `3.0`). Normalize the weights to O(1) (as in the examples) or scale
   `grad_tol` with the weights — then neither the point nor the flag depends on
   λ at all (verified for λ from `1e-4` to `1e4`).
3. **The point depends on the step `lr`** (table in "Convergence"). "Converged"
   means "within the achieved accuracy", not "frozen": a second `tick()` moves
   the point by about `2e-4` (in raw) at `lr=0.03` and `2e-3` at `lr=0.1` with
   noise `t0=0.02`.
4. **The method is local.** Adam + thermal noise: with a non-convex sum of
   invariants the global minimum is not guaranteed. All test guarantees apply
   to problems where the minimum is unique or known.
5. **Cost.** The gradient uses central differences: `2n` evaluations of `E` per
   step, each `O(m)` in the number of invariants; `tick()` runs up to
   `max_iter` steps. Measurement: `n=2`, `m=2` → 153 steps in 0.66 ms
   (~1.1 µs per `E` evaluation). For large `n` an analytic gradient would be
   cheaper — the core neither requires nor provides one.

For the closed loop (validation harness) the limitations are stricter:
invariance with respect to Ω is not guaranteed at all (see §7 in
`docs/formal-setup.md` — no Lyapunov function, no projection onto Ω, the
filter is grid-based and scalar-input, prediction at constant `u`).

## Battle use

The core is used in an agent for Kaggriculture (kaggle): `HomeoPolicy` balances
6 coefficients (wheat/melon planting density, hand hiring, sales aggressiveness,
goose/cow investments) with season-phase invariants:
`plant_rate → target_plant(t)`, `sell_eagerness → shed pressure`,
`hire_rate → workforce demand`, etc. — the agent's average income is ~21k per
seed (benchmark v4 — 13.7k). This has nothing to do with a physical closed
loop: the validation harness in this repository is a simulation.

## Documentation

* `docs/formal-setup.md` — the formal statement: state x, admissible set Ω,
  what the core actually does, barrier hinges, the safety filter, the research
  question "stochastic relaxation vs guaranteed invariance", honest limitations;
* `docs/validation.md` — the validation protocol, environment, tables and
  plots strictly from `results/*.json` (baselines, robustness, ablations, demo);
* `results/*.json` — machine-readable results of the latest run
  (`metrics.json` — the summary with the environment block);
* `CHANGELOG.md` — change history.

## Origin

The core is a port of the C++ implementations `include/homeo/*` of the CIDHC
project (a homeostatic controller for problems with conflicting goals).

## Provenance

The snapshot that added the convergence tests, CI, the real-system example and
packaging grew out of the **`homeo`** entry in the "Portfolio of kostyk348"
digest of the Museum of Lost Ideas (data: `data/portfolio-review.json`, digest:
<https://kostyk348.github.io/lost-tech-museum/digests/portfolio-review/>).

* The entry's pitch: "Publish as a small library + a note 'invariants instead
  of metrics' with two examples (a scheduler and memory limits). A cheap way to
  stake the idea and gain an audience — conference talks grow from such notes."
* What the audit said was missing: "5 sources, 1 test, no releases, no paper
  with a formal description and convergence analysis, no examples on real
  systems (demos only)" — this snapshot closes the convergence analysis
  (112 tests + mutation testing), the real-system example, CI and the install
  check. Releases, the paper and publication remain with the owner.
* The validation upgrade (audit 2026‑11) adds what was missing next: a formal
  statement, a closed loop with baselines, and an honest account of which of
  the core's claimed behaviors actually reproduce (see `docs/validation.md`).
