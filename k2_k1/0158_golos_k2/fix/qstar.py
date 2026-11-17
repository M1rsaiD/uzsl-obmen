"""q* и ожидаемые истинные/ложные принятия для R3 {3, 6} на местах 6-7 после 5 нулевых (а′); аддитивно
(кандидат = текущая + μ). Шум macro (база, Sohn) — как refute_q1/mc_tradeoff.py; micro — σ macro / множитель
(refute_q1/mc_micro.py). Правило — черновик 22.4 (Δ̄ ≥ THR, ≥ 6/10). ЦП."""
import numpy as np
THR, MINB, N = 1.0081957667939179, 6, 300_000
base = np.array([2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09])
mult = np.array([2.15, 2.05, 2.58, 2.54, 2.23, 1.21, 1.05, 1.13, 1.16, 1.41])
SIG = {"macro база": base, "macro Sohn": np.array([2.20, 1.98, 2.35, 2.38, 2.28, 1.05, 1.08, 1.11, 1.14, 1.65]),
       "micro": base / mult}
def acc(d): return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)
def chain(sig, mus, rng):
    cur = rng.standard_normal((N, 10)) * sig; m = np.zeros(N); take = []
    for mu in mus:
        e = rng.standard_normal((N, 10)) * sig + (m + mu)[:, None]
        ok = acc(e - cur); take.append(ok.mean()); cur = np.where(ok[:, None], e, cur); m = np.where(ok, m + mu, m)
    return np.array(take)
rng = np.random.default_rng(16)
q3, q6 = 0.10, 0.05
for name, s in SIG.items():
    z = chain(s, [0] * 7, rng)
    for mu in (1.0, 1.5):
        p6 = chain(s, [0] * 5 + [mu, 0], rng)[5]; p7 = chain(s, [0] * 6 + [mu], rng)[6]
        qs6, qs7 = z[5] / (z[5] + p6), z[6] / (z[6] + p7)
        tru = q3 * p6 + q6 * p7; fal = (1 - q3) * z[5] + (1 - q6) * z[6]
        print(f"{name:10s} μ=+{mu}: P0 поз.6/7 {z[5]:.3f}/{z[6]:.3f}; мощн. {p6:.3f}/{p7:.3f}; q* {qs6:.3f}/{qs7:.3f}; "
              f"при q3={q3}, q6={q6}: E[истинных] {tru:.3f}, E[ложных] {fal:.3f}")
