"""Если 22.4 считать по micro (записи валидации), а не macro по 15 исполнителям: σ ячейки = σ macro / множитель
(summary_table.out:29–30). Тот же симулятор (жадно, рычаги нулевые, мощность на позиции после нулевых). ЦП."""
import numpy as np
THR, MINB = 1.0081957667939179, 6
macro = np.array([2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09])
mult = np.array([2.15, 2.05, 2.58, 2.54, 2.23, 1.21, 1.05, 1.13, 1.16, 1.41])
sig = macro / mult
print("σ micro по ячейкам:", " ".join(f"{x:.2f}" for x in sig), "| СКО Δ̄", round(float(np.sqrt(2 * np.sum(sig ** 2)) / 10), 3))
N = 400_000
def acc(d): return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)
def chain(mus, rng):
    cur = rng.standard_normal((N, 10)) * sig; anyf = np.zeros(N, bool); take = []
    for mu in mus:
        e = rng.standard_normal((N, 10)) * sig + mu; ok = acc(e - cur); take.append(ok.mean())
        if mu == 0: anyf |= ok
        cur = np.where(ok[:, None], e, cur)
    return np.array(take), anyf.mean()
rng = np.random.default_rng(3)
for k in (5, 7, 8, 13):
    print(f"k={k}: α_k {chain([0]*k, rng)[1]:.3f}")
for mu in (1.0, 1.5):
    print(f"μ=+{mu}: поз.1 {chain([mu], rng)[0][-1]:.3f} поз.6 {chain([0]*5+[mu], rng)[0][-1]:.3f}")
