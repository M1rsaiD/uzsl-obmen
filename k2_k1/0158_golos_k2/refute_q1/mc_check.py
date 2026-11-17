"""Независимый пересчёт α_k и мощности Э2 (правило П22.4 (б), жадная схема 22.3) — проверка голоса к2, ЦП.
σ ячеек — из risk_e2_mc.out (стр. 2 и 12); всё остальное — свой код. Дополнительно: мощность на позициях 6–7 (место R3),
порядок «рычаги 0059 первыми», точка безубыточности по априорной доле рабочих рычагов."""
import numpy as np, sys
THR, MINB = 1.0081957667939179, 6
SIG = {"base": [2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09],
       "sohn": [2.20, 1.98, 2.35, 2.38, 2.28, 1.05, 1.08, 1.11, 1.14, 1.65]}
N = 400_000

def acc(d):
    return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)

def chain(sig, mus, rng, n=N):
    """mus — истинные эффекты рычагов по порядку; возвращает доли принятия каждого рычага и P(≥1 ложного)."""
    sig = np.asarray(sig)
    cur = rng.standard_normal((n, 10)) * sig          # контроль, затравка 48
    curmu = 0.0
    take = np.zeros(len(mus)); anyfalse = np.zeros(n, bool); nfalse = np.zeros(n)
    for j, mu in enumerate(mus):
        e = rng.standard_normal((n, 10)) * sig + mu
        ok = acc(e - cur)
        take[j] = ok.mean()
        if mu == 0:
            anyfalse |= ok; nfalse += ok
        cur = np.where(ok[:, None], e, cur)
    return take, anyfalse.mean(), nfalse.mean()

rng = np.random.default_rng(20261116)
assert abs(np.sqrt(2 * np.sum(np.square(SIG["base"]))) / 10 - 1.064) < 0.002   # СКО Δ̄ = 1,064 (risk_e2_mc.out:3)
for name, sig in SIG.items():
    print(f"== шум {name}")
    for k in (1, 5, 7, 8, 13):
        _, a, ef = chain(sig, [0.0] * k, rng)
        print(f"  все нулевые, k={k:2d}: α_k {a:.3f}  E[ложных] {ef:.2f}")
    for mu in (1.0, 1.5, 2.0):
        pw = []
        for pos in (1, 5, 6, 7, 8):
            t, _, _ = chain(sig, [0.0] * (pos - 1) + [mu], rng)
            pw.append(f"поз.{pos} {t[-1]:.3f}")
        print(f"  мощность μ=+{mu}: " + " ".join(pw))
