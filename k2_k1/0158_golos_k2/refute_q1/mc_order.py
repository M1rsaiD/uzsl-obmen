"""Аддитивная жадная цепочка (кандидат = текущая + рычаг): мощность двух рычагов 0059 на местах 1–2 против 6–7. ЦП."""
import numpy as np
THR, MINB = 1.0081957667939179, 6
SIG = {"base": [2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09],
       "sohn": [2.20, 1.98, 2.35, 2.38, 2.28, 1.05, 1.08, 1.11, 1.14, 1.65]}
N = 400_000
def acc(d): return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)
def chain(sig, mus, rng, n=N):
    sig = np.asarray(sig); cur = rng.standard_normal((n, 10)) * sig; curmu = np.zeros((n, 1)); take = []
    for mu in mus:
        e = rng.standard_normal((n, 10)) * sig + curmu + mu
        ok = acc(e - cur); take.append(ok.mean())
        cur = np.where(ok[:, None], e, cur); curmu = np.where(ok[:, None], curmu + mu, curmu)
    return np.array(take)
rng = np.random.default_rng(11)
for name, sig in SIG.items():
    for mu in (1.0, 1.5):
        a = chain(sig, [mu, mu] + [0] * 5, rng); b = chain(sig, [0] * 5 + [mu, mu], rng)
        print(f"{name} μ=+{mu}: два рычага на местах 1–2: {a[0]:.3f}/{a[1]:.3f} (сумма {a[0]+a[1]:.2f}); на местах 6–7: {b[5]:.3f}/{b[6]:.3f} (сумма {b[5]+b[6]:.2f})")
