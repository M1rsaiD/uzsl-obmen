"""к3 0037: у смеси p·N(−1, s²) + (1 − p)·N(p/(1 − p), s²) при ЗАДАННОЙ асимметрии γ — какова наибольшая доля ложного «H»
(t₉ 99 %, n = 10)? При s > 0 ту же γ даёт меньшее p, а (1 − p)¹⁰ при меньшем p больше. Для каждой γ: сетка p, s из γ точно."""
import numpy as np
from scipy import stats
n = 10; T9 = stats.t.ppf(0.995, n - 1); g = np.random.default_rng(37)
def two_pt(p): return (2 * p - 1) / np.sqrt(p * (1 - p))           # асимметрия при s → 0
def s_for(p, gam):                                                   # s, при котором асимметрия смеси = gam
    V2 = p / (1 - p); M3 = p * (2 * p - 1) / (1 - p) ** 2
    s2 = (M3 / gam) ** (2 / 3) - V2
    return np.sqrt(s2) if s2 >= 0 else None
def skew_exact(p, s):
    b = p / (1 - p); return (-p + (1 - p) * b ** 3) / (p + (1 - p) * b * b + s * s) ** 1.5
def rate(p, s, N=2_000_000, CH=400_000):
    h = 0
    for _ in range(N // CH):
        d = np.where(g.random((CH, n)) < p, -1.0, p / (1 - p)) + s * g.standard_normal((CH, n))
        h += ((d.mean(1) - T9 * d.std(1, ddof=1) / np.sqrt(n)) > 0).sum()
    q = h / N; return 100 * q, 100 * np.sqrt(q * (1 - q) / N)
assert abs(skew_exact(0.2, 0.1) + 1.414) < 1e-3 and abs(skew_exact(0.1, 0.3) + 1.095) < 1e-3
if __name__ == "__main__":
  for gam in (-0.7, -1.0, -1.41, -1.58):
    print(f"\nγ = {gam}:   p      s     s/выигрыш   ложное «H», %    (1 − p)¹⁰, %")
    best = (0, None)
    for p in np.round(np.arange(0.05, 0.40, 0.01), 2):
        if two_pt(p) > gam: continue                                   # при этом p такую γ не получить
        s = s_for(p, gam); assert abs(skew_exact(p, s) - gam) < 1e-9
        r, se = rate(p, s)
        print(f"          {p:.2f}  {s:.3f}   {s/(p/(1-p)):6.2f}      {r:6.2f} ± {se:.2f}     {100*(1-p)**n:6.2f}")
        if r > best[0]: best = (r, p)
    print(f"  наибольшая: {best[0]:.2f} % при p = {best[1]}")
