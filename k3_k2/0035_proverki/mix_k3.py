"""к3: левая асимметрия другой формы — смесь «2 из 10 иначе»: p·N(−a, s²) + (1 − p)·N(a·p/(1 − p), s²), среднее 0
(iid-модель; у нас 2 исполнителя пилота фиксированы, так что это лишь иллюстрация формы). n = 10, t₉ 99 %, 2 000 000 наборов."""
import numpy as np
from scipy import stats
n, N, CH = 10, 2_000_000, 500_000
T9 = stats.t.ppf(0.995, n - 1)
g = np.random.default_rng(5)
for p, s in ((0.2, 0.3), (0.2, 0.1), (0.1, 0.3), (0.3, 0.3)):
    a = 1.0; b = a * p / (1 - p)
    z = np.r_[g.normal(-a, s, 200000)[:int(2e5*p)], g.normal(b, s, 200000)[:int(2e5*(1-p))]]
    sk = stats.skew(z)
    fa = fn = 0
    for _ in range(N // CH):
        m = g.random((CH, n)) < p
        d = np.where(m, -a, b) + s * g.standard_normal((CH, n))
        lo = d.mean(1) - T9 * d.std(1, ddof=1) / np.sqrt(n)
        fh = lo > 0; fa += fh.sum(); fn += (stats.skew(d[fh], axis=1, bias=False) < -1).sum()
    q = fa / N
    print(f"p = {p}, s = {s}: асимметрия формы ≈ {sk:+.2f}; ложное «H» {100*q:.2f} % ± {100*np.sqrt(q*(1-q)/N):.2f}; G1 < −1 среди ложных {100*fn/max(fa,1):.1f} %")
