"""к3 0037: уточнение максимума у смеси при заданной асимметрии (шаг p 0,002 около максимума, 4 000 000 наборов);
плюс доля строки G1 < −1 среди ложных «H» в точке максимума и нижний край при γ = −0,7."""
import numpy as np
from scipy import stats
from max_pri_asimm_k3 import s_for, skew_exact, two_pt
n = 10; T9 = stats.t.ppf(0.995, n - 1); g = np.random.default_rng(371)
def rate(p, s, N=4_000_000, CH=400_000):
    h = f1 = 0
    for _ in range(N // CH):
        d = np.where(g.random((CH, n)) < p, -1.0, p / (1 - p)) + s * g.standard_normal((CH, n))
        H = (d.mean(1) - T9 * d.std(1, ddof=1) / np.sqrt(n)) > 0
        h += H.sum(); f1 += (stats.skew(d[H], axis=1, bias=False) < -1).sum()
    q = h / N; return 100 * q, 100 * np.sqrt(q * (1 - q) / N), 100 * f1 / max(h, 1)
for gam, ps in ((-1.4, np.arange(0.180, 0.2001, 0.002)), (-1.41, np.arange(0.180, 0.2001, 0.002)),
                (-1.58, np.arange(0.160, 0.1801, 0.002)), (-0.7, (0.05, 0.10, 0.29))):
    print(f"\nγ = {gam}:  p       s/выигрыш   ложное «H», %     G1 < −1 среди ложных, %")
    for p in np.round(ps, 3):
        if two_pt(p) > gam: continue
        s = s_for(p, gam); assert abs(skew_exact(p, s) - gam) < 1e-9
        r, se, f = rate(p, s)
        print(f"           {p:.3f}   {s/(p/(1-p)):5.2f}      {r:6.2f} ± {se:.2f}       {f:5.1f}")
