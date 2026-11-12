"""к3: перекрёстная проверка к1 0135 (свод 4.7) — «оба 99 %» = t₉ в точке максимума смеси (−1,4: p = 0,19, s из асимметрии).
Свой код, затравка 1350; 20 000 наборов, перцентильный бутстреп 2 000 (нижняя граница — 0,5-й процентиль средних)."""
import numpy as np
from scipy import stats
from max_pri_asimm_k3 import s_for
n, N, B, CH = 10, 20_000, 2_000, 500
T9 = stats.t.ppf(0.995, n - 1); g = np.random.default_rng(1350)
p = 0.19; s = s_for(p, -1.4)
t9 = pc = both = t_only = p_only = 0
for _ in range(N // CH):
    d = np.where(g.random((CH, n)) < p, -1.0, p / (1 - p)) + s * g.standard_normal((CH, n))
    T = (d.mean(1) - T9 * d.std(1, ddof=1) / np.sqrt(n)) > 0
    idx = g.integers(0, n, (CH, B, n))
    P = np.quantile(np.take_along_axis(d[:, None, :].repeat(B, 1), idx, 2).mean(2), 0.005, axis=1) > 0
    t9 += T.sum(); pc += P.sum(); both += (T & P).sum(); t_only += (T & ~P).sum(); p_only += (P & ~T).sum()
se = 100 * np.sqrt(0.116 * 0.884 / N)
print(f"p = {p}, s = {s:.3f}: t₉ {100*t9/N:.2f} %; перцентильная {100*pc/N:.2f} %; оба {100*both/N:.2f} % (СКО ≈ {se:.2f})")
print(f"t₉ > 0, перцентильная — нет: {t_only};  перцентильная > 0, t₉ — нет: {p_only}")
