"""к3: ложное «H установлена» по правилу «оба» (перцентильный 99 % бутстреп + t₉ 99 %, нижние границы > 0) при нулевом
среднем и скошенных формах; выборочная асимметрия g1 (bias=True) и G1 (bias=False) при n = 10; и главное — видна ли
скошенность в тех наборах, где «H установлена» ложно. Свой код (numpy/scipy), формы −(χ²_k − k)/√(2k)."""
import numpy as np
from scipy import stats

n, N, B = 10, 40000, 2000
T9 = stats.t.ppf(0.995, n - 1)
rng = np.random.default_rng(20261112)
shapes = {"норм.": None, "−1,0 (k=8)": 8, "−1,41 (k=4)": 4, "−1,63 (k=3)": 3, "−2,0 (k=2)": 2}
print(f"t₉ 0,995 = {T9:.4f}; N = {N} наборов × n = {n}; бутстреп B = {B}")
for name, k in shapes.items():
    d = rng.standard_normal((N, n)) if k is None else -(rng.chisquare(k, (N, n)) - k) / np.sqrt(2 * k)
    m, s = d.mean(1), d.std(1, ddof=1)
    t_lo = m - T9 * s / np.sqrt(n)
    p_lo = np.empty(N)
    for i in range(0, N, 1000):
        blk = d[i:i + 1000]
        idx = rng.integers(0, n, (len(blk), B, n))
        bm = np.take_along_axis(blk[:, None, :].repeat(B, 1), idx, 2).mean(2)
        p_lo[i:i + 1000] = np.quantile(bm, 0.005, axis=1)
    both = (t_lo > 0) & (p_lo > 0)
    g1 = stats.skew(d, axis=1, bias=True); G1 = stats.skew(d, axis=1, bias=False)
    print(f"{name:12s} ложное «H уст.» {both.mean()*100:5.2f} % (t₉ {np.mean(t_lo>0)*100:5.2f}, перц. {np.mean(p_lo>0)*100:5.2f});"
          f" среднее g1 {g1.mean():+.2f}, G1 {G1.mean():+.2f}; P(G1<−1) {np.mean(G1<-1)*100:4.1f} % (g1: {np.mean(g1<-1)*100:4.1f} %);"
          f" среди ложных «H уст.»: P(G1<−1) {np.mean(G1[both]<-1)*100 if both.any() else float('nan'):4.1f} %, медиана G1 {np.median(G1[both]) if both.any() else float('nan'):+.2f}")
