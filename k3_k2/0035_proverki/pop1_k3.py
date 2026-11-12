"""к3, сверка «Поправки 1»: ложное «H установлена» по t₉ 99 % (= правило «оба», 0034 разд. 2.2 и 0146: перцентильный
уже t₉) при нулевом среднем, n = 10, 4 000 000 наборов на форму; доля G1 < −1 среди ложных. Сверх к2 — симметричные формы
с лёгкими хвостами и двугорбые (разности исполнителей могут быть двугорбыми: пилот против остальных)."""
import numpy as np
from scipy import stats
n, N, CH = 10, 4_000_000, 500_000
T9 = stats.t.ppf(0.995, n - 1)
g = np.random.default_rng(20261112)
def chi_left(k):  return lambda m: -(g.chisquare(k, (m, n)) - k) / np.sqrt(2 * k)
shapes = {
    "норм. (0)": lambda m: g.standard_normal((m, n)),
    "t₅ (0)": lambda m: g.standard_t(5, (m, n)),
    "t₃ (0)": lambda m: g.standard_t(3, (m, n)),
    "равномерная (0, лёгкие хвосты)": lambda m: g.uniform(-1, 1, (m, n)),
    "двугорбая N(±1; 0,3²) (0)": lambda m: g.choice([-1.0, 1.0], (m, n)) + 0.3 * g.standard_normal((m, n)),
    "двугорбая N(±1; 0,1²) (0)": lambda m: g.choice([-1.0, 1.0], (m, n)) + 0.1 * g.standard_normal((m, n)),
    "−1 (χ²₈)": chi_left(8), "−2 (χ²₂)": chi_left(2),
}
for name, f in shapes.items():
    fa = fn = 0
    for _ in range(N // CH):
        d = f(CH)
        lo = d.mean(1) - T9 * d.std(1, ddof=1) / np.sqrt(n)
        fh = lo > 0
        fa += fh.sum()
        fn += (stats.skew(d[fh], axis=1, bias=False) < -1).sum()
    p = fa / N
    print(f"{name:32s} ложное «H» {100*p:.3f} % ± {100*np.sqrt(p*(1-p)/N):.3f}; G1 < −1 среди ложных {100*fn/max(fa,1):.1f} %")
