# Печать g1 (разд. 6): какая оценка асимметрии 10 разностей — g1 = m3/m2^1.5 (bias=True) или G1 (bias=False, как в fix_r2/evidence/onesided/sym_and_shapes.py:41)?
# Синтетика: формы с известной асимметрией, n = 10, 200 000 наборов; печать — среднее оценки и доля наборов с оценкой < −1.
import numpy as np
from scipy import stats
rng = np.random.default_rng(20261112)
N, R = 10, 200_000
shapes = {
    "норм. (γ1 = 0)": lambda s: rng.standard_normal(s),
    "−гамма k=4 (γ1 = −1)": lambda s: -rng.gamma(4.0, size=s),
    "3 − χ²₃ (γ1 = −1,63)": lambda s: 3 - rng.chisquare(3, size=s),
    "−экспон. (γ1 = −2)": lambda s: -rng.exponential(size=s),
}
print(f"n = {N}, наборов {R}; G1 = g1 · √(n(n−1))/(n−2) = g1 · {np.sqrt(N*(N-1))/(N-2):.3f}")
for name, f in shapes.items():
    x = f((R, N))
    g1, G1 = stats.skew(x, axis=1, bias=True), stats.skew(x, axis=1, bias=False)
    print(f"{name:24s} среднее g1 {g1.mean():+.2f}, G1 {G1.mean():+.2f}; доля < −1: g1 {np.mean(g1 < -1):.3f}, G1 {np.mean(G1 < -1):.3f}")
