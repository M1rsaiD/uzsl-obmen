# к1 0141: α_k (хотя бы один пустой рычаг из k принят) при правиле 22.4 (Δ̄ ≥ 1,0082 и > 0 у ≥ 6/10 ячеек), общий контроль.
# σ ячеек — из mc_micro.py к2 (строки 5–7): macro (биномиальная модель) и micro = macro / множитель. Свой генератор и код.
import numpy as np
THR = 1.0081957; macro = np.array([2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09])
mult = np.array([2.15, 2.05, 2.58, 2.54, 2.23, 1.21, 1.05, 1.13, 1.16, 1.41])
def alpha(sig, ks=(5, 8, 13), N=300_000, seed=141):
    r = np.random.default_rng(seed); e0 = r.standard_normal((N, 10)) * sig; first = np.full(N, 99)
    for j in range(max(ks)):
        d = r.standard_normal((N, 10)) * sig - e0
        a = (d.mean(1) >= THR) & ((d > 0).sum(1) >= 6)
        first = np.where((first == 99) & a, j, first)
    return {k: float(np.mean(first < k)) for k in ks}
for name, s in (("macro", macro), ("micro", macro / mult)):
    print(name, "СКО Δ̄ =", round(float(np.sqrt(2 * np.sum(s ** 2)) / 10), 3), {k: round(v, 3) for k, v in alpha(s).items()})
