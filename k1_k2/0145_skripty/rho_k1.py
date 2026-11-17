# к1 0145: сжатие косинуса dropout'ом до нормировки — аналитически ρ = √keep (|f_d| ≈ |f|/√keep, f_d·w несмещён); проверка на синтетике
import numpy as np
r = np.random.default_rng(163); d = 384; keep = 0.2
c = lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
for kind in ("gauss", "relu"):
    f = r.standard_normal((4000, d)); f = np.maximum(f, 0) if kind == "relu" else f
    w = f + r.standard_normal((4000, d)); m = (r.random((4000, d)) < keep) / keep
    print(kind, "ρ̂ =", round(float(np.mean(c(f * m, w)) / np.mean(c(f, w))), 4), " √keep =", round(keep ** 0.5, 4))
for C in (1001, 33):   # потолок p(верный) при s = 16 и косинусе цели ≈ ρ (прочие ≈ 0)
    print(f"C = {C}: потолок p при s = 16, ρ = √0,2: {np.exp(16 * keep ** .5) / (np.exp(16 * keep ** .5) + C - 1):.3f}")
