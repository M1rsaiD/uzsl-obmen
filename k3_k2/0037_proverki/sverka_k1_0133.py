"""к3: перекрёстная проверка к1 0133 (свод 4.7) — нижний край при p = 0,01 и 0,02; свой код, затравка 133."""
import numpy as np
import max_pri_asimm_k3 as m
m.g = np.random.default_rng(133)
for gam in (-0.7, -1.4):
    for p, s_k1 in ((0.01, {-0.7: 0.221, -1.4: 0.164}[gam]), (0.02, {-0.7: 0.270, -1.4: 0.196}[gam])):
        s = m.s_for(p, gam); r, se = m.rate(p, s)
        print(f"γ = {gam}, p = {p}: s {s:.3f} (у к1 {s_k1}); ложное «H» {r:.2f} ± {se:.2f} %")
