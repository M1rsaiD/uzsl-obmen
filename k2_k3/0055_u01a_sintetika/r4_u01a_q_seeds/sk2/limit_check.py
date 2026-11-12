"""Предел q при сжатии ошибок «место» к порогу: та же make, но 1000 классов (больше пар у порога на реализацию), 10 затравок.
Плюс 200 классов × 200 затравок (пул) — для узких окон h. Только синтетика, ЦП."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/r4_round4_u01a_sk2")
from indep_check import pairs, G
HS = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02)
for n_cls, seeds in ((1000, range(10)), (200, range(200))):
    for cv in (0.05, 0.10):
        pool = {h: [0, 0] for h in HS}; per = {h: [] for h in HS}
        for s in seeds:
            ps, pf = pairs(cv, s, n_cls=n_cls)
            for h in HS:
                k = (ps >= G.PCT) & (ps < G.PCT + h)
                o = (pf[k] < G.PCT)
                pool[h][0] += k.sum(); pool[h][1] += o.sum()
                if k.any(): per[h].append(o.mean())
        out = []
        for h in HS:
            n, o = pool[h]; p = o / n; v = np.array(per[h])
            out.append(f"h={h:g}: пар {n}, q={p:.3f}±{1.96*np.sqrt(p*(1-p)/n):.3f}, q≥0,5 у {(v>=0.5).sum()}/{len(v)}")
        print(f"классов {n_cls}, затравок {len(seeds)}, КВ {cv}: " + "; ".join(out))
