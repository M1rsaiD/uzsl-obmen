"""200 затравок make (200 классов): q при весе exp(−(pct − 50)/τ) и равномерном составе; примеры затравок из находки."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/r4_round4_u01a_sk2")
from indep_check import pairs, qs
for cv, s in ((0.10, 1), (0.05, 2)):
    r = qs(*pairs(cv, s)); print(f"пример КВ {cv} seed {s}: τ0,5 {r['tau=0.5']:.3f}; h1 {r['h=1'][0]:.3f}; h0,5 {r['h=0.5'][0]:.3f}; h0,1 {r['h=0.1'][0]:.3f} (n={r['h=0.1'][1]}, only={r['h=0.1'][2]})")
for cv in (0.05, 0.10):
    rows = [qs(*pairs(cv, s)) for s in range(200)]
    out = []
    for k in ("tau=inf", "tau=10", "tau=5", "tau=2", "tau=1", "tau=0.5"):
        v = np.array([r[k] for r in rows]); out.append(f"{k}: макс {v.max():.3f}, 99-й перц. {np.percentile(v, 99):.3f}, q≥0,5 у {(v>=0.5).sum()}/200")
    print(f"КВ {cv}: " + "; ".join(out))
