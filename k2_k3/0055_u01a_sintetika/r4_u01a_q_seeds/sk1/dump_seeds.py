"""Как loop_seeds.py (цикл по упорядоченным парам, G.pct), но сохраняет процентили пар «место» (ps ≥ 50) и их pf — для бинов у порога."""
import sys
from pathlib import Path
import numpy as np
sys.argv += []
exec(open(Path(__file__).with_name("loop_seeds.py")).read().split("out = {}")[0])
def arrays(cv, seed):
    recs, lab = make(cv, seed=seed)
    F = G.place_feats(recs)
    by = {}
    for i, c in enumerate(lab):
        by.setdefault(c, []).append(F[i])
    means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
    rp = G.ref_pairs(len(means[0]), np.random.default_rng(G.RNG))
    refs = [G.pct_ref(m, rp) for m in means]
    cls = sorted(by)
    ps, pf = [], []
    for a in cls:
        for b in cls:
            if a != b:
                ps.append(G.pct(refs[1], float(np.linalg.norm(means[1][a] - means[1][b]))))
                pf.append(G.pct(refs[2], float(np.linalg.norm(means[2][a] - means[2][b]))))
    return np.array(ps), np.array(pf)
res = {}
for cv in (0.05, 0.10):
    for s in range(int(sys.argv[1]), int(sys.argv[2])):
        ps, pf = arrays(cv, s); m = ps >= 50
        res[f"ps_{cv}_{s}"], res[f"pf_{cv}_{s}"] = ps[m].astype(np.float32), pf[m].astype(np.float32)
np.savez_compressed(sys.argv[3], **res)
