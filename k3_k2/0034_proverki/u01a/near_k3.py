"""к3: доля «только плечи» у пар у самого порога (процентиль «место/плечи» в [50; 50 + h)), пул по 100 реализациям."""
import numpy as np
from q_k3 import make

def near(cv, per, seeds=range(1, 101), hs=(0.1, 0.5, 1, 2, 5)):
    out = {h: [0, 0] for h in hs}
    for s in seeds:
        ps, pf, lab = make(cv, per=per, seed=s)
        cls = np.unique(lab)
        ms = np.array([ps[lab == c].mean(0) for c in cls]); mf = np.array([pf[lab == c].mean(0) for c in cls])
        g = np.random.default_rng(20261111)
        rp = g.integers(0, len(cls), (20000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
        rs = np.sort(np.linalg.norm(ms[rp[:, 0]] - ms[rp[:, 1]], axis=1)); rf = np.sort(np.linalg.norm(mf[rp[:, 0]] - mf[rp[:, 1]], axis=1))
        a, b = np.where(~np.eye(len(cls), dtype=bool))
        p_s = 100 * np.searchsorted(rs, np.linalg.norm(ms[a] - ms[b], axis=1)) / len(rs)
        p_f = 100 * np.searchsorted(rf, np.linalg.norm(mf[a] - mf[b], axis=1)) / len(rf)
        lo = 0
        for h in hs:
            m = (p_s >= 50 + lo) & (p_s < 50 + h)
            out[h][0] += m.sum(); out[h][1] += (m & (p_f < 50)).sum(); lo = h
    return {h: (n, k / n if n else float('nan')) for h, (n, k) in out.items()}

for cv in (0.05, 0.10):
    for per in (6, 2):
        r = near(cv, per)
        print(f"КВ {cv:.2f}, {per} зап.: " + "; ".join(f"[{lo}; {h}) n={n} доля {f:.3f}" for (h, (n, f)), lo in zip(r.items(), (0, 0.1, 0.5, 1, 2))))
