"""Скептик 3: независимый повтор (цикл по упорядоченным парам и G.pct, как u01a_gate_synth_boundary.py) на затравках make 0..19."""
import sys, time
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}; exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

def cell(cv, seed):
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
    ps, pf = np.array(ps), np.array(pf)
    m = ps >= G.PCT; only = pf[m] < G.PCT
    r = {}
    for tau in (np.inf, 10.0, 5.0, 2.0, 1.0, 0.5):
        w = np.ones(m.sum()) if np.isinf(tau) else np.exp(-(ps[m] - G.PCT) / tau)
        r[f"tau{tau:g}"] = float((w * only).sum() / w.sum())
    for h in (1.0, 0.5, 0.1, 0.05):
        k = ps[m] < G.PCT + h
        r[f"h{h:g}"] = (float(only[k].mean()) if k.any() else float("nan"), int(k.sum()))
    return r

if __name__ == "__main__":
    t0 = time.time()
    for cv in (0.05, 0.10):
        rows = [cell(cv, s) for s in range(20)]
        print(f"КВ={cv}")
        for s, r in enumerate(rows):
            print(f"  seed {s:2d}: " + "; ".join(f"{k}={v:.3f}" if not isinstance(v, tuple) else f"{k}={v[0]:.3f}(n={v[1]})" for k, v in r.items()))
        for k in rows[0]:
            v = np.array([r[k] if not isinstance(r[k], tuple) else r[k][0] for r in rows])
            print(f"  {k:7s} min/med/max {np.nanmin(v):.3f}/{np.nanmedian(v):.3f}/{np.nanmax(v):.3f}; q>=0.5: {(v >= 0.5).sum()}/20")
        for h in (1.0, 0.5, 0.1, 0.05):
            n = sum(r[f'h{h:g}'][1] for r in rows); o = sum(r[f'h{h:g}'][0] * r[f'h{h:g}'][1] for r in rows)
            print(f"  пул h={h}: упорядоченных пар {n}, q={o/n:.3f}")
    print(f"время {time.time()-t0:.0f} с")
