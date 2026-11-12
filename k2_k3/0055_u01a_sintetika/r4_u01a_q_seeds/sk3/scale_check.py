"""Скептик 3: как q у порога зависит от числа классов (шум малой выборки пар) и записей на класс (шум средних); векторно.
Векторная версия сверена с циклом G.pct (loop_seeds.out) на 20 затравках при n_cls=200, per=6."""
import sys
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}; exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

def run(cv, seed, n_cls, per):
    recs, lab = make(cv, n_cls=n_cls, per=per, seed=seed)
    F = G.place_feats(recs)
    by = {}
    for i, c in enumerate(lab):
        by.setdefault(c, []).append(F[i])
    means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
    rp = G.ref_pairs(len(means[0]), np.random.default_rng(G.RNG))
    refs = [G.pct_ref(m, rp) for m in means]
    cls = sorted(by)
    M1 = np.array([means[1][c] for c in cls]); M2 = np.array([means[2][c] for c in cls])
    a, b = np.triu_indices(len(cls), 1)
    d1 = np.linalg.norm(M1[a] - M1[b], axis=1); d2 = np.linalg.norm(M2[a] - M2[b], axis=1)
    return 100 * np.searchsorted(refs[1], d1) / len(refs[1]), 100 * np.searchsorted(refs[2], d2) / len(refs[2])

HS = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05)
TAUS = (np.inf, 10, 5, 2, 1, 0.5)
for n_cls, per, seeds in ((600, 6, 10), (600, 2, 10), (200, 2, 20)):
    for cv in (0.05, 0.10):
        rows, pool = [], {h: [0, 0] for h in HS}
        for s in range(seeds):
            ps, pf = run(cv, 1000 + s, n_cls, per); m = ps >= 50; only = pf[m] < 50
            r = {}
            for tau in TAUS:
                w = np.ones(m.sum()) if np.isinf(tau) else np.exp(-(ps[m] - 50) / tau)
                r[f"tau{tau:g}"] = (w * only).sum() / w.sum()
            for h in HS:
                k = ps[m] < 50 + h
                r[f"h{h:g}"] = only[k].mean(); pool[h][0] += k.sum(); pool[h][1] += only[k].sum()
            rows.append(r)
        print(f"n_cls={n_cls} per={per} КВ={cv}: {seeds} затравок (1000+), min/мед/max; q≥0,5")
        print("  " + "; ".join(f"{k} {np.min([r[k] for r in rows]):.3f}/{np.median([r[k] for r in rows]):.3f}/{np.max([r[k] for r in rows]):.3f} ({sum(r[k] >= 0.5 for r in rows)})" for k in rows[0]))
        print("  пул: " + "; ".join(f"h={h:g}: n={n // seeds}/затр, q={o / n:.3f}" for h, (n, o) in pool.items()))
