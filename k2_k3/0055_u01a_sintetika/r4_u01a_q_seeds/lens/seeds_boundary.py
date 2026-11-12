"""Проверка устойчивости q у порога: та же синтетика (make из revision_r1), разные затравки make (seed) — 0..19.
Без данных и моделей. Вектора вместо цикла по парам (тот же результат, что G.pct)."""
import sys
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}
exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

def run(cv, seed):
    recs, lab = make(cv, seed=seed)
    F = G.place_feats(recs)
    by = {}
    for i, c in enumerate(lab):
        by.setdefault(c, []).append(F[i])
    means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
    rp = G.ref_pairs(len(means[0]), np.random.default_rng(G.RNG))
    refs = [G.pct_ref(m, rp) for m in means]
    cls = sorted(by)
    M1 = np.array([means[1][c] for c in cls]); M2 = np.array([means[2][c] for c in cls])
    a, b = np.triu_indices(len(cls), 1)           # неупорядоченные пары (обе стороны дают то же расстояние)
    d1 = np.linalg.norm(M1[a] - M1[b], axis=1); d2 = np.linalg.norm(M2[a] - M2[b], axis=1)
    ps = 100 * np.searchsorted(refs[1], d1) / len(refs[1]); pf = 100 * np.searchsorted(refs[2], d2) / len(refs[2])
    return ps, pf

if __name__ == "__main__":
    # сверка с исходной синтетикой на seed 0 (упорядоченные пары = 2 × неупорядоченные)
    for cv in (0.05, 0.10):
        ps, pf = run(cv, 0); m = ps >= 50
        print(f"seed0 cv={cv}: place(ordered)={2*m.sum()} only(ordered)={2*(pf[m]<50).sum()} q={(pf[m]<50).mean():.4f}")
    res = {}
    for cv in (0.05, 0.10):
        rows = []
        pool = {h: [0, 0] for h in (1.0, 0.5, 0.1, 0.05)}
        for s in range(20):
            ps, pf = run(cv, s); m = ps >= 50; only = pf[m] < 50
            r = {"tau_inf": only.mean()}
            for tau in (10, 5, 2, 1, 0.5):
                w = np.exp(-(ps[m] - 50) / tau); r[f"tau{tau}"] = (w * only).sum() / w.sum()
            for h in (1.0, 0.5, 0.1, 0.05):
                k = ps[m] < 50 + h
                r[f"h{h}"] = only[k].mean() if k.any() else np.nan
                pool[h][0] += k.sum(); pool[h][1] += only[k].sum()
            rows.append(r)
        keys = rows[0].keys()
        print(f"\nКВ={cv}: по 20 затравкам make — min / медиана / max; доля затравок с q ≥ 0,5")
        for k in keys:
            v = np.array([r[k] for r in rows], float)
            print(f"  {k:8s}: {np.nanmin(v):.3f} / {np.nanmedian(v):.3f} / {np.nanmax(v):.3f};  q≥0,5: {(v >= 0.5).sum()}/20")
        for h, (n, o) in pool.items():
            print(f"  пул h={h}: неупорядоченных пар {n}, q={o/n:.3f}")
