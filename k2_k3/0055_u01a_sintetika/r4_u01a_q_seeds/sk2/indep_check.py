"""Скептик 2 (р4, линза u01a): независимая проверка устойчивости q у порога по затравкам make (синтетика revision_r1).
Функции ворот — из scripts/uzsohn_gates.py без изменений; без данных и моделей, ЦП."""
import sys, time
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}
exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

def pairs(cv, seed, n_cls=200, exact=False):
    recs, lab = make(cv, n_cls=n_cls, seed=seed)
    F = G.place_feats(recs)
    by = {}
    for i, c in enumerate(lab):
        by.setdefault(c, []).append(F[i])
    means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
    rp = G.ref_pairs(len(means[0]), np.random.default_rng(G.RNG))
    refs = [G.pct_ref(m, rp) for m in means]
    cls = sorted(by)
    if exact:   # упорядоченные пары, G.pct — как в evidence/u01a_gate_synth_boundary.py
        ps, pf = [], []
        for a in cls:
            for b in cls:
                if a != b:
                    ps.append(G.pct(refs[1], float(np.linalg.norm(means[1][a] - means[1][b]))))
                    pf.append(G.pct(refs[2], float(np.linalg.norm(means[2][a] - means[2][b]))))
        return np.array(ps), np.array(pf)
    M1 = np.array([means[1][c] for c in cls]); M2 = np.array([means[2][c] for c in cls])
    a, b = np.triu_indices(len(cls), 1)
    d1 = np.linalg.norm(M1[a] - M1[b], axis=1); d2 = np.linalg.norm(M2[a] - M2[b], axis=1)
    return 100 * np.searchsorted(refs[1], d1) / len(refs[1]), 100 * np.searchsorted(refs[2], d2) / len(refs[2])

TAUS, HS = (np.inf, 10, 5, 2, 1, 0.5), (1.0, 0.5, 0.1, 0.05)
def qs(ps, pf):
    m = ps >= G.PCT; only = pf[m] < G.PCT; r = {}
    for t in TAUS:
        w = np.ones(m.sum()) if np.isinf(t) else np.exp(-(ps[m] - G.PCT) / t)
        r[f"tau={t:g}"] = float((w * only).sum() / w.sum())
    for h in HS:
        k = ps[m] < G.PCT + h
        r[f"h={h:g}"] = (float(only[k].mean()) if k.any() else np.nan, int(k.sum()), int(only[k].sum()))
    return r

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    t0 = time.time()
    for cv in (0.05, 0.10):   # сверка seed 0: точный цикл (как evidence) против векторов
        e = qs(*pairs(cv, 0, exact=True)); v = qs(*pairs(cv, 0))
        print(f"seed0 КВ={cv}: exact " + "; ".join(f"{k} {x if isinstance(x,float) else x[0]:.3f}" for k, x in e.items()))
        assert all(abs((e[k] if isinstance(e[k], float) else e[k][0]) - (v[k] if isinstance(v[k], float) else v[k][0])) < 1e-12 for k in e)
    print(f"seed0 exact == vectorized ok ({time.time()-t0:.0f} s)")
    for cv in (0.05, 0.10):
        rows = [qs(*pairs(cv, s)) for s in range(N)]
        print(f"\nКВ={cv}, затравки 0..{N-1}: min / медиана / max; число затравок с q ≥ 0,5 (решение «run»)")
        for k in rows[0]:
            v = np.array([r[k] if isinstance(r[k], float) else r[k][0] for r in rows], float)
            extra = ""
            if not isinstance(rows[0][k], float):
                n = sum(r[k][1] for r in rows); o = sum(r[k][2] for r in rows)
                extra = f"; пул: пар {n}, q={o/n:.3f}, 95% ДИ ±{1.96*np.sqrt(o/n*(1-o/n)/n):.3f}; пар на затравку {n/N:.1f}"
            print(f"  {k:8s}: {np.nanmin(v):.3f} / {np.nanmedian(v):.3f} / {np.nanmax(v):.3f}; q≥0,5: {(v >= 0.5).sum()}/{N}{extra}")
    print(f"\n({time.time()-t0:.0f} s)")
