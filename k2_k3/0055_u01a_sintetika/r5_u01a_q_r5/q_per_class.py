"""Скептик 1, р5: независимая проверка q при 2 записях на класс. make() из r1 + функции ворот G без изменений;
своя агрегация (свои процентили через G.pct на первых затравках как сверка векторной). Только синтетика, ЦП."""
import sys, time
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts"), str(OM / "src")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}; exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

def pcts(cv, seed, n_cls, per, loop=False):
    recs, lab = make(cv, n_cls=n_cls, per=per, seed=seed)
    F = G.place_feats(recs)
    assert len(F) == len(lab)
    lab = np.array(lab); cls = np.unique(lab)
    M = [np.array([np.mean([F[i][k] for i in np.where(lab == c)[0]], 0) for c in cls]) for k in range(3)]
    rp = G.ref_pairs(len(cls), np.random.default_rng(G.RNG))
    refs = [G.pct_ref({c: M[k][j] for j, c in enumerate(cls)}, rp) for k in range(3)]
    a, b = np.triu_indices(len(cls), 1)
    if loop:  # медленно, как r1: G.pct по одной паре
        ps = np.array([G.pct(refs[1], float(np.linalg.norm(M[1][i] - M[1][j]))) for i, j in zip(a, b)])
        pf = np.array([G.pct(refs[2], float(np.linalg.norm(M[2][i] - M[2][j]))) for i, j in zip(a, b)])
    else:
        ps = 100 * np.searchsorted(refs[1], np.linalg.norm(M[1][a] - M[1][b], axis=1)) / len(refs[1])
        pf = 100 * np.searchsorted(refs[2], np.linalg.norm(M[2][a] - M[2][b], axis=1)) / len(refs[2])
    return ps, pf

def qs(ps, pf):
    m = ps >= G.PCT; only = pf[m] < G.PCT
    out = {"uni": only.mean()}
    for tau in (2.0, 1.0):
        w = np.exp(-(ps[m] - G.PCT) / tau); out[f"t{tau:g}"] = (w * only).sum() / w.sum()
    return out

if sys.argv[1] == "loopcheck":   # векторная = цикл G.pct на 3 затравках
    for s in (1000, 1001, 1002):
        p1, f1 = pcts(0.05, s, 200, 2); p2, f2 = pcts(0.05, s, 200, 2, loop=True)
        print(s, np.abs(p1 - p2).max(), np.abs(f1 - f2).max(), qs(p1, f1))
    sys.exit()
n_cls, per, s0, N = map(int, sys.argv[1:5])
for cv in (0.05, 0.10):
    t = time.time(); R = [qs(*pcts(cv, s0 + s, n_cls, per)) for s in range(N)]
    for k in ("uni", "t2", "t1"):
        v = np.array([r[k] for r in R]); v20 = v[:20]
        print(f"n_cls={n_cls} per={per} КВ={cv} seeds {s0}..{s0+N-1} {k}: мед {np.median(v):.3f} p95 {np.percentile(v,95):.3f} "
              f"p99 {np.percentile(v,99):.3f} max {v.max():.3f}; первые 20: max {v20.max():.3f}; доля > max20: {(v > v20.max()).mean():.3f}; q≥0,5: {(v>=0.5).sum()}")
    print(f"  ({time.time()-t:.0f} с)", flush=True)
