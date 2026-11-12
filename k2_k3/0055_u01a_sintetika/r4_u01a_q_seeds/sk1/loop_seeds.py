"""Скептик 1, р4: та же синтетика, что fix_sverka_p1/evidence/u01a_gate_synth_boundary.py, ТОТ ЖЕ цикл по упорядоченным парам
с G.pct (без векторизации) — только затравка make(seed) меняется. Без данных и моделей."""
import sys, json
from pathlib import Path
import numpy as np
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]
ns = {}
exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]
TAUS = (np.inf, 10.0, 5.0, 2.0, 1.0, 0.5); HS = (1.0, 0.5, 0.1)

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
    m = ps >= G.PCT
    only = pf[m] < G.PCT
    r = {}
    for tau in TAUS:
        w = np.ones(m.sum()) if np.isinf(tau) else np.exp(-(ps[m] - G.PCT) / tau)
        r[f"tau{tau:g}"] = float((w * only).sum() / w.sum())
    for h in HS:
        k = ps[m] < G.PCT + h
        r[f"h{h:g}"] = (int(k.sum()), int(only[k].sum()))
    return r

out = {}
for cv in (0.05, 0.10):
    for s in range(int(sys.argv[1]), int(sys.argv[2])):
        out[f"{cv}/{s}"] = cell(cv, s)
        print(cv, s, json.dumps(out[f"{cv}/{s}"]), flush=True)
