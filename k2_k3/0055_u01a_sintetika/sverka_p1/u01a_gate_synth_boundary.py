"""Синтетика ворот U01a (без данных и моделей): как revision_r1/u01a_gate_synth.py (те же make и функции ворот), но ошибки
«место» сосредоточены у порога 50: вес пары ∝ exp(−(процентиль «место / плечи» − 50) / τ). τ = ∞ — все пары поровну (= r1).
Вопрос: открываются ли ворота (q ≥ 0,5), если путаются в основном пары у самого порога."""
import sys
from pathlib import Path

import numpy as np

OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
src = (OM / "runs/UZSOHN/revision_r1/u01a_gate_synth.py").read_text().split("\nfor cv in")[0]   # только make и импорты
ns = {}
exec(compile(src, "u01a_gate_synth.py", "exec"), ns)
G, make = ns["G"], ns["make"]

for cv in (0.05, 0.10):
    recs, lab = make(cv)
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
    row = [f"КВ = {cv:.2f}: пар «место» {m.sum()}, «только плечи» {only.sum()}"]
    for tau in (np.inf, 10.0, 5.0, 2.0, 1.0, 0.5):
        w = np.ones(m.sum()) if np.isinf(tau) else np.exp(-(ps[m] - G.PCT) / tau)
        row.append(f"τ={tau:g}: q={float((w * only).sum() / w.sum()):.3f}")
    # худший случай: все ошибки «место» — в парах, ближайших к порогу (доля пар с процентилем «место / плечи» < 50 + h)
    for h in (1.0, 0.5, 0.1):
        k = ps[m] < G.PCT + h
        row.append(f"только пары [50; {50 + h:g}): n={int(k.sum())}, q={float(only[k].mean()) if k.any() else float('nan'):.3f}")
    print("; ".join(row))
