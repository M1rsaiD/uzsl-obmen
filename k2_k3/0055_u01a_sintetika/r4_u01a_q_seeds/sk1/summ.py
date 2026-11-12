import json, sys, numpy as np
rows = {}
for ln in open(sys.argv[1]):
    cv, s, js = ln.split(" ", 2); rows.setdefault(float(cv), {})[int(s)] = json.loads(js)
lo, hi = int(sys.argv[2]), int(sys.argv[3])
for cv, d in rows.items():
    S = [s for s in sorted(d) if lo <= s < hi]
    print(f"КВ={cv}: затравки {lo}..{hi-1} (n={len(S)}) — min / медиана / max; затравок с q ≥ 0,5")
    for k in d[S[0]]:
        if k.startswith("tau"):
            v = np.array([d[s][k] for s in S])
        else:
            v = np.array([d[s][k][1] / d[s][k][0] if d[s][k][0] else np.nan for s in S])
            n = np.array([d[s][k][0] for s in S]); o = np.array([d[s][k][1] for s in S])
        extra = f"; пул: пар(упоряд.) {n.sum()}, q={o.sum()/n.sum():.3f}; n на затравку {n.min()}–{n.max()}" if k.startswith("h") else ""
        print(f"  {k:7s}: {np.nanmin(v):.3f} / {np.nanmedian(v):.3f} / {np.nanmax(v):.3f}; q≥0,5: {(v >= 0.5).sum()}/{len(S)}{extra}")
