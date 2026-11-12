import json, numpy as np
Z = np.load("arrs_0_100.npz")
loop = {}
for ln in open("loop_0_100.out"):
    cv, s, js = ln.split(" ", 2); loop[(float(cv), int(s))] = json.loads(js)
edges = [50, 50.02, 50.05, 50.1, 50.2, 50.5, 51, 52, 55, 60, 70, 100.01]
for cv in (0.05, 0.10):
    ok = all([int(((Z[f"ps_{cv}_{s}"] < 50 + h)).sum()) == loop[(cv, s)][f"h{h:g}"][0] and
              int(((Z[f"ps_{cv}_{s}"] < 50 + h) & (Z[f"pf_{cv}_{s}"] < 50)).sum()) == loop[(cv, s)][f"h{h:g}"][1]
              for s in range(100) for h in (1.0, 0.5, 0.1)])
    ps = np.concatenate([Z[f"ps_{cv}_{s}"] for s in range(100)]); pf = np.concatenate([Z[f"pf_{cv}_{s}"] for s in range(100)])
    print(f"КВ={cv}: сверка с loop_0_100.out: {ok}; пары «место» (упоряд., 100 затравок) {len(ps)}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (ps >= lo) & (ps < hi)
        n = k.sum(); o = (pf[k] < 50).sum()
        print(f"  ps ∈ [{lo:g}; {hi:g}): n={n:6d} (неупоряд. {n//2}), q={o/n:.3f} ± {np.sqrt(o/n*(1-o/n)/(n/2)):.3f}")
    for h in (0.05, 0.02):
        k = ps < 50 + h; print(f"  пул h={h}: n(неупоряд.)={k.sum()//2}, q={(pf[k] < 50).mean():.3f}")
