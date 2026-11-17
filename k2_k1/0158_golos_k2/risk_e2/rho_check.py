"""Задача B: связь шума плеч на одной затравке (ρ пары) — по старым сериям (noise_records.json). Только ЦП.
1) Повтор той же затравки с тем же конфигом (M8.0: M8/slovo_a0_recheck seed0 против M1/slovo_a0 seed0) против разных затравок M1.
2) Разные условия одной серии на одной затравке: корреляция остатков (y − среднее по затравкам внутри (условие, фолд))
   у пар условий на общих (фолд, затравка). Независимый шум → ≈ 0; общий сдвиг затравки → > 0.
3) Главный эффект затравки, общий у разных условий (одна серия, один набор): доля от разброса главных эффектов."""
import json, itertools
from collections import defaultdict
from pathlib import Path
import numpy as np

D = json.load(open(Path(__file__).with_name("noise_records.json")))
F = [f"fold{i}" for i in range(1, 6)]
idx = {(r["cond"], r["fold"], r["seed"]): r for r in D["records"]}
for key in ("val_last", "test_macro", "test_micro"):
    same = np.array([100 * (idx[("M8/slovo_a0_recheck", f, 0)][key] - idx[("M1/slovo_a0", f, 0)][key]) for f in F])
    diff = np.array([100 * (idx[("M1/slovo_a0", f, a)][key] - idx[("M1/slovo_a0", f, b)][key])
                     for f in F for a, b in ((0, 1), (0, 2), (1, 2))])
    print(f"[1] {key}: та же затравка 0 (M8.0 − M1), по фолдам {np.round(same, 2).tolist()}; "
          f"СКО разности {np.sqrt((same ** 2).mean()):.2f} против разных затравок M1 {np.sqrt((diff ** 2).mean()):.2f} п. п. "
          f"→ ρ̂ = 1 − Var_same/Var_diff = {1 - (same ** 2).mean() / (diff ** 2).mean():+.2f} (5 фолдов, грубо)")

g = defaultdict(dict)
for r in D["records"]:
    g[(r["series"], r["ds"])].setdefault(r["cond"], {})[(r["fold"], r["seed"])] = r
for key in ("val_last", "test_macro"):
    for (ser, ds), conds in sorted(g.items()):
        full = {c: v for c, v in conds.items() if all((f, s) in v for f in F for s in (0, 1, 2))}
        if len(full) < 2:
            continue
        res = {}
        for c, v in full.items():
            Y = np.array([[100 * v[(f, s)][key] for s in (0, 1, 2)] for f in F])     # фолды × затравки
            res[c] = Y - Y.mean(1, keepdims=True)
        cors = [np.corrcoef(res[a].ravel(), res[b].ravel())[0, 1] for a, b in itertools.combinations(sorted(res), 2)]
        # главный эффект затравки по условиям: M[c, s] = среднее остатков по фолдам
        M = np.array([res[c].mean(0) for c in sorted(res)])
        common = M.mean(0)                      # общий для условий эффект затравки
        ss_common = len(res) * (common ** 2).sum()
        ss_inter = ((M - common) ** 2).sum()
        print(f"[2] {key} {ser}/{ds}: условий {len(res)}, пар {len(cors)}: средняя корр. остатков {np.mean(cors):+.3f} "
              f"(медиана {np.median(cors):+.3f}); [3] главный эффект затравки: общий у условий SS {ss_common:.2f} (df 2) "
              f"против условие×затравка SS {ss_inter:.2f} (df {2 * (len(res) - 1)})")
