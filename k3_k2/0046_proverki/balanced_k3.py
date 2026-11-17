"""к3 0046: balanced_p из диффа 0163 (строки 80–94) — свой счёт: внутри набора классы поровну, доли наборов = доли записей train."""
import re, numpy as np
src = open("k2_k1/0163_e2_levers_diff_ot_f6a5a72.diff", encoding="utf-8").read().splitlines()
i = next(k for k, l in enumerate(src) if l.startswith("+def balanced_p"))
body = []
for l in src[i:]:
    if not l.startswith("+"): break
    body.append(l[1:])
ns = {"np": np}; exec("\n".join(body), ns); balanced_p = ns["balanced_p"]
rng = np.random.default_rng(1)
# S+B: Slovo 1 001 класс (12–20 + no_event 360), Bukva 33 класса (71–94)
lab, gl = [], []
for c in range(1001):
    n = 360 if c == 0 else int(rng.integers(12, 21)); lab += [c] * n; gl += [f"slovo:{c}"] * n
for c in range(33):
    n = int(rng.integers(71, 95)); lab += [1001 + c] * n; gl += [f"bukva:{c}"] * n
lab = np.array(lab); train = list(rng.permutation(len(lab))[: int(0.9 * len(lab))])
p = balanced_p(train, lab, gl, "sb")
L = lab[train]; G = np.array([gl[k].split(":")[0] for k in train])
assert abs(p.sum() - 1) < 1e-12
for g in ("slovo", "bukva"):
    s = G == g
    assert abs(p[s].sum() - s.mean()) < 1e-12, g                       # доля набора = доля записей train
    pc = np.bincount(L[s], weights=p[s]); pc = pc[pc > 0]
    assert np.allclose(pc, pc[0]), g                                    # классы внутри набора поровну
print(f"S+B: доля Slovo {p[G == 'slovo'].sum():.4f} = {np.mean(G == 'slovo'):.4f}; классы внутри наборов поровну; no_event: "
      f"{p[L == 0].sum() / p[(G == 'slovo')].sum() * 1001:.3f} доли класса (= 1)")
p1 = balanced_p(train, lab, gl, "corpus")                               # один набор
pc = np.bincount(L, weights=p1); pc = pc[pc > 0]; assert np.allclose(pc, pc[0])
print("корпус (один набор): все классы поровну — верно")
