"""Поправка 2 UzSohn: риск одного шага (α₁) и накопленный по серии (α_n, n ≤ 5) — та же модель шума, что
runs/UZSOHN/revision_r2/false_accept_sim.py (M1–M3) и false_accept_sim_discrete.py (M1, M2 в целых записях V_k).
Все плечи серии — против одного контроля (этап 1 ничего не принял): accept_j = решение по e_j − e_0.
Печатает: α₁ (по каждому плечу и среднее), α_n = P(∃ j ≤ n: accept_j), жадную цепочку (как в false_accept_sim, должна
совпасть с α_n), 1 − (1 − α₁)^n и P(accept_5 | ни одно из 1–4 не принято). Только ЦП, только готовые metrics.json."""
import json
import numpy as np
R = "/home/error_404/uzsign/osnovnaya_model/runs/UzSohn/alluz_pre"
F = sorted(json.load(open("/home/error_404/uzsign/uzsign_ai_v2/splits/uzsl_allz_loso.json"))["folds"])
A = np.array([[100 * json.load(open(f"{R}/{f}/seed{s}/metrics.json"))["top1"] for f in F] for s in (42, 43, 44)])
N = np.array([json.load(open(f"{R}/{f}/seed42/metrics.json"))["n_test"] for f in F])
NV = np.array([143, 168, 479, 1158, 621, 135, 265, 812, 76, 77])   # как false_accept_sim_discrete.py (порядок F)
THR, MINB, NARM, K = 1.0081957667939179, 6, 5, 400_000
sd = A.std(0, ddof=1)
macro_sd_obs = A.mean(1).std(ddof=1)
p = A.mean(0) / 100
binvar = p * (1 - p) / N * 1e4
c = (sd ** 2 / binvar).mean()
s2 = sd * macro_sd_obs / (np.sqrt((sd ** 2).sum()) / 10)
models = {"M1": sd, "M2": s2, "M3": np.sqrt(c * binvar)}


def acc_cont(d):
    return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)


def acc_disc(dr):   # dr — разность в записях; ничья Δ_k = 0 — «не лучше» (как discrete)
    return ((100 * dr / NV).mean(1) >= THR - 1e-12) & ((dr > 0).sum(1) >= MINB)


def report(name, e, acc):
    a = [acc(e[j] - e[0]) for j in range(1, NARM + 1)]
    a1 = np.mean([x.mean() for x in a])
    line = [f"{name}: α₁ по плечам " + " ".join(f"{x.mean():.4f}" for x in a) + f"; среднее {a1:.4f}"]
    anyacc = np.zeros(K, bool)
    cur = e[0].copy(); greedy = np.zeros(K, bool)
    for n in range(1, NARM + 1):
        anyacc |= a[n - 1]
        ok = acc(e[n] - cur); greedy |= ok; cur = np.where(ok[:, None], e[n], cur)
        assert (greedy == anyacc).all()   # до первого принятия все плечи — против e_0: жадный перенос P(≥ 1 ложного) не меняет
        line.append(f"   n = {n}: α_n = {anyacc.mean():.4f} (жадно — то же); 1 − (1 − α₁)^n = {1 - (1 - a1) ** n:.4f}")
    none4 = ~(a[0] | a[1] | a[2] | a[3])
    line.append(f"   P(принять 5-е | 1–4 не приняты) = {(a[4] & none4).mean() / none4.mean():.4f} (α₁ 5-го без условия {a[4].mean():.4f})")
    print("\n".join(line))


print("СКО по исп.:", np.round(sd, 2), "| СКО macro набл.:", round(macro_sd_obs, 3), "| c =", round(c, 3))
rng = np.random.default_rng(20261116)
for name, s in models.items():
    report(name + " непрерывно", [rng.normal(0, s, (K, 10)) for _ in range(NARM + 1)], acc_cont)
for name, scale in (("M1", 1.0), ("M2", macro_sd_obs / (np.sqrt((sd ** 2).sum()) / 10))):
    s_rec = sd * scale * NV / 100
    report(name + " в целых записях V_k", [np.rint(rng.normal(0, s_rec, (K, 10))) for _ in range(NARM + 1)], acc_disc)
