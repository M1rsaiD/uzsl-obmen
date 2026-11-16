"""к3 0040: независимый пересчёт рисков отбора UzSohn (поправка 2 разд. 3.5) по таблице входов к2.
Код к2 (risk_series_k2.py) не читался; модель — по тексту поправки 2, 3.5. ЦП, numpy."""
import json, sys
import numpy as np

inp = json.load(open(sys.argv[1]))
K = inp["order"]
sd = np.array([inp["by_signer"][k]["sd_seeds_pp"] for k in K])
p = np.array([inp["by_signer"][k]["top1_mean_pp"] for k in K]) / 100
n = np.array([inp["by_signer"][k]["n_test"] for k in K])
V = np.array([inp["by_signer"][k]["n_as_Vk"] for k in K])
mac = np.array(inp["macro_by_seed_pp"])

delta = mac.max() - mac.min()                      # δ — наибольшая разность затравок
THR = max(1.0, delta)
sd_macro = mac.std(ddof=1)                          # 0,512
sd_model = np.sqrt((sd**2).sum()) / 10              # 0,662: СКО macro по M1
c3 = (10 * sd_macro) ** 2 / (p * (1 - p) / n * 1e4).sum()   # M3: c под наблюдённое СКО macro
print(f"delta={delta:.7f} THR={THR:.4f} sd_macro={sd_macro:.4f} sd_model_M1={sd_model:.4f} "
      f"ratio={sd_macro/sd_model:.4f} c3(n_test)={c3:.4f}")
c3V = (10 * sd_macro) ** 2 / (p * (1 - p) / V * 1e4).sum()
print(f"c3 при n = V: {c3V:.4f}")

models = {"M1": sd, "M2": sd * 0.512 / 0.662,
          "M3": np.sqrt(0.219 * p * (1 - p) / n * 1e4)}
NARM, R = 5, 400_000

def run(sig, discrete, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros((R, NARM), bool)
    for lo in range(0, R, 50_000):
        m = min(50_000, R - lo)
        e = rng.standard_normal((m, NARM + 1, 10)) * sig        # шум контроля (0) и 5 плеч, по V_k
        if discrete:                                            # счёт в целых записях V_k
            s = np.rint((p * 100 + e) * V / 100) / V * 100
        else:
            s = e
        d = s[:, 1:, :] - s[:, :1, :]
        acc[lo:lo + m] = (d.mean(-1) >= THR) & ((d > 0).sum(-1) >= 6)   # ничья Δ_k = 0 — «не лучше»
    a1 = acc[:, 0].mean()
    a4 = acc[:, :4].any(1).mean()
    a5 = acc.any(1).mean()
    none4 = ~acc[:, :4].any(1)
    cond = acc[none4, 4].mean()
    return a1, a4, a5, 1 - (1 - a1) ** 5, cond

print("модель | α1 | α4 | α5 | 1-(1-α1)^5 | P(A5 | ни одно A1..A4)")
for nm, sig in models.items():
    for disc in ([False, True] if nm in ("M1", "M2") else [False]):
        r = run(sig, disc, 20261117)
        print(f"{nm}{' целые' if disc else ''} | " + " | ".join(f"{x:.3f}" for x in r))
