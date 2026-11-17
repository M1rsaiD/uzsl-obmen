"""Независимая сверка чисел U28 (вопрос 2) по 10 разностям из T9_recheck (округлены до 0,01). Только ЦП."""
import json
import numpy as np
from scipy import stats, optimize
T9 = "/home/error_404/uzsign/uzsign_ai_v2/reports/T9_recheck_2026-11-11.json"
row = next(r for r in json.load(open(T9))["rows"] if r["pair"] == "а − Sohn одиночная")
d = np.array(list(row["per_signer_diff"].values()), float)
n = len(d); m, sd = d.mean(), d.std(ddof=1); se = sd / n ** 0.5
t995 = stats.t.ppf(0.995, 9)
print("JSON:", row["diff"], row["pct_ci"], row["t_ci"])
print(f"m={m:.4f} sd={sd:.4f} se={se:.4f} t={m/se:.3f} t9_99=[{m-t995*se:.3f}; {m+t995*se:.3f}] need_m={t995*se:.3f}")
bs = d[np.random.default_rng(20261106).integers(0, n, (20000, n))].mean(1)
print("pct99 (округл. разности):", np.round(np.quantile(bs, [0.005, 0.995]), 3), "плюс", (d > 0).sum(), "минус", (d < 0).sum())
for s in (1.022, sd, 1.918):
    pw = {dl: 1 - stats.nct.cdf(t995, 9, dl * n ** 0.5 / s) for dl in (0.5, 1.05, 2.0)}
    d80 = optimize.brentq(lambda x: 1 - stats.nct.cdf(t995, 9, x * n ** 0.5 / s) - 0.8, 0.05, 20)
    print(f"sigma={s:.3f}: power", {k: round(v, 3) for k, v in pw.items()}, f"delta80={d80:.2f}")
z = (2 - m) / se
print(f"P(mu>=2): normal {stats.norm.sf(z)*100:.2f} %, t9 {stats.t.sf(z, 9)*100:.2f} %")
# Прогнозная мощность при апостериорном mu ~ m + se*t9 (неизвестная дисперсия), новая проба σ = sd
rng = np.random.default_rng(1)
mu = m + se * rng.standard_t(9, 400000)
pw = 1 - stats.nct.cdf(t995, 9, mu * n ** 0.5 / sd)
print(f"прогноз (t-апостериорное, σ=sd): {pw.mean():.3f}")
