"""U28 (ансамбль 3 затравок Sohn, ALL-UZ): сверка чисел по логитам и мощность новой пробы (б). Только ЦП, только чтение.

Запуск (ГП не трогает, 4 потока):
  cd <эта папка>; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 \
    /home/error_404/.local/share/mise/installs/python/3.14.7/bin/python u28_power.py > u28_power.out
Вход (только чтение): логиты Sohn ALL-UZ OM runs/UzSohn/alluz_pre/<фолд>/seed{42,43,44}; разбиение, коды, кэш — V2;
T9_recheck_2026-11-11.json (сверка). Выход: только S-коды (имена фолдов и записей не печатаются).
Правило «оба» (предрег. UzSohn a302d0f, строки 537–540): перцентильный 99 % (alluz.boot: 20 000, default_rng(20261106)) и
t₉ 99 % — обе нижние границы > 0.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

V2 = Path("/home/error_404/uzsign/uzsign_ai_v2")
sys.path.insert(0, str(V2 / "scripts"))
from alluz import N_BOOT, SEED_BOOT, SOHN_RUNS, SPL, boot  # noqa: E402  тот же бутстреп, что в серии
from uzsohn_diag import macro  # noqa: E402  то же macro (запись — среднее по затравкам; macro по 10 исполнителям)

OUT = Path(__file__).resolve().parent
T995 = float(stats.t.ppf(0.995, 9))
M = 20_000  # наборов на условие
rng = np.random.default_rng(20261116)
res = {}

# ---------- 1. логиты Sohn ALL-UZ → по записям: 3 одиночки, пары, ансамбль 3 (сумма логитов, как uzsohn_diag)
sp = json.loads((SPL / "uzsl_allz_loso.json").read_text())["folds"]
scode = json.loads((V2 / "data/signer_codes.local.json").read_text())["signers"]
clips = json.loads(str(np.load(V2 / "cache/uzsl_allz.npz")["clips"]))
gl = {c["sample_id"]: c["gloss"] for c in clips}
classes = sorted(set(gl.values()))
ci = {c: i for i, c in enumerate(classes)}
SEEDS = (42, 43, 44)
PAIRS = ((0, 1), (0, 2), (1, 2))
rec = {}  # S-код → dict массивов по записям
for f, fo in sp.items():
    te = sorted(fo["test"])
    yy = np.array([ci[gl[s]] for s in te])
    mats = []
    for s in SEEDS:
        d = SOHN_RUNS / "alluz_pre" / f / f"seed{s}"
        z, p = np.load(d / "logits.npz"), np.load(d / "probs.npz")
        assert (z["sample_ids"] == p["ids"]).all()
        ids, lg, cls = z["sample_ids"], z["logits"], p["classes"]
        assert sorted(ids) == te
        full = np.full((len(ids), len(classes)), -np.inf, np.float32)
        full[:, [ci[c] for c in cls]] = lg[np.argsort(ids)]
        mats.append(full)
    hit = lambda x: (x.argmax(1) == yy).astype(float)  # noqa: E731  как uzsohn_diag.hit
    rec[scode[f]] = {"ids": te, "c": np.stack([hit(x) for x in mats], 1), "E3": hit(sum(mats)),
                     "E2": np.stack([hit(mats[i] + mats[j]) for i, j in PAIRS], 1)}
S = sorted(rec)  # S01 … S12 — порядок как в alluz.boot (sorted)
sig = {s: k for k in S for s in rec[k]["ids"]}
single = macro({s: float(v) for k in S for s, v in zip(rec[k]["ids"], rec[k]["c"].mean(1))}, sig)
ens3 = macro({s: float(v) for k in S for s, v in zip(rec[k]["ids"], rec[k]["E3"])}, sig)
b = boot(ens3[1], single[1], 0.99)
d_obs = np.array([ens3[1][k][0] - single[1][k][0] for k in S])
t9 = json.loads((V2 / "reports/T9_recheck_2026-11-11.json").read_text())
row = next(r for r in t9["rows"] if r["pair"] == "а − Sohn одиночная")
assert {k: round(float(v), 2) for k, v in zip(S, d_obs)} == row["per_signer_diff"], "разности не совпали с T9"
assert [b["diff"]] + b["ci"] == [row["diff"]] + row["pct_ci"], (b, row)
m, sd = d_obs.mean(), d_obs.std(ddof=1)
tci = [m - T995 * sd / 10 ** 0.5, m + T995 * sd / 10 ** 0.5]
assert [round(x, 2) for x in tci] == row["t_ci"]
res["check"] = {"macro_single": single[0], "macro_ens3": ens3[0], "boot": b, "sd": round(sd, 3), "t9_ci": [round(x, 2) for x in tci],
                "per_signer": {k: round(float(v), 2) for k, v in zip(S, d_obs)},
                "n_test": {k: len(rec[k]["ids"]) for k in S}, "matches_T9_json": True}

# ---------- 2. шум затравок в разности «ансамбль − одиночка» по исполнителю (те же записи теста — новые затравки)
# σ_rec,k = sqrt(var_r(g_r)/n_k), g_r = E3_r − mean_s c_rs: верхняя оценка шума затравок при независимых записях
# (включает и постоянный разброс записей); проверка модели — на парах: s²(3 пар) / σ²_rec(пары) ожидается в [0,5; 1].
noise = {}
for k in S:
    r = rec[k]
    n = len(r["ids"])
    g = r["E3"] - r["c"].mean(1)
    g2 = r["E2"] - np.stack([(r["c"][:, i] + r["c"][:, j]) / 2 for i, j in PAIRS], 1)
    v2rec = np.mean([np.var(g2[:, q], ddof=1) / n for q in range(3)]) * 1e4
    noise[k] = {"n": n, "sigma_rec": 100 * (np.var(g, ddof=1) / n) ** 0.5, "share_unstable": float((r["c"].std(1) > 0).mean()),
                "pairs_s2_over_rec": float(np.var(100 * g2.mean(0), ddof=1) / v2rec),
                "seed_sd_single": float(np.std(100 * r["c"].mean(0), ddof=1))}
se = np.array([noise[k]["sigma_rec"] for k in S])
tau2 = max(0.0, sd ** 2 - np.mean(se ** 2))
ratio = np.mean([noise[k]["pairs_s2_over_rec"] for k in S])
res["seed_noise"] = {"per_signer": {k: {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in noise.items()},
                     "mean_sigma_e2": round(float(np.mean(se ** 2)), 3), "sd2_total": round(sd ** 2, 3), "tau2": round(tau2, 3),
                     "f_seed_share": round(float(np.mean(se ** 2) / sd ** 2), 3), "pairs_ratio_mean": round(float(ratio), 3)}

# ---------- 3. правило «оба» (точно как в серии: тот же массив индексов бутстрепа)
IDX = np.random.default_rng(SEED_BOOT).integers(0, 10, (N_BOOT, 10))
W = np.zeros((N_BOOT, 10))
np.add.at(W, (np.repeat(np.arange(N_BOOT), 10), IDX.ravel()), 1)


def both_plus(D):
    """D (M, 10) в порядке S → (оба: «установлено (+)», t₉: нижняя > 0, перцентильный: нижняя > 0 — только где t₉ > 0)."""
    mm, ss = D.mean(1), D.std(1, ddof=1)
    tok = mm - T995 * ss / 10 ** 0.5 > 0
    est = np.zeros(len(D), bool)
    cand = np.flatnonzero(tok)
    for ch in np.array_split(cand, max(1, len(cand) // 1000)) if len(cand) else []:
        est[ch] = np.quantile(W @ D[ch].T / 10, (1 - 0.99) / 2, axis=0) > 0
    return est, tok


e0, t0 = both_plus(d_obs[None])
lo_pct = np.quantile(W @ d_obs / 10, (1 - 0.99) / 2)
assert round(float(lo_pct), 2) == b["ci"][0] and not e0[0], "правило не воспроизвело U28"

# ---------- 4А. мощность новой пробы, «свежие» 10 исполнителей (классическая): d ~ δ + σ·ε
chi = stats.chi2.ppf([0.9, 0.1], 9)
sig_lo, sig_hi = sd * (9 / chi) ** 0.5  # 80 % ДИ σ по 10 разностям
zobs = (d_obs - m) / d_obs.std(ddof=0)  # форма наблюдённых разностей, дисперсия 1
A = {}
for delta in (0.5, 1.05, 2.0):
    for sg in (sig_lo, sd, sig_hi):
        nc = delta * 10 ** 0.5 / sg
        an = 1 - stats.nct.cdf(T995, 9, nc)
        en, tn = both_plus(delta + sg * rng.standard_normal((M, 10)))
        ee, te_ = both_plus(delta + sg * zobs[rng.integers(0, 10, (M, 10))])
        A[f"d={delta} s={sg:.2f}"] = {"t9_analytic": round(an, 3), "both_normal": round(en.mean(), 3),
                                      "t9_only_normal": round(tn.mean(), 3), "both_obs_shape": round(ee.mean(), 3)}
d80 = optimize.brentq(lambda x: 1 - stats.nct.cdf(T995, 9, x * 10 ** 0.5 / sd) - 0.8, 0.1, 10)
res["A_fresh_signers"] = {"sigma_80ci": [round(sig_lo, 3), round(sig_hi, 3)], "power": A, "delta_for_80pct_at_sd": round(d80, 2)}

# ---------- 4Б. те же 10 исполнителей и записи: условная мощность при μ = δ (данные U28 известны) и прогнозная
# g_k | d_k, μ ~ N(μ + B_k(d_k − μ), B_k σ_e,k²), B_k = τ²/(τ² + σ_e,k²); d_new = g + e, e ~ N(0, σ_e,k²)


def cond(mu, se_, tau2_):
    Bk = tau2_ / (tau2_ + se_ ** 2) if tau2_ > 0 else np.zeros_like(se_)
    mu = np.broadcast_to(np.asarray(mu, float).reshape(-1, 1), (M, 10))
    g = mu + Bk * (d_obs - mu) + (Bk * se_ ** 2) ** 0.5 * rng.standard_normal((M, 10))
    dn = g + se_ * rng.standard_normal((M, 10))
    en, _ = both_plus(dn)
    ep, _ = both_plus((d_obs + dn) / 2)
    return round(en.mean(), 3), round(ep.mean(), 3), round(float(dn.mean(1).mean()), 2)


def post_mu(se_, tau2_):
    w = 1 / (tau2_ + se_ ** 2)
    return (w * d_obs).sum() / w.sum(), w.sum() ** -0.5


Bres = {}
for name, se_, tau2_ in (("оценка (σ_e,k по записям, τ² по моментам)", se, tau2),
                         ("τ² = 0 (весь разброс — шум затравок)", np.full(10, sd), 0.0),
                         ("половина — шум затравок", np.full(10, sd / 2 ** 0.5), sd ** 2 / 2)):
    pm, ps = post_mu(se_, tau2_)
    rowb = {f"mu={delta}": cond(delta, se_, tau2_) for delta in (0.0, 0.5, 1.05, 2.0)}  # μ = 0 — ложное «установлено»
    rowb["прогноз (μ ~ апостериорное)"] = cond(pm + ps * rng.standard_normal(M), se_, tau2_)
    rowb["post_mu"] = [round(pm, 2), round(ps, 2)]
    rowb["P(mu>=2 | старые), %"] = round(100 * float(stats.norm.sf(2, pm, ps)), 2)
    Bres[name] = rowb
res["B_same_signers"] = {"format": "(новая одна: P(оба), объединение (старая+новая)/2: P(оба), среднее новой разности)", "rows": Bres}

# ---------- 4В. ошибка I рода «наивного объединения» (стоп, если старая установила; иначе — новая и объединение), μ = 0
Cres = {}
for name, se_, tau2_ in (("оценка", se, tau2), ("τ² = 0", np.full(10, sd), 0.0)):
    g = tau2_ ** 0.5 * rng.standard_normal((M * 5, 10))
    d1 = g + se_ * rng.standard_normal((M * 5, 10))
    d2 = g + se_ * rng.standard_normal((M * 5, 10))
    s1, _ = both_plus(d1)
    sp_, _ = both_plus((d1 + d2) / 2)
    Cres[name] = {"stage1": round(s1.mean() * 100, 3), "total_two_stage_pct": round((s1 | sp_).mean() * 100, 3)}
res["C_type1_naive_pooling_pct"] = Cres

# ---------- 5. вариант: новый ансамбль против СТАРОЙ одиночной (затравки разные) — σ больше
v_single = np.mean([noise[k]["seed_sd_single"] ** 2 / 3 for k in S])  # дисперсия среднего 3 одиночек по исполнителю
sd_ii = (sd ** 2 + 2 * v_single) ** 0.5
res["D_new_ens_vs_old_single"] = {"sd": round(sd_ii, 3), "power_t9_at_1.05": round(1 - stats.nct.cdf(T995, 9, 1.05 * 10 ** 0.5 / sd_ii), 3)}

print(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "u28_power.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
