"""Задача B: модель шума Э2 по старым сериям протокола П (noise_records.json от collect_noise.py). Только ЦП.
Считает по каждому условию (одна серия, один набор, ≥ 2 затравки на всех 5 фолдах):
  - СКО между затравками по фолдам (ddof 1): вал. micro на лучшей точке (val_best), на последней (val_last),
    тест macro по исполнителям (test_macro), тест micro (test_micro), в п. п.;
  - долю общего сдвига (главный эффект затравки по 5 фолдам, двухфакторный дисперсионный анализ «затравка × фолд»);
  - множитель c = Var затравок / биномиальная Var (по исполнителям теста и по вал. micro);
  - перевод на счёт Э2 — macro по исполнителям ВАЛИДАЦИИ фолда — тремя путями (V, T, C; см. ниже).
Пишет noise_summary.json и печатает таблицы."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
D = json.load(open(HERE / "noise_records.json"))
FOLDS = [f"fold{i}" for i in range(1, 6)]
# семейства прокси (по близости к Sohn: современный рецепт с EMA и выбором по валидации — «M7/M8»)
def family(r):
    return "M7/M8 (EMA или выбор по вал.)" if r["series"] in ("M7", "M8") else "M1/M2/M6/E1F (старый рецепт)"

g = defaultdict(list)
for r in D["records"]:
    g[(r["series"], r["cond"], r["ds"])].append(r)

comp = D["composition"]


def binvar_struct(ds, fold, part, p):
    """Биномиальные Var (в п. п.²) macro по исполнителям и micro для состава части; p — словарь исп. → точность."""
    c = comp[ds][fold][part]
    pm = np.mean(list(p.values()))
    ps = np.array([p.get(int(s), pm) for s in c])
    ns = np.array([c[s] for s in c], float)
    v_macro = (ps * (1 - ps) / ns).sum() / len(ns) ** 2 * 1e4
    v_micro = (ns * ps * (1 - ps)).sum() / ns.sum() ** 2 * 1e4
    return v_macro, v_micro


rows = []
for (ser, cond, ds), v in sorted(g.items()):
    seeds = sorted({r["seed"] for r in v})
    by = {(r["fold"], r["seed"]): r for r in v}
    if len(seeds) < 2 or any((f, s) not in by for f in FOLDS for s in seeds):
        continue
    # точность исполнителей (тест; каждый исп. — в тесте одного фолда), среднее по затравкам
    acc = defaultdict(list)
    for r in v:
        for sid, (a, n) in r["per_signer"].items():
            acc[int(sid)].append(a)
    p = {s: float(np.clip(np.mean(a), 0.02, 0.98)) for s, a in acc.items()}
    out = {"series": ser, "cond": cond, "ds": ds, "n_seeds": len(seeds), "family": family(v[0])}
    for key in ("val_best", "val_last", "test_macro", "test_micro"):
        Y = np.array([[100 * by[(f, s)][key] for f in FOLDS] for s in seeds])   # затравки × фолды
        S, F = Y.shape
        sd_f = Y.std(0, ddof=1)
        # двухфакторный анализ без повторов: Y = μ + a_f + b_s + e
        gm = Y.mean(); rs = Y.mean(1) - gm; cs = Y.mean(0) - gm
        E = Y - gm - rs[:, None] - cs[None, :]
        ss_seed, df_seed = F * (rs ** 2).sum(), S - 1
        ss_res, df_res = (E ** 2).sum(), (S - 1) * (F - 1)
        out[key] = {"mean": float(Y.mean()), "sd_by_fold": sd_f.tolist(), "var_within": float((sd_f ** 2).mean()),
                    "ss_seed": float(ss_seed), "df_seed": df_seed, "ss_res": float(ss_res), "df_res": df_res}
    # множитель c по исполнителям теста: Var затравок / (p(1−p)/n)
    num = den = 0.0
    for f in FOLDS:
        sig = by[(f, seeds[0])]["per_signer"].keys()
        for sid in sig:
            a = np.array([by[(f, s)]["per_signer"][sid][0] for s in seeds])
            n = by[(f, seeds[0])]["per_signer"][sid][1]
            pp = p[int(sid)]
            num += a.var(ddof=1) * 1e4
            den += pp * (1 - pp) / n * 1e4
    out["c_test_signer"] = num / den
    # биномиальные Var по фолдам: валидация и тест
    bv = {f: {part: binvar_struct(ds, f, part, p) for part in ("val", "test")} for f in FOLDS}
    out["binvar"] = bv
    # c по вал. micro и тест. macro (прямые Var затравок / биномиальные Var)
    out["c_val_micro"] = float(np.mean([out["val_last"]["sd_by_fold"][i] ** 2 / bv[f]["val"][1] for i, f in enumerate(FOLDS)]))
    out["c_test_macro"] = float(np.mean([out["test_macro"]["sd_by_fold"][i] ** 2 / bv[f]["test"][0] for i, f in enumerate(FOLDS)]))
    out["c_test_micro"] = float(np.mean([out["test_micro"]["sd_by_fold"][i] ** 2 / bv[f]["test"][1] for i, f in enumerate(FOLDS)]))
    # перевод на счёт Э2 (macro по исполнителям валидации фолда), Var в п. п.²:
    #  V: Var вал. micro (последняя точка) × [биномиальная Var macro / micro на той же валидации];
    #  T: Var тест. macro × [биномиальная Var macro валидации / macro теста];
    #  C: c_test_signer × биномиальная Var macro валидации (шум исполнителей независим).
    out["E2_var"] = {
        "V": [out["val_last"]["sd_by_fold"][i] ** 2 * bv[f]["val"][0] / bv[f]["val"][1] for i, f in enumerate(FOLDS)],
        "Vb": [out["val_best"]["sd_by_fold"][i] ** 2 * bv[f]["val"][0] / bv[f]["val"][1] for i, f in enumerate(FOLDS)],
        "T": [out["test_macro"]["sd_by_fold"][i] ** 2 * bv[f]["val"][0] / bv[f]["test"][0] for i, f in enumerate(FOLDS)],
        "C": [out["c_test_signer"] * bv[f]["val"][0] for f in FOLDS]}
    rows.append(out)


def pooled(rs, ds, fam=None):
    rs = [r for r in rs if r["ds"] == ds and (fam is None or r["family"] == fam)]
    res = {"n_cond": len(rs)}
    for key in ("val_best", "val_last", "test_macro", "test_micro"):
        # объединённая Var внутри фолда (взвешенно по df = S − 1), по фолдам
        vf = np.array([sum((r["n_seeds"] - 1) * r[key]["sd_by_fold"][i] ** 2 for r in rs) / sum(r["n_seeds"] - 1 for r in rs)
                       for i in range(5)])
        ss_s = sum(r[key]["ss_seed"] for r in rs); df_s = sum(r[key]["df_seed"] for r in rs)
        ss_e = sum(r[key]["ss_res"] for r in rs); df_e = sum(r[key]["df_res"] for r in rs)
        ms_s, ms_e = ss_s / df_s, ss_e / df_e
        tau2 = max(0.0, (ms_s - ms_e) / 5)            # главный эффект затравки (общий сдвиг по 5 фолдам)
        res[key] = {"sd_by_fold": np.sqrt(vf).round(3).tolist(), "sd_pooled": float(np.sqrt(vf.mean())),
                    "ms_seed": ms_s, "ms_res": ms_e, "df": [df_s, df_e], "F": ms_s / ms_e,
                    "share_common": tau2 / (tau2 + ms_e) if tau2 + ms_e > 0 else 0.0}
    for k in ("V", "Vb", "T", "C"):
        w = np.array([r["n_seeds"] - 1 for r in rs], float)
        res["E2_sd_" + k] = np.sqrt(np.array([[r["E2_var"][k][i] for i in range(5)] for r in rs]).T @ w / w.sum()).round(3).tolist()
    # объединённые (взвешенные по df) множители c: Σ df·Var затравок / Σ df·биномиальная Var
    w = lambda r: r["n_seeds"] - 1
    res["c_pooled"] = {
        "val_micro_last": sum(w(r) * r["val_last"]["sd_by_fold"][i] ** 2 for r in rs for i in range(5)) /
                          sum(w(r) * r["binvar"][f"fold{i + 1}"]["val"][1] for r in rs for i in range(5)),
        "val_micro_best": sum(w(r) * r["val_best"]["sd_by_fold"][i] ** 2 for r in rs for i in range(5)) /
                          sum(w(r) * r["binvar"][f"fold{i + 1}"]["val"][1] for r in rs for i in range(5)),
        "test_macro": sum(w(r) * r["test_macro"]["sd_by_fold"][i] ** 2 for r in rs for i in range(5)) /
                      sum(w(r) * r["binvar"][f"fold{i + 1}"]["test"][0] for r in rs for i in range(5)),
        "test_micro": sum(w(r) * r["test_micro"]["sd_by_fold"][i] ** 2 for r in rs for i in range(5)) /
                      sum(w(r) * r["binvar"][f"fold{i + 1}"]["test"][1] for r in rs for i in range(5)),
        "test_signer_median": float(np.median([r["c_test_signer"] for r in rs]))}
    # 95 % граница сверху доли общего сдвига по F (MS_seed/MS_res ~ (1 + 5θ)·F(df1, df2), θ = τ²/σ²)
    from scipy.stats import f as fdist
    for key in ("val_last", "test_macro"):
        df1, df2 = res[key]["df"]
        th_up = max(0.0, (res[key]["F"] / fdist.ppf(0.05, df1, df2) - 1) / 5)
        res[key]["share_common_up95"] = th_up / (1 + th_up)
        res[key]["p_seed_effect"] = float(fdist.sf(res[key]["F"], df1, df2))
    res["c_test_signer"] = float(np.median([r["c_test_signer"] for r in rs]))
    res["c_val_micro"] = float(np.median([r["c_val_micro"] for r in rs]))
    res["c_test_macro"] = float(np.median([r["c_test_macro"] for r in rs]))
    res["c_test_micro"] = float(np.median([r["c_test_micro"] for r in rs]))
    res["mean_val"] = float(np.mean([r["val_last"]["mean"] for r in rs]))
    res["mean_test_macro"] = float(np.mean([r["test_macro"]["mean"] for r in rs]))
    return res


summary = {}
for ds in ("slovo", "bukva"):
    for fam in (None, "M7/M8 (EMA или выбор по вал.)", "M1/M2/M6/E1F (старый рецепт)"):
        summary[f"{ds}|{fam or 'все'}"] = pooled(rows, ds, fam)

# Sohn, предобучение S+B: 4 затравки, одна валидация (micro)
sb = D["sohn_sb"]
for key in ("val_top1_last", "val_top1_best"):
    a = np.array([100 * x[key] for x in sb])
    pbar = a.mean() / 100
    summary["sohn_sb|" + key] = {"values": a.round(2).tolist(), "sd": float(a.std(ddof=1)),
                                  "binom_sd": float(100 * np.sqrt(pbar * (1 - pbar) / sb[0]["n_val"])),
                                  "c": float(a.var(ddof=1) / (pbar * (1 - pbar) / sb[0]["n_val"] * 1e4)), "df": len(a) - 1}

# состав валидации (для отчёта; только числа)
summary["val_composition"] = {ds: {f: {"n_signers": len(comp[ds][f]["val"]), "n_records": int(sum(comp[ds][f]["val"].values())),
                                       "min_per_signer": int(min(comp[ds][f]["val"].values())),
                                       "max_per_signer": int(max(comp[ds][f]["val"].values()))} for f in FOLDS}
                              for ds in comp}
summary["conditions"] = [{k: r[k] for k in ("series", "cond", "ds", "n_seeds", "family", "c_test_signer", "c_val_micro")} |
                         {"sd_val_last": r["val_last"]["sd_by_fold"], "sd_test_macro": r["test_macro"]["sd_by_fold"],
                          "E2_sd_V": np.sqrt(r["E2_var"]["V"]).round(3).tolist(), "E2_sd_T": np.sqrt(r["E2_var"]["T"]).round(3).tolist(),
                          "E2_sd_C": np.sqrt(r["E2_var"]["C"]).round(3).tolist()} for r in rows]
json.dump(summary, open(HERE / "noise_summary.json", "w"), ensure_ascii=False, indent=1)

for k, v in summary.items():
    if k in ("conditions",):
        continue
    print("==", k)
    print(json.dumps(v, ensure_ascii=False)[:1600])
