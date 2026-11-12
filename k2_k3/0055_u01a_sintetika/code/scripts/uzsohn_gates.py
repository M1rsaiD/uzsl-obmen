"""UzSohn этап 1 — ЦП-ворота (reports/UZSOHN_prereg.md, разд. 4). Только ЦП, без обучения; тест S_k не читается.

cd osnovnaya_model && CUDA_VISIBLE_DEVICES= PYTHONPATH=src:../uzsign_e1/src:../uzsign_ai_v2/src python scripts/uzsohn_gates.py <команда>
  u02stats       только печать (ворот U02 нет: два плеча U02-1 → U02-2 и медиана — заранее, к1 0123 п. 2, 0124 п. 2, к3 0031
                 п. 2, 0032 п. 2): КВ s_y/s_x между исполнителями (доли кадра и пиксели × медианы и средние), разброс от записи
                 к записи, |ln медианы|, КВ ширины лица после U02 с медианой и со средним → runs/UZSOHN/stats_u02.json
  u01a CUR       «место относительно лица» (к3 0015; 0130 (а), (г)) на ошибках текущей модели CUR на V_k → gate_u01a.json
                 (q, n и 99 % ДИ q — к1 0121 п. 4, 0122 п. 1, к3 0030 разд. 3); только печать — вариант (в) разд. 12 п. 5
                 (к1 0125, к3 0033): плечо U01a ставится при любом q; выход 0 / 10 — что дало бы правило q ≥ Q_MIN (цепочка
                 его не использует)
  u04 TOKENS     прототипы (NCM, косинус) Б против А на V_k, классы harf:* (к3 0018 пп. 1–3, к1 0111); правило (ii) к1 0123
                 п. 1, 0124 п. 1, к3 0031, 0032 п. 1: micro Δ ≥ 1,008 и нижняя граница 99 % ДИ > 0 (бутстреп записей в стратах V_k)
                 → gate_u04.json; 0 / 10
  facts          описательные ЦП-факты (к3 0029 п. 2.1; раскладка позы — сверкой с Э0, uzsohn_pose_check.py; локти ниже кадра
                 по исполнителям; записи harf:* на V_k) → runs/UZSOHN/facts_stage1.json (прежний — в копию с датой)
  dry            проверка кода u02stats и ворот U01a, U04 на ПЕРЕМЕШАННЫХ исполнителях / метках; значения s_y/s_x и ширины
                 лица по записям заменены случайными; числа бессмысленны и не печатаются, файлы — во временный каталог
  selftest       проверки правил на синтетике
Правила и пороги — до счёта, в предрегистрации; файлы gate_*.json не перезаписываются (повтор — сверка решения).
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UZ / "uzsign_ai_v2/scripts"))
OUT = ROOT / "runs/UZSOHN"
RUNS = ROOT / "runs/UzSohn"
SPLIT = UZ / "uzsign_ai_v2/splits/uzsl_allz_loso.json"
SEED = 48                # отбор этапа 1 (разд. 5; к1 0123 п. 20, к3 0031): прогоны runs/UzSohn/<тег>/<фолд>/seed48
THR = max(1.0, 1.0081957667939179)   # max(1; δ), δ — как uzsohn_sel.DELTA (разд. 3)
Q_MIN = 0.5              # U01a: правило «доля „место — только относительно плеч“ среди ошибок «место» ≥ 0,5 — ставить» (порог —
                         # решение к2) — (в): только печать (разд. 12 п. 5; к1 0125, к3 0033), плечо ставится при любом q
PCT = 50                 # порог «различаются» (процентиль), как [Д] разд. 3; 40 и 60 — чувствительность (0130 (г))
PCT_SENS = (40, 50, 60)
MIN_REC = 10             # группы с ≥ 10 записями (как UZSOHN_stage1_cpu_checks)
RNG = 20261111
N_BOOT, SEED_BOOT, LEVEL = 20_000, 20261106, 0.99   # бутстреп ворот — как alluz.boot (разд. 6): перцентильный, двусторонний
DRY = False


def write(name: str, res: dict) -> None:
    """runs/UZSOHN/<name>.json (gate_u01a, gate_u04, stats_u02); есть — не перезаписывается, повтор сверяет решение."""
    import m13_train as T
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    if p.is_file():                                   # повтор после перезапуска — то же решение
        old = json.loads(p.read_text())
        assert old.get("decision") == res.get("decision"), f"{p}: решение изменилось {old.get('decision')} → {res.get('decision')}"
        return
    om, v2 = T.uzsohn_code()                          # на каком коде посчитаны ворота (урок 5: коммит каждого репозитория)
    res = {**res, "code": {"osnovnaya_model": T.git(ROOT, om), "uzsign_ai_v2": T.git(UZ / "uzsign_ai_v2", v2)}}
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))


def hboot(groups, n_boot: int = N_BOOT, seed: int = SEED_BOOT, level: float = LEVEL) -> list[float]:
    """Иерархический бутстреп (свод 5.3; ДИ q ворот U01a — только печать, к1 0121 п. 4; у ворот U04 с р4 — sboot):
    исполнители с возвращением — только те, у кого есть записи, — затем записи внутри каждого с возвращением; статистика —
    Σ значений / Σ записей выборки (micro). groups — по исполнителю значения его записей. → [нижняя, верхняя] граница
    перцентильного ДИ уровня level (квантили (1 ∓ level) / 2)."""
    g = [np.asarray(v, float) for v in groups if len(v)]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    num, den = np.zeros(n_boot), np.zeros(n_boot)
    for s, v in enumerate(g):
        rows = np.nonzero(pick == s)[0]                  # реплика на каждое попадание исполнителя s (с повторами)
        np.add.at(num, rows, v[rng.integers(0, len(v), (len(rows), len(v)))].sum(1))
        np.add.at(den, rows, len(v))
    return np.quantile(num / den, [(1 - level) / 2, (1 + level) / 2]).tolist()


def sboot(groups, n_boot: int = N_BOOT, seed: int = SEED_BOOT, level: float = LEVEL) -> list[float]:
    """Бутстреп записей в стратах (ворота U04, правило (ii): к1 0123 п. 1, 0124 п. 1; к3 0031, 0032 п. 1): исполнители —
    фиксированные страты, записи внутри каждого — с возвращением; статистика — Σ значений / Σ записей (micro, знаменатель
    постоянен). → [нижняя, верхняя] граница перцентильного ДИ уровня level (квантили (1 ∓ level) / 2)."""
    g = [np.asarray(v, float) for v in groups if len(v)]
    rng = np.random.default_rng(seed)
    num = sum(v[rng.integers(0, len(v), (n_boot, len(v)))].sum(1) for v in g)
    return np.quantile(num / sum(len(v) for v in g), [(1 - level) / 2, (1 + level) / 2]).tolist()


def load_all():
    """ALL-UZ: записи после filter_nans по 118 (как Recs), поза 11–14 и H/W."""
    import m13_train as T
    recs, glosses, clips, _ = T.records("alluz", None, inp=True)
    if DRY:   # проверка кода: метки перемешаны — числа ворот бессмысленны
        glosses = list(np.random.default_rng(RNG).permutation(glosses))
    return recs, glosses, clips


def cv_between(vals, grp, how: str) -> float:
    """КВ между исполнителями: СКО (ddof 1) медиан (how = median) или средних (mean) исполнителей / |та же мера по всем
    записям|; группы с ≥ MIN_REC записями (как spread в uzsohn_stage1_cpu)."""
    by = defaultdict(list)
    for v, g in zip(vals, grp):
        if np.isfinite(v):
            by[g].append(v)
    use = [g for g, v in by.items() if len(v) >= MIN_REC]
    f = np.median if how == "median" else np.mean
    c = np.array([f(by[g]) for g in use])
    return float(c.std(ddof=1)) / abs(float(f(np.concatenate([by[g] for g in use]))))


def spread_within(vals, grp) -> tuple:
    """Разброс от записи к записи — IQR / медиана, группы ≥ MIN_REC: (медиана по исполнителям, по всем записям). По всем
    записям — от r / медиана своего исполнителя (как скептик 0141, C3): разброс между исполнителями сюда не входит, его несут
    КВ cv_between. Групп нет — (None, None)."""
    by = defaultdict(list)
    for v, g in zip(vals, grp):
        if np.isfinite(v):
            by[g].append(v)
    big = [np.asarray(v, float) for v in by.values() if len(v) >= MIN_REC]
    if not big:
        return None, None
    iqr = lambda v: float(np.subtract(*np.percentile(v, [75, 25])) / np.median(v))
    return float(np.median([iqr(v) for v in big])), iqr(np.concatenate([v / np.median(v) for v in big]))


def u02stats() -> None:
    """Только печать (ворот U02 нет: два плеча U02-1 → U02-2 и медиана — заранее, к1 0123 п. 2, 0124 п. 2, к3 0031 п. 2, 0032
    п. 2): КВ между исполнителями s_y/s_x (СКО Sohn по осям) в долях кадра и в пикселях ((s_y·H/W)/s_x) по медианам и по
    средним; разброс пиксельного s_y/s_x от записи к записи (IQR / медиана: медиана по исполнителям и по всем записям без
    весов — от r / медиана своего исполнителя, spread_within); |ln медианы s_y/s_x| в пикселях (к3 0032 п. 2) и в долях кадра
    (= |ln ρ_пикс + ln(W/H)|; раунд 3 проверки р4) — справочно: механизм U02-1 ни один уровень не задаёт (разброс множителя
    «база → U02-1» по примерам ближе к уровню в долях кадра, у жестов — к анизотропии вторых моментов; разница σ1/σ2 базы и
    U02-1 зависит от обоих и бывает < 0; разд. 4.1, угроза 12, раунд 4 проверки р4); КВ ширины лица после U02 с медианой и со
    средним → runs/UZSOHN/stats_u02.json. Ничего не решает."""
    from mpk import sohn
    from uzsohn_stage1_cpu import code
    recs, gl, clips = load_all()
    grp = [code(c["signer_id"]) for c in clips]
    if DRY:
        grp = list(np.random.default_rng(RNG).permutation(grp))
    rf, rp = [], []
    fw = {"median": [], "mean": []}
    L = sohn.LOCAL
    for i in range(len(recs)):
        x, _ = recs[i]
        asp = float(recs.aspect[i])
        _, std = sohn.center_scale(x[:, :118])
        rf.append(float(std[1] / std[0]))
        rp.append(float(std[1] * asp / std[0]))
        for st in fw:
            k, mean, s = sohn.center_scale_u(x, asp, ("u02s", "u02c"), st)
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (x[:, :118] * k - mean) / s
            w = np.linalg.norm(z[:, L[33]] - z[:, L[263]], axis=-1)
            fw[st].append(float(np.nanmedian(w)) if np.isfinite(w).any() else np.nan)
    if DRY:   # проверка кода: значения по записям случайные
        r = np.random.default_rng(RNG)
        rf, rp = list(r.lognormal(0, 0.3, len(rf))), list(r.lognormal(0, 0.3, len(rp)))
        fw = {st: list(r.lognormal(0, 0.3, len(v))) for st, v in fw.items()}
    ratio = {"frame": np.array(rf), "pixel": np.array(rp)}
    wi, al = spread_within(rp, grp)
    pf = ratio["pixel"][np.isfinite(ratio["pixel"])]
    ff = ratio["frame"][np.isfinite(ratio["frame"])]
    res = {"note": "только печать: ворот U02 нет (два плеча U02-1 → U02-2, медиана — заранее; к1 0124 п. 2, к3 0032 п. 2); "
                   f"КВ между исполнителями — группы ≥ {MIN_REC} записей; IQR / медиана пиксельного s_y/s_x от записи к записи "
                   f"(группы ≥ {MIN_REC} записей) — медиана по исполнителям и по всем записям без весов (r / медиана своего "
                   "исполнителя: без разброса между исполнителями, 0141 C3)",
           "cv_sy_sx": {f"{u}_{h}": cv_between(v, grp, h) for u, v in ratio.items() for h in ("median", "mean")},
           "cv_face_width_u02": {st: cv_between(np.array(v), grp, "median") for st, v in fw.items()},
           "sy_sx_median_frame": float(np.median(rf)), "sy_sx_median_pixel": float(np.median(pf)), "n_records": len(rf),
           "pixel_iqr_over_median_within_signer_median": wi,
           "pixel_iqr_over_median_all_records": al, "abs_ln_median_pixel": float(abs(np.log(np.median(pf)))),
           "abs_ln_median_frame": float(abs(np.log(np.median(ff))))}
    write("stats_u02", res)
    if not DRY:
        print(json.dumps({k: res[k] for k in ("cv_sy_sx", "pixel_iqr_over_median_all_records", "abs_ln_median_pixel",
                                              "abs_ln_median_frame")}, ensure_ascii=False))


# ---------------------------------------------------------------- общая часть U01a / U04: фолды отбора
def folds_sel(glosses, clips):
    """→ {фолд: (обучение отбора, записи V_k для счёта, V_k)} — как m13_train --inner-val (alluz)."""
    import m13_train as T
    sp = json.loads(SPLIT.read_text())["folds"]
    pos = {c["sample_id"]: i for i, c in enumerate(clips)}
    folds = sorted(sp)
    out = {}
    for f in folds:
        vk = T.inner_val(f, folds)
        tr = [pos[s] for s in sp[f]["train"] + sp[f]["val"]]
        vt = set(sp[vk]["test"])
        train = [i for i in tr if clips[i]["signer_id"] != vk]
        cls = {glosses[i] for i in train}
        te = [i for i in tr if clips[i]["sample_id"] in vt and glosses[i] in cls]
        assert te and all(clips[i]["signer_id"] == vk for i in te)
        out[f] = (train, te, vk)
    return out


def ref_pairs(n_cls: int, rng) -> np.ndarray:
    """20 000 случайных пар классов (без пар класса с собой) — ОДНИ на фолд для всех трёх пространств: иначе процентили одной
    пары в разных пространствах сравнивались бы с разными опорными выборками (ложные «только плечи», ревизия раунд 1)."""
    rp = rng.integers(0, n_cls, (20000, 2))
    return rp[rp[:, 0] != rp[:, 1]]


def pct_ref(means: dict, rp: np.ndarray) -> np.ndarray:
    cl = sorted(means)
    return np.sort([np.linalg.norm(means[cl[a]] - means[cl[b]]) for a, b in rp])


def pct(ref: np.ndarray, d: float) -> float:
    return 100 * float(np.searchsorted(ref, d)) / len(ref)


def place_feats(recs) -> dict:
    """Запись → (форма доминантной кисти 42, место от носа / ширина плеч 2, место от носа / ширина лица 2); пиксели (y · H/W),
    левая кисть — зеркально по x; средние по кадрам, где есть всё нужное (как UZSOHN_diagnostics разд. 3, но y · H/W)."""
    from mpk import sohn
    L = sohn.LOCAL
    out = {}
    for i in range(len(recs)):
        x, _ = recs[i]
        X = x * np.array([1.0, recs.aspect[i]], np.float32)
        nl, nr = np.isfinite(X[:, sohn.LH_LOC[0], 0]).sum(), np.isfinite(X[:, sohn.RH_LOC[0], 0]).sum()
        if nl + nr == 0:
            continue
        H, sgn = (sohn.LH_LOC, -1.0) if nl > nr else (sohn.RH_LOC, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = X[:, H]
            rel = (q - q[:, :1]) / np.linalg.norm(q[:, 9] - q[:, 0], axis=-1)[:, None, None] * [sgn, 1]
            v = X[:, H[0]] - X[:, L[1]]
            sw = np.linalg.norm(X[:, 118] - X[:, 119], axis=-1)
            fw = np.linalg.norm(X[:, L[33]] - X[:, L[263]], axis=-1)
            ps, pf = v / sw[:, None] * [sgn, 1], v / fw[:, None] * [sgn, 1]
        m = np.isfinite(rel).all((1, 2)) & np.isfinite(ps).all(1) & np.isfinite(pf).all(1)
        if m.any():
            out[i] = (rel[m].mean(0).ravel(), ps[m].mean(0), pf[m].mean(0))
    return out


def u01a_rule(n_place: int, n_only: int) -> tuple:
    """q = доля ошибок «место», где место различимо только относительно плеч (не относительно лица); правило — «ставить», если
    q ≥ Q_MIN; ошибок «место» нет — «не ставить» (как у Sohn; решение к2). (в): только печать (разд. 12 п. 5; к1 0125, к3 0033) —
    плечо U01a ставится при любом q, decision — лишь то, что дало бы правило: мерой оно не служит (ворота вырождены, 4.2) — на
    синтетике (runs/UZSOHN/revision_r1/u01a_gate_synth.py; 20–400 реализаций — revision_r4/fix_r4/evidence/u01a_q_seeds/,
    fix_r5/evidence/u01a_q_r5/) закрывается (равномерный состав ошибок — медиана q ≈ 0,01–0,04 при 6 / 2 записях на класс; сгущение
    к порогу 50 с τ ≥ 2 — q < 0,5 во всех реализациях, максимум 0,34 при 6 записях и 0,41 при 2), а если почти все ошибки «место» у
    самого порога 50 — q ≈ 0,5, решение случайно."""
    q = n_only / n_place if n_place else None
    return q, "run" if q is not None and q >= Q_MIN else "skip"


def u01a_ci(per: dict) -> list[float] | None:
    """99 % ДИ q (к1 0121 п. 4, 0122 п. 1; к3 0030 разд. 3) по ошибкам «место» при пороге PCT: hboot по V_k, затем ошибкам."""
    g = [np.r_[np.ones(v[f"only_shoulders_{PCT}"]), np.zeros(v[f"place_{PCT}"] - v[f"only_shoulders_{PCT}"])] for v in per.values()]
    return hboot(g) if any(len(x) for x in g) else None


def cur_rows(cur: str, f: str, te: list, gl: list, clips: list) -> list[dict]:
    if DRY:   # проверка кода: «ответы» — случайные классы обучения
        rng = np.random.default_rng([RNG, len(te)])
        cls = sorted(set(gl))
        return [{"sample_id": clips[i]["sample_id"], "gloss": gl[i], "pred": cls[int(rng.integers(len(cls)))], "correct": "0"}
                for i in te]
    with open(RUNS / cur / f / f"seed{SEED}/predictions.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def u01a(cur: str) -> None:
    recs, gl, clips = load_all()
    F = place_feats(recs)
    pos = {c["sample_id"]: i for i, c in enumerate(clips)}
    tot = {t: [0, 0] for t in PCT_SENS}                    # порог → [ошибок «место», из них «только плечи»]
    n_err, per = 0, {}
    for f, (train, te, vk) in folds_sel(gl, clips).items():
        by = defaultdict(list)
        for i in train:
            if i in F:
                by[gl[i]].append(F[i])
        means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
        assert sorted(means[0]) == sorted(means[1]) == sorted(means[2])
        rp = ref_pairs(len(means[0]), np.random.default_rng(RNG))
        refs = [pct_ref(m, rp) for m in means]
        rows = cur_rows(cur, f, te, gl, clips)
        assert {pos[r["sample_id"]] for r in rows} == set(te), f"{cur} {f}: записи V_k ≠ правилу отбора"
        e, cnt = 0, {t: [0, 0] for t in PCT_SENS}
        for r in rows:
            if r["correct"] == "1":
                continue
            e += 1
            t_, p_ = r["gloss"], r["pred"]
            if t_ not in means[0] or p_ not in means[0]:
                continue
            ps, pp_s, pp_f = (pct(refs[k], float(np.linalg.norm(means[k][t_] - means[k][p_]))) for k in range(3))
            for t in PCT_SENS:
                if pp_s >= t and ps < t:
                    cnt[t][0] += 1
                    cnt[t][1] += pp_f < t
        per[f] = {"vk": vk, "errors": e, **{f"place_{t}": cnt[t][0] for t in PCT_SENS}, **{f"only_shoulders_{t}": cnt[t][1] for t in PCT_SENS}}
        n_err += e
        for t in PCT_SENS:
            tot[t][0] += cnt[t][0]
            tot[t][1] += cnt[t][1]
    sens = {t: dict(zip(("place_errors", "place_only_shoulders"), tot[t]), q=u01a_rule(*tot[t])[0], decision=u01a_rule(*tot[t])[1])
            for t in PCT_SENS}
    q, go = u01a_rule(*tot[PCT])
    res = {"role": f"только печать: вариант (в) разд. 12 п. 5 (к1 0125, к3 0033) — плечо U01a ставится при любом q; decision — что "
                   f"дало бы правило q ≥ {Q_MIN} (ворота вырождены, 4.2)",
           "rule": f"ошибки CUR на V_k; «место» — место от носа / ширина плеч ≥ {PCT}-го процентиля и форма < {PCT} (как разд. 3 "
                   f"диагностики, y·H/W; классы — средние по обучению отбора фолда; 20 000 случайных пар, общих для трёх пространств); q — доля из них, где место "
                   f"от носа / ширина лица < {PCT}; правило (только печать, вариант (в)): «ставить», если q ≥ {Q_MIN}; ошибок «место» нет — «не ставить»; "
                   "пороги 40 и 60 — только чувствительность (правило — по 50); q, n и 99 % ДИ q — только печать", "current": cur, "errors": n_err,
           "place_errors": tot[PCT][0], "place_only_shoulders": tot[PCT][1], "q": q, "n": tot[PCT][0],
           "q_ci99": u01a_ci(per), "q_ci_method": f"иерархический бутстреп: V_k с ошибками «место» с возвращением, затем ошибки внутри; "
                                                f"{N_BOOT}, default_rng({SEED_BOOT}), перцентильный {LEVEL:.0%}; ничего не решает",
           "sensitivity": {str(t): v for t, v in sens.items()}, "stable_40_60": len({v["decision"] for v in sens.values()}) == 1,
           "per_fold": per, "decision": go}
    write("gate_u01a", res)
    if not DRY:
        print(json.dumps({k: res[k] for k in ("errors", "place_errors", "place_only_shoulders", "q", "q_ci99", "stable_40_60",
                                              "decision")}, ensure_ascii=False))
    sys.exit(0 if go == "run" else 10)   # что дало бы правило; цепочка у U01a код не использует (плечо всегда, вариант (в))


def ncm_top1(Xtr, ytr, Xte, yte) -> np.ndarray:
    cls = sorted(set(ytr))
    P = np.stack([Xtr[np.array(ytr) == c].mean(0) for c in cls])
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    Z = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-12)
    return np.array(cls)[(Z @ P.T).argmax(1)] == np.array(yte)


def u04_rule(per: dict) -> tuple:
    """Правило (ii) к1 0123 п. 1, 0124 п. 1 и к3 0031, 0032 п. 1 (вместо (б) с иерархическим бутстрепом: при 87 % записей у
    двух исполнителей нижнюю границу задавали перевыборки без них; вместо «≥ 6 из 10 V_k» к3 0018 п. 4 — harf:* лишь у 6 V_k):
    micro по записям harf:* V_k — Δ (Б − А) top-1 (на одних записях) ≥ THR и нижняя граница 99 % ДИ > 0 (sboot: V_k с
    записями harf:* — страты, записи внутри — с возвращением). Исключение из свода 5.3: ворота — фильтр механизма; решение о
    шаге — отбор на ГП по всем классам (≥ 6/10 V_k); перенос механизма букв на исполнителей не проверяют ни отбор, ни
    подтверждение — буквы и на V_k, и на S_k на 87 % у пилота (разд. 4.3). per[фолд]: n_harf, harf_b_only (Б верно, А нет),
    harf_a_only (А верно, Б нет). → (Δ п. п., [ДИ п. п.], записей, run|skip)."""
    g = [np.r_[np.ones(r["harf_b_only"]), -np.ones(r["harf_a_only"]), np.zeros(r["n_harf"] - r["harf_b_only"] - r["harf_a_only"])]
         for r in per.values() if r["n_harf"]]
    if not g:
        return None, None, 0, "skip"
    n = sum(len(v) for v in g)
    diff = 100 * float(sum(v.sum() for v in g)) / n
    ci = [100 * c for c in sboot(g)]
    return diff, ci, n, "run" if diff >= THR and ci[0] > 0 else "skip"


def u04(tokens: str) -> None:
    from mpk import sohn
    tok = tuple(t for t in sohn.INPUT_TOKENS if t in tokens.split(","))
    assert set(tokens.split(",")) - {""} == set(tok) and "u04" not in tok, tokens
    recs, gl, clips = load_all()
    A, B = [], []
    pr, cz = [0, 0], [0, 0]       # печать (к1 0123 пп. 15, 19): кадры с |9 − 0| < 0,5 медианы записи; 1 + cos = 0 у видимой кисти
    for i in range(len(recs)):
        x, _ = recs[i]
        k, mean, std = sohn.center_scale_u(x, float(recs.aspect[i]), tok)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (x[:, :118] * k - mean) / std
        X = x[:, :118] * np.array([1.0, recs.aspect[i]], np.float32)
        hf = sohn.hand_feats(X)
        a, b = [], []
        for h, H in enumerate((sohn.LH_LOC, sohn.RH_LOC)):
            m = np.isfinite(x[:, H]).all((1, 2))
            a.append(z[m][:, H].reshape(m.sum(), -1).mean(0) if m.any() else np.full(42, np.nan))
            b.append(hf[m, 225 * h:225 * (h + 1)].mean(0) if m.any() else np.full(225, np.nan))
            if m.any():
                pal = np.linalg.norm(X[m, H[9]] - X[m, H[0]], axis=-1)
                pr[0] += int((pal < 0.5 * np.median(pal)).sum())
                pr[1] += len(pal)
                cz[0] += int((hf[m, 225 * h + 210:225 * (h + 1)] == 0).sum())
                cz[1] += 15 * int(m.sum())
        A.append(np.concatenate(a))
        B.append(np.concatenate(a + b))
    A, B = np.array(A), np.array(B)
    import m13_train as T
    codes = json.loads(T.CODES.read_text())["signers"]
    per = {}
    for f, (train, te, vk) in folds_sel(gl, clips).items():
        hm = np.array([gl[i].startswith("harf:") for i in te])
        row = {"vk": vk, "n_harf": int(hm.sum()), "n_all": len(te)}
        ok = {}
        for nm, X in (("A", A), ("B", B)):
            mu, sd = np.nanmean(X[train], 0), np.nanstd(X[train], 0)
            Z = np.nan_to_num((X - mu) / np.where(sd > 0, sd, 1))      # z по обучению отбора; нет кисти — среднее (0)
            ok[nm] = ncm_top1(Z[train], [gl[i] for i in train], Z[te], [gl[i] for i in te])
            row[f"{nm}_all"] = 100 * float(ok[nm].mean())
            row[f"{nm}_harf"] = 100 * float(ok[nm][hm].mean()) if hm.any() else None
        row["harf_b_only"] = int((ok["B"] & ~ok["A"])[hm].sum())
        row["harf_a_only"] = int((ok["A"] & ~ok["B"])[hm].sum())
        per[f] = row
    diff, ci, n_h, go = u04_rule(per)
    pilot = sum(r["n_harf"] for r in per.values() if codes.get(r["vk"]) in ("S11", "S12"))
    res = {"rule": f"NCM (косинус, z по обучению отбора): Б (A + 210 расст. / медиана ладони записи + 15 значений 1 + cos углов на "
                   f"кисть) − A (21 точка кисти в нормировке {list(tok) or 'Sohn'}), top-1 на записях V_k классов harf:* (прототипы — "
                   f"все классы обучения отбора); правило (ii) к1 0123 п. 1, 0124 п. 1, к3 0031, 0032 п. 1: micro по всем записям "
                   f"harf:* V_k, ставить, если Δ ≥ {THR:.3f} п. п. и нижняя граница {LEVEL:.0%} ДИ > 0 (бутстреп записей в стратах "
                   f"V_k: исполнители фиксированы, записи внутри — с возвращением; {N_BOOT}, default_rng({SEED_BOOT}), перцентильный "
                   "двусторонний; исключение из свода 5.3 — ворота фильтр механизма, решение о шаге — отбор на ГП по всем классам; "
                   "перенос механизма букв на исполнителей не проверяют ни отбор, ни подтверждение — буквы в основном у пилота, "
                   "harf_share_S11_S12); "
                   "доля S11 + S12, все классы, кадры с ладонью < 0,5 медианы и нули 1 + cos — справочно", "tokens": list(tok),
           "per_fold": per, "diff_harf": diff, "ci99_harf": ci, "n_harf": n_h, "level": LEVEL,
           "harf_share_S11_S12": pilot / n_h if n_h else None,
           "diff_all_ref": float(np.mean([r["B_all"] - r["A_all"] for r in per.values()])),
           "palm_frames_lt_half_median": pr[0] / pr[1] if pr[1] else None, "one_plus_cos_zero_visible": cz[0] / cz[1] if cz[1] else None,
           "decision": go}
    write("gate_u04", res)
    if not DRY:
        print(json.dumps({k: res[k] for k in ("diff_harf", "ci99_harf", "n_harf", "harf_share_S11_S12", "diff_all_ref",
                                              "palm_frames_lt_half_median", "one_plus_cos_zero_visible", "decision")}, ensure_ascii=False))
    sys.exit(0 if go == "run" else 10)


def facts() -> None:
    """Описательные факты данных (не выбор, не ворота): кадры ALL-UZ, где пусты все 118 точек Sohn, а поза видна (к3 0029
    п. 2.1); раскладка позы — прямой сверкой с Э0 553 (uzsohn_pose_check.pose_e0: кэш = Э0[:, 11:15], не бёдра 23/24, запястья
    15/16 у кистей 33/54); доли x11 > x12 и y13 > y11 — только справочно (локти от бёдер не отличают); локти ниже кадра по
    исполнителям (uzsohn_pose_check.elbows, угроза 7); кадры позы = кадры Sohn (запястья МПК = Sohn); записи V_k и harf:* на V_k."""
    import time

    import alluz
    import m13_build
    import uzsohn_pose_check
    from mpk import data, sohn
    from uzsohn_stage1_cpu import source
    S = m13_build.load("alluz")
    A = data.load(alluz.CACHE)
    with np.load(ROOT / "cache/UZSOHN/alluz_pose.npz", allow_pickle=False) as z:
        P, po = z["xy"], z["offsets"]
    so = S["offsets"]
    assert np.array_equal(po, so) and [c["sample_id"] for c in A["clips"]] == [c["sample_id"] for c in S["clips"]]
    assert data.POSE_MP[:4] == (11, 12, 13, 14) and (data.MPK_POINTS[6], data.MPK_POINTS[27]) == (33, 54)
    empty = np.isnan(S["xy"]).all((1, 2))
    p4, p1 = np.isfinite(P).all((1, 2)), np.isfinite(P).any((1, 2))
    src = np.repeat([source(c["signer_id"]) for c in S["clips"]], np.diff(so))
    by = {}
    for s in sorted(set(src)):
        m = src == s
        ok = m & p4
        by[s] = {"frames": int(m.sum()), "all118_empty": int((m & empty).sum()), "all118_empty_pose4": int((m & empty & p4).sum()),
                 "all118_empty_pose_any": int((m & empty & p1).sum()),
                 "x11_gt_x12": round(float((P[ok, 0, 0] > P[ok, 1, 0]).mean()), 4),
                 "y13_gt_y11": round(float((P[ok, 2, 1] > P[ok, 0, 1]).mean()), 4),
                 "y14_gt_y12": round(float((P[ok, 3, 1] > P[ok, 1, 1]).mean()), 4)}
    worst = 0.0
    for i in range(len(S["clips"])):                       # кадр позы ↔ кадр Sohn: запястья кэша МПК (to30) = запястья Sohn
        a0, a1 = A["offsets"][i], A["offsets"][i + 1]
        f = m13_build.to30(a1 - a0, float(S["fps_src"][i]))
        xy, pr, sx = A["xy"][a0:a1][f].astype(np.float32), A["present"][a0:a1][f], S["xy"][so[i]:so[i + 1]]
        for mp, loc, part in ((6, sohn.LH_LOC[0], 1), (27, sohn.RH_LOC[0], 2)):
            mm = pr[:, part] & np.isfinite(sx[:, loc, 0])
            if mm.any():
                worst = max(worst, float(np.abs(xy[mm, mp] - sx[mm, loc]).max()))
    import m13_train as T
    recs, gl, clips = load_all()
    codes = json.loads(T.CODES.read_text())["signers"]
    vks = {codes[vk]: {"S_k": codes[f], "n_vk_test": len(te), "n_vk_harf": sum(gl[i].startswith("harf:") for i in te)}
           for f, (train, te, vk) in folds_sel(gl, clips).items()}
    res = {"note": "описательная статистика данных (не выбор и не ворота); кадры — cache/M13/alluz.npz до filter_nans, поза — "
                   "cache/UZSOHN/alluz_pose.npz (МПК 0–3 = поза MediaPipe 11–14, Э0 553: поза 0–32)",
           "frames_total": int(len(empty)), "frames_all118_empty": int(empty.sum()),
           "frames_all118_empty_pose4": int((empty & p4).sum()), "frames_all118_empty_pose_any": int((empty & p1).sum()),
           "by_source": by, "by_source_note": "x11_gt_x12, y13_gt_y11, y14_gt_y12 — справочно: локти от бёдер не отличают; "
                                               "раскладка — pose_vs_e0", "wrist_mpk_vs_sohn_max_abs": worst, "vk": vks,
           "pose_vs_e0": uzsohn_pose_check.pose_e0(), "elbows_by_signer": uzsohn_pose_check.elbows()}
    assert all(v["sohn_eq_e0"] and v["nan_match"] for v in res["pose_vs_e0"].values()), "кэш ≠ Э0 (раскладка, разд. 10)"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "facts_stage1.json"
    if p.is_file():   # копии вместо перезаписи (правило автора)
        p.rename(p.with_name(f"facts_stage1_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("frames_total", "frames_all118_empty", "frames_all118_empty_pose4",
                                          "frames_all118_empty_pose_any", "wrist_mpk_vs_sohn_max_abs")}, ensure_ascii=False))


def dry() -> None:
    """Код u02stats (перемешанные исполнители) и ворот U01a, U04 (перемешанные метки), выход — во временный каталог; печать — «ok»."""
    global OUT, DRY
    OUT, DRY = Path(tempfile.mkdtemp(prefix="uzsohn_gates_dry_")), True
    u02stats()
    for fn, args in ((u01a, ("dry",)), (u04, ("u02s,u02c",))):
        try:
            fn(*args)
        except SystemExit as e:
            assert e.code in (0, 10), e.code
    assert all((OUT / f"{g}.json").is_file() for g in ("stats_u02", "gate_u01a", "gate_u04"))
    gu = json.loads((OUT / "gate_u01a.json").read_text()); assert "только печать" in gu["role"] and "решени" not in gu["rule"] + gu["q_ci_method"]   # (в), 12 п. 5
    print(f"gates dry ok ({OUT})")


def selftest() -> None:
    # бутстреп ворот закреплён до счёта (4.3, разд. 12 п. 1): мутации уровня 0,95 / 0,90 поведенческие проверки ниже пропускают;
    # действующий уровень и статистику (ii) закрепляют два случая с известным ответом ниже (раунд 3 проверки р4)
    assert (N_BOOT, SEED_BOOT, LEVEL) == (20_000, 20261106, 0.99), (N_BOOT, SEED_BOOT, LEVEL)
    rng = np.random.default_rng(0)
    ref = np.sort(rng.random(1000))
    assert abs(pct(ref, 0.5) - 50) < 5 and pct(ref, -1) == 0 and pct(ref, 2) == 100
    Xtr = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], float)
    ok = ncm_top1(Xtr, ["a", "a", "b", "b"], np.array([[1, 0.2], [0.2, 1]]), ["a", "a"])
    assert list(ok) == [True, False]
    g = [f"s{k}" for k in range(4) for _ in range(12)]
    v = np.array([1.0 + 0.01 * k for k in range(4) for _ in range(12)])
    assert abs(cv_between(v, g, "median") - np.std([1.0, 1.01, 1.02, 1.03], ddof=1) / 1.015) < 1e-9
    assert np.isclose(cv_between(v, g + ["x"], "mean"), cv_between(v, g, "mean"))       # группа < 10 записей не входит
    # разброс от записи к записи: различие уровней исполнителей в «по всем записям» не входит (r / медиана исполнителя)
    assert spread_within(v, g) == (0.0, 0.0) and spread_within(v[:5], g[:5]) == (None, None)
    s12 = float(np.subtract(*np.percentile(np.arange(1, 13), [75, 25])) / 6.5)
    assert np.allclose(spread_within(np.r_[np.arange(1, 13), 10 * np.arange(1, 13)], ["a"] * 12 + ["b"] * 12), (s12, s12))
    assert u01a_rule(10, 5) == (0.5, "run") and u01a_rule(10, 4)[1] == "skip" and u01a_rule(0, 0) == (None, "skip")
    m1 = {f"c{k}": rng.random(2) for k in range(30)}                    # пропорциональные пространства (×2 точно в float):
    m2 = {c: 2 * v for c, v in m1.items()}                              # с общими парами процентиль любой пары одинаков
    rp = ref_pairs(30, np.random.default_rng(RNG))
    r1, r2 = pct_ref(m1, rp), pct_ref(m2, rp)
    assert all(pct(r1, float(np.linalg.norm(m1[a] - m1[b]))) == pct(r2, float(np.linalg.norm(m2[a] - m2[b])))
               for a in m1 for b in m1 if a != b)
    # hboot: одинаковые значения — точка; затравка — тот же ДИ; иерархия: исполнитель из одной записи «весит» как остальные
    assert hboot([np.ones(5), np.ones(3)]) == [1.0, 1.0] and hboot([np.r_[1, 0, 0], np.r_[0, 1]]) == hboot([np.r_[1, 0, 0], np.r_[0, 1]])
    lo_r, _ = hboot([np.ones(100)] + [np.zeros(1)] * 5, n_boot=4000)          # (5/6)^6 = 33 % выборок без «единичного» исполнителя
    assert lo_r == 0.0
    assert abs(hboot([np.r_[np.ones(30), np.zeros(70)]] * 6, n_boot=4000)[0] - 0.3) < 0.07   # 600 записей, p = 0,3: ±2,58·σ ≈ ±0,05
    # sboot (ворота U04, (ii)): страты фиксированы — исполнитель из одной записи весит одну запись; знаменатель постоянен
    assert np.allclose(sboot([np.ones(100)] + [np.zeros(1)] * 5, n_boot=4000), 100 / 105)
    assert abs(sboot([np.r_[np.ones(30), np.zeros(70)]] * 6, n_boot=4000)[0] - 0.3) < 0.07
    row = lambda n, b, a: {"n_harf": n, "harf_b_only": b, "harf_a_only": a}
    # эффект у двух крупных V_k (как пилот S11, S12): (ii) открывается, иерархический (б) — нет (перевыборки без них — Δ = 0)
    pil = {"a": row(143, 20, 5), "b": row(114, 15, 4), "c": row(22, 0, 0), "d": row(11, 0, 0), "e": row(3, 0, 0), "f": row(1, 0, 0)}
    gp = [np.r_[np.ones(r["harf_b_only"]), -np.ones(r["harf_a_only"]), np.zeros(r["n_harf"] - r["harf_b_only"] - r["harf_a_only"])]
          for r in pil.values()]
    assert u04_rule(pil)[3] == "run" and hboot(gp)[0] <= 0
    big = {str(k): row(50, 10, 0) for k in range(6)}                                         # +20 п. п. у всех 6 V_k
    d, ci, n, go = u04_rule({**big, **{str(k): row(0, 0, 0) for k in range(6, 10)}})
    assert go == "run" and n == 300 and abs(d - 20) < 1e-9 and 0 < ci[0] < 20 < ci[1]
    assert u04_rule({"0": row(3, 1, 0)})[3] == "skip"                    # Δ = +33 п. п., но ДИ от 0 (одна запись решает) — нет
    assert u04_rule({"0": row(1000, 10, 0)})[0] == 1.0 and u04_rule({"0": row(1000, 10, 0)})[3] == "skip"   # +1,00 < 1,008
    assert u04_rule({str(k): row(0, 0, 0) for k in range(10)}) == (None, None, 0, "skip")
    # действующий ДИ и статистика (ii), а не только константы (раунд 3 проверки р4): квантили [1 − level, level] (98 % / односторонний)
    # и macro по стратам меняют решение; «только страты ≥ 10» решения не меняет — ловят n == 294 и Δ; ДИ — точно, на решётке 1/294
    # (раунд 4; ловит многие n_boot / seed, но не все: 40 000, 129 000 и ≈ 2,5–4,6 % затравок дают тот же ДИ — раунд 5 проверки р4),
    # поэтому параметры закреплены прямо: умолчания sboot / hboot = константам, которые пишет rule в gate_u04.json, и u04_rule
    # вызывает sboot без переопределений (вызов записывается)
    import inspect
    assert all([inspect.signature(f).parameters[k].default for k in ("n_boot", "seed", "level")] == [N_BOOT, SEED_BOOT, LEVEL]
               for f in (sboot, hboot))
    z4 = {"c": row(22, 0, 0), "d": row(11, 0, 0), "e": row(3, 0, 0), "f": row(1, 0, 0)}
    calls, real = [], sboot
    globals()["sboot"] = lambda g, *a, **k: calls.append((a, k)) or real(g, *a, **k)
    try:
        d, ci, n, go = u04_rule({"a": row(143, 24, 6), "b": row(114, 12, 12), **z4})   # net +18 у одной страты; [0,5; 99,5] [−0,34; +12,24]
    finally:
        globals()["sboot"] = real
    assert calls == [((), {})], calls
    assert go == "skip" and n == 294 and abs(d - 100 * 18 / 294) < 1e-9 and np.allclose(ci, [-100 / 294, 3600 / 294])
    d, ci, n, go = u04_rule({"a": row(143, 25, 5), "b": row(114, 20, 5), **{k: row(v["n_harf"], 0, 1) for k, v in z4.items()}})
    assert go == "run" and n == 294 and abs(d - 100 * 31 / 294) < 1e-9 and np.allclose(ci, [1300 / 294, 5000 / 294])   # micro Σ / 294 (macro — Δ −19,97)
    pu = {"a": {"place_50": 4, "only_shoulders_50": 2}, "b": {"place_50": 0, "only_shoulders_50": 0}}
    ci = u01a_ci(pu)
    assert ci is not None and ci[0] < 0.5 < ci[1] and u01a_ci({"a": {"place_50": 0, "only_shoulders_50": 0}}) is None
    print("gates selftest ok")


if __name__ == "__main__":
    c = sys.argv[1]
    {"u02stats": u02stats, "u01a": lambda: u01a(sys.argv[2]), "u04": lambda: u04(sys.argv[2]),
     "facts": facts, "dry": dry, "selftest": selftest}[c]()
