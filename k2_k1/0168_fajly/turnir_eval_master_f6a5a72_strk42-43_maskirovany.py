"""Турнир архитектур + UzSohn — единый стенд оценки (reports/TURNIR_prereg.md; письма к1 0059, 0062).

cd osnovnaya_model
python scripts/turnir_eval.py --selftest            # синтетика: статистика, ансамбли, формат FORMAT.md (0062)
python scripts/turnir_eval.py inventory             # → reports/TURNIR_inventory.md (что есть, чего нет, ч ГП)
CUDA_VISIBLE_DEVICES= python scripts/turnir_eval.py export [--only A1] [--dataset Slovo]   # вероятности готовых моделей (ЦП)
python scripts/turnir_eval.py check                 # сверка стенда с m11_tables.md / m8_tables.md / M12_slgcn_loso_tables.md
python scripts/turnir_eval.py summary               # → runs/TURNIR/turnir_summary.csv (FORMAT.md) + runs/TURNIR/pairs.md

Формат предсказаний (один для всех моделей): runs/TURNIR/preds/<набор>/<модель>/<рецепт>/<условие>/<фолд>__seed<k>.npz
  ids [N] (sample_id), probs [N, C] float16 (softmax), classes [C], signer [N], gloss [N] (истинная метка), role [N]
  ('test' | 'val'), run (откуда); рядом metrics.json (top-1 теста каждого файла, сверка с исходником).
  Корпус (LOSO): signer = исполнитель теста фолда (как loso_slgcn_tables.py); фолд «other» в macro не входит.
Статистика: бутстреп по исполнителям (исполнители отсортированы); протокол П — 2 000, default_rng(0) (M9.1/m11_tables),
  корпус — 10 000, default_rng(20261104) (loso_slgcn_tables). micro — ДИ отношения сумм, macro — среднее точностей
  исполнителей. Одиночная = среднее по затравкам на запись; однородный ансамбль N = среднее softmax N затравок,
  точность усреднена по всем C(n, N) сочетаниям; смешанный — состав по валидации, затравки выровнены по номеру.
  Уровень главных ДИ: 1 − 0,05/m (не ниже 99 %), m — число главных пар семейства.
Персональные данные (ids, исполнители) — только в runs/ (не в git).
"""
import argparse
import csv
import glob
import hashlib
import itertools
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent / "uzsign_ai_v2"
RUNS = ROOT / "runs"
OUT = RUNS / "TURNIR"
PRED = OUT / "preds"
M10C = [RUNS / "M10/cache", UZ / "runs/M10/cache"]
FOLDS = {"Slovo": [f"fold{k}" for k in range(1, 6)], "Bukva": [f"fold{k}" for k in range(1, 6)],
# [строка 42 замаскирована при отправке: имена фолдов — правило обмена (только S-коды); в блобе c4e557d — определение фолдов]
# [строка 43 замаскирована — то же]
BOOT = {"Slovo": (2000, 0), "Bukva": (2000, 0), "UzSL": (10000, 20261104)}  # справочные строки и сверка с прежними таблицами
SOHN_GAP = 2.71  # недобор порта Sohn на Э0 (asl-signs: наша средняя 5 фолдов − кривая ноутбука, −2,71 п. п.; 0078 п. 3)
SOHN_LIKE = ("Sohn", "UzSohn")
B_MAIN = 20000  # главные пары и ДИ сводки: крайние квантили (1 − 0,01/m) — 20 000 повторов (0063, п. 6); затравка ГСЧ — та же
# главная строка архитектуры — её лучший рецепт по валидации (0063 п. 2): SL-GCN, ST-GCN — общий (Г2/Г3), остальные — родной
MAIN_RECIPE = {"SL-GCN": "shared", "ST-GCN": "shared"}
N_SEEDS = 5
FMT = ["dataset", "model", "kind", "members", "recipe", "condition", "top1", "top1_lo", "top1_hi", "ci_level", "gflops",
       "latency_ms", "params_m", "n_seeds", "source"]

# ── Ячейки турнира: (набор, модель, рецепт, условие) → источники готовых прогонов и цена одного прогона в ч ГП ─────────
# kind: run — каталоги mpk.train (config.yaml + best.pt; вероятности — из кэша M10 или выгрузкой на ЦП);
#       samslr / stgcn — их код (M11 Г/Г3; оценки теста сохранены их main.py: eval_results/*.pkl, test_result.pkl);
#       m13 — порт Sohn (серия M13, другой агент; формат его вывода подключается после M13).
# h — ч ГП на один (фолд, затравка): журнал runs/journal.csv (секунды / jobs, медиана) или длительность серии / число
#     прогонов; «~» — оценка по аналогу (см. reports/TURNIR_inventory.md). pre — ч ГП одного предобучения S+B (на затравку).
P, LO = "Slovo", "UzSL"
C = []


def cell(ds, model, recipe, cond, kind, src, h, note="", pre=None, main=True):
    src = src + [f"TURNIR/train/{ds}/{model}/{recipe}/{cond}/*/seed*"]  # новые затравки турнира (scripts/turnir_train.py)
    C.append(dict(ds=ds, model=model, recipe=recipe, cond=cond, kind=kind, src=src, h=h, note=note, pre=pre, main=main))


V2 = "../../uzsign_ai_v2/runs"
for ds, low in (("Slovo", "slovo"), ("Bukva", "bukva")):
    S = ds == "Slovo"
    cell(ds, "A1", "native", "scratch", "run",
         ["M7/slovo_a_ema/fold*/seed*"] if S else ["M7/bukva_a_ema/fold*/seed*"],
         0.068 if S else 0.010, "рецепт A1 = общий (наш); Slovo з. 0 (M3, иной cache/slovo.npz) — переобучается (0063 п. 1)")
    cell(ds, "A2", "native", "scratch", "run", [f"M8/m85_{low}_a1_bez_grafa/fold*/seed*"], 0.058 if S else 0.003,
         "A2 = A1 без графа (M8.5); рецепт = общий")
    cell(ds, "MPK_large", "native", "scratch", "run", [f"M8/m83_{low}_a1_big/fold*/seed*"], 0.096 if S else 0.015,
         "D 256, N 6 (M8.3); рецепт = общий")
    cell(ds, "ST-GCN", "native", "scratch", "stgcn", ["M11/stgcn/fold[1-5]" if S else "M11/stgcn_bukva/fold[1-5]"],
         0.53 if S else 0.07, "их код yysijie (Г3-1/Г3-3), 1 запуск; ч — по времени файлов эпох")
    cell(ds, "ST-GCN", "shared", "scratch", "run",
         ["M11G3/g3_slovo_stgcn_orig_mpkrec/fold*/seed*"] if S else ["M12/m12_1_bukva_stgcn_orig_A3/fold*/seed*"],
         0.135 if S else 0.014, "Г3-2 / M12-1 A")
    cell(ds, "SL-GCN", "native", "scratch", "samslr", ["M11/samslr/fold[1-5]" if S else "M11/samslr_bukva/fold[1-5]"],
         3.6 if S else 0.54, "их код SAM-SLR (Г/Г2-3), 1 запуск; ч — по времени файлов эпох")
    cell(ds, "SL-GCN", "shared", "scratch", "run",
         ["M11G2/g2_slovo_slgcn_orig_mpkrec/fold*/seed*"] if S else ["M12/m12_1_bukva_slgcn_orig_A3/fold*/seed*"],
         0.297 if S else 0.047, "Г2-2 / M12-1 A")
    cell(ds, "Sohn", "native", "scratch", "m13", [f"UzSohn/{low}/fold*/seed*", f"M13/{low}/fold*/seed*"], 0.43 if S else 0.05,
         "M13: 4 затравки (42–45) идут; 5-я — дообучить")
    cell(ds, "Sohn", "shared", "scratch", "none", [], 0.30 if S else 0.04, "~ нет порта Sohn в mpk.train (вопрос)")
    # справочный состав (только Slovo/Bukva, одиночки)
    cell(ds, "E1", "native", "scratch", "run", [f"M8/m81_{low}_e1_transformer_e1rec/fold*/seed*"], 0.020 if S else 0.003,
         "трансформер E1, рецепт e1rec (M8.1)", main=False)
    if S:
        cell(ds, "E1", "shared", "scratch", "run", ["M8/m81_slovo_e1_transformer_mpkrec/fold*/seed*"], 0.020,
             "трансформер E1, рецепт mpkrec", main=False)
    for m, c, hs, hb in (("BiLSTM", "e16_bilstm", 0.024, 0.004), ("transformer2L", "e16_transformer", 0.011, 0.001),
                         ("MLP", "e16_mlp", 0.008, 0.001)):
        cell(ds, m, "native", "scratch", "run", [f"E1F/{low}_{c}/fold*/seed*"], hs if S else hb,
             "E1.6 (общий рецепт E1, без EMA); коммит не записан (урок 5)", main=False)
    cell(ds, "hybrid_narrow", "shared", "scratch", "none", [f"M12_2/m12_2_{low}_hybrid_narrow/fold*/seed*"],
         0.344 if S else 0.053, "M12-2n (идёт); модель — свой цикл m12_2n_train, вывод подключить после M12-2n", main=False)

for cond in ("transfer", "scratch"):
    T = cond == "transfer"
    cell(LO, "A1", "native", cond, "run", [f"{V2}/loso/*/seed*"] if T else [f"{V2}/loso1b/B1/*/seed*"],
         0.038 if T else 0.26, "шаг 1 (V2) / шаг 1b Б1", pre=0.087 if T else None)
    cell(LO, "A2", "native", cond, "run", [], 0.033 if T else 0.22, "~ A1 × 0,85 (Slovo)", pre=0.041 if T else None)
    cell(LO, "MPK_large", "native", cond, "run", [], 0.053 if T else 0.37, "~ A1 × 1,4 (Slovo)",
         pre=0.128 if T else None)
    cell(LO, "ST-GCN", "native", cond, "their", [], 0.08, "~ их код: Bukva 0,07 ч × объём (scripts/turnir_their.py)", pre=0.65 if T else None)
    cell(LO, "ST-GCN", "shared", cond, "run", [], 0.045 if T else 0.32, "~ SL-GCN × 0,45 (Slovo)",
         pre=0.18 if T else None)
    cell(LO, "SL-GCN", "native", cond, "their", [], 0.6, "~ их код: Bukva 0,54 ч × объём (scripts/turnir_their.py)", pre=4.5 if T else None)
    cell(LO, "SL-GCN", "shared", cond, "run", [f"{V2}/M12_slgcn_loso/*/seed*"] if T else [], 0.10 if T else 0.70,
         "M12 LOSO / ~ шаги с нуля × 7", pre=0.394 if T else None)
    cell(LO, "Sohn", "native", cond, "m13", [f"{r}/uzsl_{'pre' if T else 'scratch'}/*/seed*" for r in ("UzSohn", "M13")], 0.07,
         "M13: 4 затравки идут; 5-я — дообучить", pre=0.5 if T else None)
    cell(LO, "Sohn", "shared", cond, "none", [], 0.05 if T else 0.35, "~ нет порта Sohn в mpk.train (вопрос)",
         pre=0.4 if T else None)


def cell_dir(c) -> Path:
    return PRED / c["ds"] / c["model"] / c["recipe"] / c["cond"]


def src_dirs(c) -> list[Path]:
    out = []
    for g in c["src"]:
        out += sorted(Path(p).resolve() for p in glob.glob(str(RUNS / g)) if Path(p).is_dir() and not Path(p).name.startswith("_"))
    return out


# ── Сбор сведений о готовых прогонах (для инвентаризации) ─────────────────────────────────────────────────────────────
_SHA = {}


def sha(p: Path) -> str:
    p = p.resolve()
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _SHA[p] = h.hexdigest()
    return _SHA[p]


def run_info(d: Path, kind: str) -> dict:
    if kind == "run":
        m = json.loads((d / "metrics.json").read_text())
        cfg = m["config"]
        code = (m.get("code") or {}).get("osnovnaya_model") or {}
        ins = m.get("inputs_sha256") or {}
        src_diff = sorted(Path(k).name for k, v in ins.items() if "/src/mpk/" in k and Path(k).is_file() and sha(Path(k)) != v)
        data = {Path(k).name: v[:8] for k, v in ins.items() if "/src/" not in k}
        return dict(fold=cfg["data"]["fold"], seed=cfg["seed"], commit=code.get("commit") or f"нет (e1 {str(m.get('git_commit'))[:7]})",
                    dirty=code.get("dirty"), data=data, src_diff=src_diff, has_pt=(d / "best.pt").is_file(),
                    init=bool(m.get("init_from")), cfgsig=json.dumps({k: v for k, v in cfg.items() if k not in ("seed", "data", "finetune", "experiment", "condition", "plan")},
                                                                       sort_keys=True, default=str))
    if d.name.startswith("seed"):  # turnir_their.py: <фолд>/seed<k>
        return dict(fold=d.parent.name, seed=int(d.name[4:]), commit="их код (турнир)", dirty=False, data={}, src_diff=[],
                    has_pt=True, init=False, cfgsig="")
    return dict(fold=d.name, seed=0, commit="их код", dirty=False, data={}, src_diff=[], has_pt=True, init=False, cfgsig="")


def data_ok(d: Path) -> bool:
    """Данные и фолды прогона = текущим файлам: хеши кэша и разбиения из inputs_sha256 (0063 п. 1). Их код (pkl) — True."""
    if not (d / "metrics.json").is_file():
        return True
    ins = json.loads((d / "metrics.json").read_text()).get("inputs_sha256", {})
    for k, v in ins.items():
        if "/src/" in k:
            continue
        p = Path(k) if Path(k).is_absolute() else ROOT / k
        if not p.is_file() or sha(p) != v:
            return False
    return True


def m10_file(d: Path) -> Path | None:
    cfg = json.loads((d / "metrics.json").read_text())["config"]
    for r in M10C:
        p = r / cfg["experiment"] / cfg["condition"] / f"{cfg['data']['fold']}_seed{cfg['seed']}.npz"
        if p.is_file():
            return p
    return None


def inventory() -> None:
    nfold = {k: len(v) for k, v in FOLDS.items()}
    rows, tot, tots = [], defaultdict(float), defaultdict(float)
    for c in C:
        ds = src_dirs(c) if c["kind"] in ("run", "samslr", "stgcn", "m13", "their") else []
        if c["kind"] == "run":
            ds = [d for d in ds if (d / "metrics.json").is_file()]
            n_stale = sum(not data_ok(d) for d in ds)
            ds = [d for d in ds if data_ok(d)]  # затравки на иных данных не допускаются (0063 п. 1)
        elif c["kind"] == "m13":
            ds = [d for d in ds if (d / "metrics.json").is_file()]
        else:
            ds = [d for d in ds if (d / "predictions.csv").is_file()]
        info = [run_info(d, "run" if c["kind"] == "m13" else c["kind"]) for d in ds] if c["kind"] != "m13" else \
            [dict(fold=d.parent.name, seed=d.name, commit="M13", data={}, src_diff=[], has_pt=True, cfgsig="") for d in ds]
        seeds = defaultdict(set)
        for i in info:
            seeds[i["seed"]].add(i["fold"])
        full = sorted(s for s, f in seeds.items() if len(f) == nfold[c["ds"]])
        have = len(full) if c["kind"] != "m13" else 0  # M13 считается «в работе»
        need = max(0, N_SEEDS - have) if c["kind"] != "m13" else 1
        commits = sorted({i["commit"] for i in info})
        datas = {json.dumps(i["data"], sort_keys=True) for i in info}
        cfgs = defaultdict(set)
        for i in info:
            cfgs[i["fold"]].add(i["cfgsig"])
        diff = sorted({f for i in info for f in i["src_diff"]})
        if c["kind"] == "run":
            probs = sum(m10_file(d) is not None for d in ds)
            pstat = f"кэш M10 {probs}/{len(ds)}" + (f", выгрузить {len(ds) - probs}" if probs < len(ds) else "") if ds else "—"
        elif c["kind"] in ("samslr", "stgcn"):
            pstat = f"оценки их теста {len(ds)}/{len(ds)} (pkl)" if ds else "—"
        else:
            pstat = "—"
        n_runs = need * nfold[c["ds"]]
        h = n_runs * c["h"] + (need * c["pre"] if c["pre"] else 0)
        if c["kind"] == "m13":
            h = 1 * nfold[c["ds"]] * c["h"] + (c["pre"] or 0)
        hs = N_SEEDS * nfold[c["ds"]] * c["h"] + (N_SEEDS * c["pre"] if c["pre"] else 0)  # строго: все 5 з. заново
        if c["kind"] == "m13":
            hs = h
        stage = "Э1" if c["main"] else "Э1 справ."
        tot[(stage, c["ds"], c["recipe"])] += h
        tots[(stage, c["ds"], c["recipe"])] += hs
        n_stale = n_stale if c["kind"] == "run" else 0
        verdict = ("есть" if need == 0 else f"дообучить {need} з." if have else f"обучить {need} з.") if c["kind"] != "m13" else "M13 идёт → +1 з."
        reuse = "—"
        if info and c["kind"] == "run":
            ok_data = len(datas) == 1
            ok_cfg = all(len(v) == 1 for v in cfgs.values())  # бюджет шагов LOSO зависит от фолда — сравнение внутри фолда
            reuse = ("данные/фолды ✓" if ok_data else "данные/фолды ✗") + ("; конфиг ✓" if ok_cfg else "; конфиг ≠ между затравками фолда") + \
                    ("; src = HEAD" if not diff else "; src ≠ HEAD: " + ",".join(diff))
        elif info:
            reuse = "их код без правок; данные — runs/M11/*/data"
        if n_stale:
            verdict += f" (не допущено {n_stale} прогонов: данные ≠ текущим)"
        rows.append(f"| {c['ds']} | {c['model']} | {c['recipe']} | {c['cond']} | {len(ds)} | "
                    f"{','.join(map(str, full)) or '—'} | {', '.join(commits) or '—'} | {reuse} | {pstat} | **{verdict}** | "
                    f"{h:.1f} | {hs:.1f} | {c['note']} |")
    out = ["# Турнир — инвентаризация прогонов (генерируется: `python scripts/turnir_eval.py inventory`)", "",
           "Ячейка — (набор, модель, рецепт, условие); цель — 5 затравок × все фолды (протокол П — 5, LOSO — 11). "
           "«Полные затравки» — у которых есть все фолды. Совпадение кода: хеши `src/mpk/*.py` из `inputs_sha256` "
           "прогона против текущего дерева (HEAD); данные/фолды — хеши кэша и разбиения внутри ячейки. "
           "Ч ГП — на недостающие затравки (+ предобучения S+B для переноса); для M13 — только 5-я затравка.", "",
           "| набор | модель | рецепт | условие | прогонов | полные затравки | коммиты | совпадение | вероятности | итог | ч ГП | ч ГП строго | примечание |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows + ["", "## Сумма ч ГП (недостающее, Э1)", "",
                                                                           "| этап | набор | рецепт | ч ГП | ч ГП строго |", "|---|---|---|---|---|"]
    out += [f"| {k[0]} | {k[1]} | {k[2]} | {v:.1f} | {tots[k]:.1f} |" for k, v in sorted(tot.items())]
    for st in ("Э1", "Э1 справ."):
        for r in ("native", "shared"):
            sel = [k for k in tot if k[0] == st and k[2] == r]
            out.append(f"| **{st} итого** | все | {r} | **{sum(tot[k] for k in sel):.1f}** | **{sum(tots[k] for k in sel):.1f}** |")
    (ROOT / "reports/TURNIR_inventory.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[-12:]))


# ── Выгрузка вероятностей (без обучения) ──────────────────────────────────────────────────────────────────────────────
def softmax16(lg: np.ndarray) -> np.ndarray:
    e = np.exp(lg - lg.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float16)


def save(path: Path, **a) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **a)
    tmp.rename(path)


def export_cell(c, threads: int) -> None:
    od = cell_dir(c)
    meta = json.loads((od / "metrics.json").read_text()) if (od / "metrics.json").is_file() else {}
    for d in src_dirs(c):
        if (d / "probs.npz").is_file() and (d / "metrics.json").is_file() and c["kind"] != "run":
            kind = "m13"  # единый вывод probs.npz + predictions.csv: M13/UzSohn и наши прогоны их кодом (turnir_their.py)
        else:
            kind = c["kind"]
        if kind == "run":
            if not (d / "best.pt").is_file() or not (d / "metrics.json").is_file():
                continue
            cfg = json.loads((d / "metrics.json").read_text())["config"]
            fold, seed = cfg["data"]["fold"], cfg["seed"]
            out = od / f"{fold}__seed{seed}.npz"
            if out.is_file():
                continue
            if not data_ok(d):  # другой кэш/разбиение, чем сейчас — затравка не допускается и обучается заново
                print("данные ≠ текущим, пропуск", d, flush=True)
                continue
            src = m10_file(d)
            if src is None:  # выгрузка тем же кодом, что кэш M10 (окна mpk.train.infer, ЦП float32, сверка top-1)
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                sys.path.insert(0, str(ROOT / "scripts"))
                import m10_cache
                from m10_cache_slgcn import CudaToCpu
                m10_cache.torch.set_num_threads(threads)
                try:
                    with CudaToCpu():
                        src = m10_cache.one(d.resolve(), OUT / "m10cache")
                except AssertionError as e:  # вывод на HEAD не воспроизвёл top-1 прогона — файл не пишется, прогон не допускается
                    print("!!! сверка не пройдена", d, e, flush=True)
                    with open(OUT / "export_failed.tsv", "a") as fh:
                        fh.write(f"{c['ds']}\t{c['model']}\t{d}\t{e}\n")
                    continue
            z = np.load(src)
            keep = np.isin(z["role"], ["test", "val"])
            signer = z["signer"][keep]
            if c["ds"] == LO:  # LOSO: исполнитель = тестовый исполнитель фолда (коды кэша — внутренние, урок 13в)
                signer = np.full(keep.sum(), fold)
            save(out, ids=z["sample_id"][keep], probs=softmax16(z["logits"][keep]), classes=z["labels"], signer=signer,
                 gloss=z["gloss"][keep], role=z["role"][keep], run=str(d))
            meta[out.name] = dict(run=str(d.relative_to(ROOT.parent)), src=str(src.relative_to(ROOT.parent)),
                                  top1_test=float(z["top1_test"]))
        elif kind in ("samslr", "stgcn"):
            out = od / f"{d.name}__seed0.npz"
            if out.is_file() or not (d / "predictions.csv").is_file():
                continue
            data = d.parent / "data" / d.name
            pk = d / "work_dir_test/test_result.pkl" if kind == "stgcn" else \
                next(iter(sorted((d / "work_dir" / f"sam_slr_joint_{d.name}_test/eval_results").glob("epoch_*_*.pkl"))))
            score = pickle.loads(pk.read_bytes())
            labels = json.loads((data / "labels.json").read_text())
            tm = json.loads((data / "test_meta.json").read_text())
            ids = list(score)
            save(out, ids=np.array(ids), probs=softmax16(np.stack([score[k] for k in ids]).astype(np.float32)),
                 classes=np.array(labels), signer=np.array([tm[k]["signer_id"] for k in ids]),
                 gloss=np.array([tm[k]["gloss"] for k in ids]), role=np.full(len(ids), "test"), run=str(d))
            meta[out.name] = dict(run=str(d.relative_to(ROOT.parent)), src=str(pk.relative_to(ROOT.parent)))
        elif kind == "m13":  # порт Sohn (M13/UzSohn): probs.npz (ids, probs float16, classes) + predictions.csv
            if not (d / "probs.npz").is_file():
                continue
            fold, seed = d.parent.name, d.name.removeprefix("seed")
            out = od / f"{fold}__seed{seed}.npz"
            if out.is_file():
                continue
            z = np.load(d / "probs.npz")
            with open(d / "predictions.csv", newline="", encoding="utf-8") as f:
                pr = {r["sample_id"]: r for r in csv.DictReader(f)}
            ids = z["ids"]
            signer = np.full(len(ids), fold) if c["ds"] == LO else np.array([pr[k]["signer_id"] for k in ids])
            save(out, ids=ids, probs=z["probs"].astype(np.float16), classes=z["classes"], signer=signer,
                 gloss=np.array([pr[k]["gloss"] for k in ids]), role=np.full(len(ids), "test"), run=str(d))
            meta[out.name] = dict(run=str(d.relative_to(ROOT.parent)), src="probs.npz")
        else:
            continue
        p = load_pred(out)
        t = p["role"] == "test"
        meta[out.name]["top1_stand"] = float((p["classes"][p["probs"][t].argmax(1)] == p["gloss"][t]).mean())
        od.mkdir(parents=True, exist_ok=True)
        (od / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print("готово", out.relative_to(OUT), round(meta[out.name]["top1_stand"], 4), flush=True)


def load_pred(p: Path) -> dict:
    z = np.load(p)
    return {k: z[k] for k in ("ids", "probs", "classes", "signer", "gloss", "role")}


# ── Модель = набор файлов предсказаний (фолды × затравки) ─────────────────────────────────────────────────────────────
def load_model(d: Path, role: str = "test") -> dict:
    """{seed: {'ids','probs','classes','signer','gloss'}} — фолды склеены; ключ записи val — 'фолд/id'."""
    by = defaultdict(list)
    for f in sorted(d.glob("*__seed*.npz")):
        fold, seed = f.stem.split("__seed")
        p = load_pred(f)
        k = p["role"] == role
        if not k.any():
            continue
        ids = p["ids"][k] if role == "test" else np.char.add(fold + "/", p["ids"][k])
        by[int(seed)].append(dict(ids=ids, probs=p["probs"][k], classes=p["classes"], signer=p["signer"][k], gloss=p["gloss"][k]))
    out = {}
    for s, parts in by.items():
        cls = parts[0]["classes"]
        assert all((q["classes"] == cls).all() for q in parts), "классы различаются между фолдами"
        ids = np.concatenate([q["ids"] for q in parts])
        assert len(set(ids)) == len(ids), f"{d}: запись в тесте двух фолдов"
        o = np.argsort(ids)
        out[s] = {k: np.concatenate([q[k] for q in parts])[o] for k in ("ids", "probs", "signer", "gloss")} | {"classes": cls}
    return out


def align(models: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Общие записи всех моделей и затравок → ids, signer, gloss."""
    common = None
    for m in models:
        for s in m.values():
            common = set(s["ids"]) if common is None else common & set(s["ids"])
    ids = np.array(sorted(common))
    ref = next(iter(models[0].values()))
    ix = np.searchsorted(ref["ids"], ids)
    return ids, ref["signer"][ix], ref["gloss"][ix]


def probs_on(seedrun: dict, ids: np.ndarray, classes: np.ndarray) -> np.ndarray:
    assert (seedrun["classes"] == classes).all(), "классы моделей различаются"
    return seedrun["probs"][np.searchsorted(seedrun["ids"], ids)].astype(np.float32)


def correct_ens(model: dict, N: int, ids, gloss) -> np.ndarray:
    """Верность на запись, усреднённая по всем сочетаниям N затравок (N = 1 — среднее по затравкам)."""
    seeds = sorted(model)
    cls = model[seeds[0]]["classes"]
    P = {s: probs_on(model[s], ids, cls) for s in seeds}
    combos = list(itertools.combinations(seeds, N))
    c = np.zeros(len(ids))
    for cb in combos:
        c += cls[sum(P[s] for s in cb).argmax(1)] == gloss
    return c / len(combos)


def correct_mixed(models: list[dict], counts: tuple, ids, gloss) -> np.ndarray:
    """Смешанный ансамбль: модель a даёт counts[a] затравок; повторение r берёт у модели a затравки (r + j) mod n_a."""
    cls = next(iter(models[0].values()))["classes"]
    R = min(len(m) for m in models)
    c = np.zeros(len(ids))
    for r in range(R):
        p = 0
        for m, k in zip(models, counts):
            ss = sorted(m)
            for j in range(k):
                p = p + probs_on(m[ss[(r + j) % len(ss)]], ids, cls)
        c += cls[p.argmax(1)] == gloss
    return c / R


def compositions(n_models: int, N: int, cap: int):
    for cb in itertools.product(range(min(N, cap) + 1), repeat=n_models):
        if sum(cb) == N:
            yield cb


def correct_mixed_nested(models, names, N, ids, gloss, signer):
    """Корпус (нет валидации): для каждого исполнителя s состав выбирается по macro-точности на остальных исполнителях
    (их тест LOSO — модели фолдов, не видевших s) и применяется к записям s. Имя — самый частый состав (плюс «*», если
    составы различаются между исполнителями)."""
    cap = min(len(m) for m in models)
    cbs = list(compositions(len(models), N, cap))
    if not cbs:
        return None, None
    C = np.stack([correct_mixed(models, cb, ids, gloss) for cb in cbs])
    sg = sorted(set(signer))
    acc = np.stack([C[:, signer == x].mean(1) for x in sg], 1)  # [состав, исполнитель]
    real = np.array([x != "other" for x in sg])
    out, chosen = np.zeros(len(ids)), []
    for k, x in enumerate(sg):
        others = real & (np.arange(len(sg)) != k)
        best = max(range(len(cbs)), key=lambda q: (acc[q, others].mean(), -sum(v > 0 for v in cbs[q])))
        out[signer == x] = C[best, signer == x]
        chosen.append(cbs[best])
    top = max(set(chosen), key=chosen.count)
    name = "+".join(f"{n}×{k}" if k > 1 else n for n, k in zip(names, top) if k) + ("*" if len(set(chosen)) > 1 else "")
    return out, name


def choose_mixed(val_models: list[dict], N: int) -> tuple:
    """Состав по валидации: перебор всех составов (сумма N, не больше n затравок у модели); при равенстве — меньше
    различных моделей, затем порядок перебора. Тест не читается."""
    ids, _, gloss = align(val_models)
    best = None
    for cb in compositions(len(val_models), N, min(len(m) for m in val_models)):
        acc = correct_mixed(val_models, cb, ids, gloss).mean()
        key = (acc, -sum(k > 0 for k in cb))
        if best is None or key > best[0]:
            best = (key, cb)
    return best[1] if best else None  # мало затравок для N — смешанного ансамбля нет


# ── Бутстреп по исполнителям ──────────────────────────────────────────────────────────────────────────────────────────
def per_signer(c: np.ndarray, signer: np.ndarray, drop=("other",)) -> tuple[np.ndarray, np.ndarray]:
    sg = sorted(set(signer) - set(drop))
    num = np.array([c[signer == s].sum() for s in sg])
    den = np.array([(signer == s).sum() for s in sg], dtype=float)
    return num, den


def boot(num, den, level, ds, macro: bool, B: int | None = None) -> tuple[float, float, float]:
    B0, seed = BOOT[ds]
    B = B or B0
    b = np.random.default_rng(seed).integers(0, len(num), (B, len(num)))
    if macro:
        a = num / den
        point, bs = a.mean(), a[b].mean(1)
    else:
        point, bs = num.sum() / den.sum(), num[b].sum(1) / den[b].sum(1)
    lo, hi = np.quantile(bs, [(1 - level) / 2, (1 + level) / 2])
    return 100 * point, 100 * lo, 100 * hi


def acc_ci(c, signer, level, ds, macro=True, B=None):
    return boot(*per_signer(c, signer), level, ds, macro, B)


def paired(ca, cb, signer, level, ds, macro=True, B=None) -> dict:
    """Δ = a − b на одних записях; macro — разность точностей исполнителей (как loso_slgcn_tables), micro — как
    m11_tables.paired (ДИ отношения сумм разностей)."""
    na, den = per_signer(ca, signer)
    nb, _ = per_signer(cb, signer)
    d, lo, hi = boot(na - nb, den, level, ds, macro, B)
    worse = float(np.mean((na - nb) / den < 0))
    return dict(delta=d, lo=lo, hi=hi, level=level, worse_share=worse, n_signers=len(den))


def bonf(m: int) -> float:
    return 1 - 0.01 / m  # правило 0046 в редакции 0063 п. 6: семейное α = 0,01


def verdict(lo, hi) -> str:
    return "лучше" if lo > 0 else "хуже" if hi < 0 else "не установлено"


# ── Сводка (FORMAT.md 0062) ───────────────────────────────────────────────────────────────────────────────────────────
def cost_table() -> dict:
    p = ROOT / "reports/turnir_cost.json"  # пишет сеанс стоимости (один сеанс, после Э1); без него — «нет данных»
    return json.loads(p.read_text()) if p.is_file() else {}


def summary(level_main: float) -> None:
    cost = cost_table()
    rows = []

    def add(ds, model, kind, members, recipe, cond, c, signer, n_seeds, source, parts):
        t, lo, hi = acc_ci(c, signer, level_main, ds, B=B_MAIN)
        g = [cost.get(ds, {}).get(p, {}) for p in parts]  # turnir_cost.json: {набор: {модель: {gflops, latency_ms, params_m}}}
        sm = lambda k: round(sum(x[k] for x in g), 4) if g and all(k in x for x in g) else "нет данных"  # noqa: E731
        rows.append([ds, model, kind, members, recipe, cond, round(t, 3), round(lo, 3), round(hi, 3),
                     round(100 * level_main, 2), sm("gflops"), sm("latency_ms"), sm("params_m"), n_seeds, source])

    head = os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()
    for ds in FOLDS:
        for recipe in ("native", "shared"):
            for cond in ("transfer", "scratch"):
                base = PRED / ds
                cells = sorted(p for p in base.glob(f"*/{recipe}/{cond}") if any(p.glob("*.npz")))
                if not cells:
                    continue
                M = {p.parts[-3]: load_model(p) for p in cells}
                M = {k: v for k, v in M.items() if v}
                if not M:
                    continue
                for name, m in M.items():
                    ids, signer, gloss = align([m])
                    src = f"runs/TURNIR/preds/{ds}/{name}/{recipe}/{cond}@{head}"
                    add(ds, name, "single", 1, recipe, cond, correct_ens(m, 1, ids, gloss), signer, len(m), src, [name])
                    for N in range(2, len(m) + 1):
                        add(ds, name, "ens_homo", N, recipe, cond, correct_ens(m, N, ids, gloss), signer, len(m), src, [name] * N)
    # смешанные: «наша толпа» и «толпа Sohn», каждая модель — в рецепте своей главной строки (MAIN_RECIPE); состав — по
    # валидации (протокол П) или вложенным LOSO (корпус: состав для исполнителя s выбирается по остальным исполнителям)
    for ds in FOLDS:
        for cond in ("transfer", "scratch"):
            for pool in (["A1", "A2", "MPK_large", "ST-GCN", "SL-GCN"], ["Sohn", "UzSohn"]):
                M = {n: load_model(PRED / ds / n / MAIN_RECIPE.get(n, "native") / cond) for n in pool}
                names = [n for n in pool if M[n]]
                if len(names) < 2:
                    continue
                ids, signer, gloss = align([M[n] for n in names])
                for N in (3, 5):
                    if ds == LO:
                        c, comp = correct_mixed_nested([M[n] for n in names], names, N, ids, gloss, signer)
                        src = f"вложенный LOSO@{head}"
                    else:
                        V = [load_model(PRED / ds / n / MAIN_RECIPE.get(n, "native") / cond, "val") for n in names]
                        cb = choose_mixed(V, N) if all(V) else None
                        if cb is None:
                            continue
                        comp = "+".join(f"{n}×{k}" if k > 1 else n for n, k in zip(names, cb) if k)
                        c, src = correct_mixed([M[n] for n in names], cb, ids, gloss), f"val-состав@{head}"
                    if c is None:
                        continue
                    parts = [p for p in comp.split("+") for _ in range(int(p.split("×")[1]) if "×" in p else 1)]
                    parts = [p.split("×")[0] for p in parts]
                    add(ds, comp, "ens_mixed", N, "native", cond, c, signer, min(len(M[n]) for n in names), src, parts)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "turnir_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FMT)
        w.writerows(rows)
    print("записано", OUT / "turnir_summary.csv", len(rows), "строк")


def pairs_table(pairs: list[tuple], main: bool, cond_corpus="transfer") -> list[str]:
    """Пары (набор, модель_a, модель_b), одиночки; рецепт — MAIN_RECIPE (главная строка архитектуры). main — уровень
    1 − 0,01/m и 20 000 повторов; иначе — справочно, 95 %. Вывод — по macro; micro рядом (0063 п. 7)."""
    lvl, B = (bonf(len(pairs)), B_MAIN) if main else (0.95, None)
    out = [f"| набор | пара | Δ macro, п. п. | ДИ {100 * lvl:.4g} % | вывод | Δ micro [ДИ] | доля исп. с ухудшением |",
           "|---|---|---|---|---|---|---|"]
    for ds, a, b in pairs:
        cond = cond_corpus if ds == LO else "scratch"
        ma, mb = (load_model(PRED / ds / x / MAIN_RECIPE.get(x, "native") / cond) for x in (a, b))
        if not ma or not mb:
            out.append(f"| {ds} | {a} − {b} | ждёт | | | | |")
            continue
        ids, signer, gloss = align([ma, mb])
        ca, cb = correct_ens(ma, 1, ids, gloss), correct_ens(mb, 1, ids, gloss)
        r = paired(ca, cb, signer, lvl, ds, True, B)
        q = paired(ca, cb, signer, lvl, ds, False, B)
        out.append(f"| {ds} | {label(a)} − {label(b)} | {r['delta']:+.2f} | [{r['lo']:+.2f}; {r['hi']:+.2f}] | "
                   f"{sohn_verdict(a, b, r) if a in SOHN_LIKE or b in SOHN_LIKE else verdict(r['lo'], r['hi'])} | "
                   f"{q['delta']:+.2f} [{q['lo']:+.2f}; {q['hi']:+.2f}] | {100 * r['worse_share']:.0f} % |")
        if a in SOHN_LIKE or b in SOHN_LIKE:
            sh = shift_for(a, b)
            out.append(f"| {ds} | ↳ устойчивость: +{SOHN_GAP} к точности {label(a if a in SOHN_LIKE else b)} | {r['delta'] + sh:+.2f} | "
                       f"[{r['lo'] + sh:+.2f}; {r['hi'] + sh:+.2f}] | {verdict(r['lo'] + sh, r['hi'] + sh)} | | |")
    return out


def label(x: str) -> str:
    return {"Sohn": "наша реализация Sohn", "UzSohn": "UzSohn (на основе нашей реализации Sohn)"}.get(x, x)


def shift_for(a: str, b: str) -> float:
    """Сдвиг разности a − b при прибавлении недобора порта к точности Sohn/UzSohn на каждой записи (macro — ровно сдвиг)."""
    return SOHN_GAP if a in SOHN_LIKE else -SOHN_GAP


def sohn_verdict(a: str, b: str, r: dict) -> str:
    """0078 п. 3: «X лучше Sohn» пишется «лучше», только если держится и после сдвига; иначе — «лучше нашей реализации Sohn;
    с поправкой на недобор порта — не установлено». Основное правило (без сдвига) не меняется."""
    lo, hi, sh = r["lo"], r["hi"], shift_for(a, b)
    x_better = (b in SOHN_LIKE and lo > 0) or (a in SOHN_LIKE and hi < 0)
    if not x_better:
        return verdict(lo, hi)
    robust = (b in SOHN_LIKE and lo + sh > 0) or (a in SOHN_LIKE and hi + sh < 0)
    base = verdict(lo, hi)
    x = b if a in SOHN_LIKE else a
    return base if robust else f"{base}: {x} лучше нашей реализации Sohn; с поправкой на недобор порта — не установлено"


# главные пары (0063 п. 6): ровно 8, m = 8 → ДИ 1 − 0,01/8 = 99,875 %
MAIN_PAIRS = [(ds, a, b) for ds in ("Slovo", "UzSL") for a, b in
              (("Sohn", "A1"), ("Sohn", "SL-GCN"), ("UzSohn", "A1"), ("UzSohn", "SL-GCN"))]
# справочно (95 %): прежние пары Э1, правило остановки UzSohn (Sohn − SL-GCN на корпусе — уже в главных), Bukva
REF_PAIRS = [(ds, a, b) for ds in ("Slovo", "UzSL", "Bukva") for a, b in
             (("A1", "SL-GCN"), ("A1", "ST-GCN"), ("A1", "A2"), ("A1", "MPK_large"))] + \
            [("Bukva", a, b) for a, b in (("Sohn", "A1"), ("Sohn", "SL-GCN"), ("UzSohn", "A1"), ("UzSohn", "SL-GCN"))]


# ── Правило остановки ветки UzSohn (поправка 2 к TURNIR_prereg.md) ─────────────────────────────────────────────────────
def stoprule() -> None:
    """Sohn (родной, перенос, M13: затравки 42–45) − SL-GCN (общий = главный, перенос, LOSO M12: затравки 0–2) на корпусе,
    macro по 10 исполнителям («other» вне), парно на общих записях, 99 % ДИ (0059), бутстреп 20 000, default_rng(20261104).
    Верхняя граница < 0 → ветка UzSohn закрыта. Считается один раз — сразу после M13 (runs/M13.done или UzSohn.done)."""
    n_pre = len(list((RUNS / "UzSohn/uzsl_pre").glob("*/seed4[2-6]/probs.npz")))
    assert n_pre == 55, f"M13: корпус с переносом готов {n_pre} из 55"
    sohn = next(c for c in C if c["ds"] == LO and c["model"] == "Sohn" and c["recipe"] == "native" and c["cond"] == "transfer")
    export_cell(sohn, 8)
    ms = load_model(cell_dir(sohn))
    mg = load_model(PRED / LO / "SL-GCN/shared/transfer")
    sk = {s: v for s, v in ms.items() if 42 <= s <= 46}  # ровно M13 (5 затравок, поправка 3)
    gk = {s: v for s, v in mg.items() if s in (0, 1, 2)}  # ровно M12 LOSO (не новые затравки группы 1)
    assert len(sk) == 5 and len(gk) == 3, (sorted(ms), sorted(mg))
    for m in (sk, gk):
        for v in m.values():
            assert len(v["ids"]) == len(next(iter(sk.values()))["ids"]), "у затравок разное число записей теста"
    ids, signer, gloss = align([sk, gk])
    n_all = {len(v["ids"]) for m in (sk, gk) for v in m.values()}
    assert n_all == {len(ids)}, f"записи теста не совпадают: общих {len(ids)}, в моделях {n_all}"
    a, b = correct_ens(sk, 1, ids, gloss), correct_ens(gk, 1, ids, gloss)
    r = paired(a, b, signer, 0.99, LO, True, B_MAIN)
    q = paired(a, b, signer, 0.95, LO, False)
    closed = r["hi"] < 0
    out = ["# Правило остановки ветки UzSohn (поправка 2 к TURNIR_prereg.md) — числа корпуса предварительные до проверки меток", "",
           f"Наша реализация Sohn (M13, Э0 не пройден; родной, перенос; затравки {sorted(sk)}) − SL-GCN (M12 LOSO, общий, перенос; затравки {sorted(gk)}),",
           f"macro по {r['n_signers']} исполнителям, {len(ids)} общих записей теста, 99 % ДИ, бутстреп {B_MAIN}.", "",
           f"**Δ = {r['delta']:+.2f} п. п., 99 % ДИ [{r['lo']:+.2f}; {r['hi']:+.2f}]** — "
           + ("**верхняя граница < 0: ветка UzSohn ЗАКРЫТА**" if closed else "**ветка UzSohn продолжается**"),
           f"Устойчивость (справочно, 0078 п. 3; решение не меняет): +{SOHN_GAP} к точности нашей реализации Sohn → "
           f"Δ = {r['delta'] + SOHN_GAP:+.2f} [{r['lo'] + SOHN_GAP:+.2f}; {r['hi'] + SOHN_GAP:+.2f}].",
           f"(справочно: micro {q['delta']:+.2f} [95 %: {q['lo']:+.2f}; {q['hi']:+.2f}]; исполнителей с ухудшением {100 * r['worse_share']:.0f} %)", ""]
    (OUT / "stoprule.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


# ── Сверка с прежними таблицами ───────────────────────────────────────────────────────────────────────────────────────
def from_predcsv(dirs: list[Path]) -> dict:
    """Та же модель из predictions.csv (без вероятностей) — проверка статистики стенда отдельно от выгрузки."""
    by = defaultdict(list)
    sig = {}
    for d in dirs:
        with open(d / "predictions.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                by[r["sample_id"]].append(int(r["correct"]))
                sig[r["sample_id"]] = r["signer_id"]
    ids = np.array(sorted(by))
    return dict(ids=ids, c=np.array([np.mean(by[k]) for k in ids]), signer=np.array([sig[k] for k in ids]))


def check() -> None:
    res, ok = [], True

    def line(name, got, ref, tol=0.005):
        nonlocal ok
        good = all(abs(g - r) <= tol for g, r in zip(got, ref))
        ok &= good
        res.append(f"| {name} | {'; '.join(f'{g:.2f}' for g in got)} | {'; '.join(f'{r:.2f}' for r in ref)} | {'✓' if good else '✗'} |")

    a1_dirs = sorted(RUNS.glob("M3/slovo_a_ema/fold*/seed0")) + sorted(RUNS.glob("M7/slovo_a_ema/fold*/seed*"))
    sl_dirs = sorted(RUNS.glob("M11/samslr/fold[1-5]"))
    a1, sl = from_predcsv(a1_dirs), from_predcsv(sl_dirs)
    line("A1 Slovo П, predictions.csv, micro 95 % (m11 Г: 70.78 [68.8; 72.7])", acc_ci(a1["c"], a1["signer"], .95, P, False), (70.78, 68.8, 72.7), 0.06)
    line("SL-GCN ориг. (их рецепт), то же (74.61 [72.6; 76.5])", acc_ci(sl["c"], sl["signer"], .95, P, False), (74.61, 72.6, 76.5), 0.06)
    assert (a1["ids"] == sl["ids"]).all()
    r = paired(a1["c"], sl["c"], a1["signer"], .95, P, macro=False)
    line("A1 − SL-GCN ориг., парно, 95 % (−3.84 [−4.99; −2.72])", (r["delta"], r["lo"], r["hi"]), (-3.84, -4.99, -2.72))
    bk = from_predcsv(sorted(RUNS.glob("M7/bukva_a_ema/fold*/seed*")))
    line("A1 Bukva П, micro 95 % (93.19 [92.3; 94.0])", acc_ci(bk["c"], bk["signer"], .95, "Bukva", False), (93.19, 92.3, 94.0), 0.06)
    # корпус: macro по исполнителям, A1 − SL-GCN, 99 % (M12_slgcn_loso_tables: −2.91 [−5.11; −1.18])
    def lo(root):
        d = [p for p in sorted((UZ / "runs" / root).glob("*/seed[012]")) if p.parent.name != "other"]
        m = from_predcsv(d)
        m["signer"] = np.array([None] * len(m["ids"]), dtype=object)
        fold_of = {}
        for p in d:
            with open(p / "predictions.csv", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fold_of[row["sample_id"]] = p.parent.name
        m["signer"] = np.array([fold_of[k] for k in m["ids"]])
        return m
    ua, us = lo("loso"), lo("M12_slgcn_loso")
    r = paired(ua["c"], us["c"], ua["signer"], .99, LO)
    line("корпус A1 − SL-GCN, macro, 99 % (−2.91 [−5.11; −1.18])", (r["delta"], r["lo"], r["hi"]), (-2.91, -5.11, -1.18))
    # то же по выгруженным вероятностям стенда (если выгружены): допуск шире — ЦП float32 против bf16 ГП (урок 13б)
    for ds, name, recipe in ((P, "A1", "native"), (P, "SL-GCN", "native"), (P, "SL-GCN", "shared"), ("Bukva", "A1", "native")):
        d = PRED / ds / name / recipe / "scratch"
        m = load_model(d) if d.is_dir() else {}
        if len(m) and all(len(v["ids"]) for v in m.values()):  # эталон — predictions.csv тех же прогонов, что выгружены
            meta = json.loads((d / "metrics.json").read_text())
            ref = from_predcsv([ROOT.parent / v["run"] for k, v in meta.items() if (d / k).is_file()])
            ids, signer, gloss = align([m])
            line(f"{ds} {name} {recipe} по вероятностям стенда ({len(meta)} прогонов, micro 95 %)",
                 acc_ci(correct_ens(m, 1, ids, gloss), signer, .95, ds, False)[:1], (100 * ref["c"].mean(),), 0.10)
    for nm, ref in (("A1", None), ("SL-GCN", None)):
        d = PRED / LO / nm / ("native" if nm == "A1" else "shared") / "transfer"
        if d.is_dir() and nm == "SL-GCN" and (PRED / LO / "A1/native/transfer").is_dir():
            ma, mb = load_model(PRED / LO / "A1/native/transfer"), load_model(d)
            if ma and mb:
                ids, signer, gloss = align([ma, mb])
                keep = signer != "other"
                r = paired(correct_ens(ma, 1, ids, gloss)[keep], correct_ens(mb, 1, ids, gloss)[keep], signer[keep], .99, LO)
                # допуск 0,3 п. п.: ЦП float32 против bf16 ГП — расхождение ≤ 0,7 записи на исполнителя (среднее 3 з.), а тест
                # исполнителя LOSO — 37–300 записей, поэтому macro сдвигается сильнее, чем на протоколе П (урок 13б)
                line("корпус A1 − SL-GCN по вероятностям стенда (99 %)", (r["delta"],), (-2.91,), 0.30)
    out = ["| сверка | стенд | прежнее | |", "|---|---|---|---|"] + res
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "check.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print("СВЕРКА", "ПРОЙДЕНА" if ok else "НЕ ПРОЙДЕНА")
    sys.exit(0 if ok else 1)


# ── Самопроверка на синтетике ─────────────────────────────────────────────────────────────────────────────────────────
def selftest() -> None:
    import tempfile
    global PRED
    rng = np.random.default_rng(1)
    tmp = Path(tempfile.mkdtemp())
    PRED = tmp / "preds"
    K, n_sig, per = 12, 20, 15
    classes = np.array([f"c{k}" for k in range(K)])
    ids_all = np.array([f"r{i:04d}" for i in range(n_sig * per)])
    signer_all = np.repeat([f"s{k:02d}" for k in range(n_sig)], per)
    gloss_all = classes[rng.integers(0, K, len(ids_all))]
    folds = np.arange(len(ids_all)) % 2
    skill = {"A1": 2.0, "SL-GCN": 2.6, "ST-GCN": 1.6}
    for name, sk in skill.items():
        for s in range(5):
            for f in (0, 1):
                k = folds == f
                y = np.array([int(g[1:]) for g in gloss_all[k]])
                lg = rng.normal(0, 1, (k.sum(), K))
                lg[np.arange(k.sum()), y] += sk
                vid = np.char.add("v", ids_all[k])  # валидация — другие записи с тем же навыком
                lv = rng.normal(0, 1, (k.sum(), K))
                lv[np.arange(k.sum()), y] += sk
                save(PRED / "Slovo" / name / MAIN_RECIPE.get(name, "native") / "scratch" / f"fold{f + 1}__seed{s}.npz",
                     ids=np.concatenate([ids_all[k], vid]), probs=softmax16(np.concatenate([lg, lv])), classes=classes,
                     signer=np.concatenate([signer_all[k]] * 2), gloss=np.concatenate([gloss_all[k]] * 2),
                     role=np.array(["test"] * k.sum() + ["val"] * k.sum()), run="синтетика")
    A, Sg = load_model(PRED / "Slovo/A1/native/scratch"), load_model(PRED / "Slovo/SL-GCN/shared/scratch")
    assert sorted(A) == [0, 1, 2, 3, 4] and len(A[0]["ids"]) == len(ids_all)
    ids, signer, gloss = align([A, Sg])
    c1 = correct_ens(A, 1, ids, gloss)
    manual = np.mean([(classes[A[s]["probs"].astype(np.float32).argmax(1)] == A[s]["gloss"]) for s in range(5)], 0)
    assert np.allclose(c1, manual), "одиночная = среднее по затравкам"
    c5 = correct_ens(A, 5, ids, gloss)
    assert np.allclose(c5, classes[sum(A[s]["probs"].astype(np.float32) for s in range(5)).argmax(1)] == gloss)
    assert c5.mean() > c1.mean(), "ансамбль 5 сильнее одиночной на синтетике"
    t, lo, hi = acc_ci(c1, signer, .99, P)
    assert lo <= t <= hi and abs(t - 100 * np.mean([c1[signer == s].mean() for s in sorted(set(signer))])) < 1e-9
    tm, lom, him = acc_ci(c1, signer, .95, P, macro=False)
    assert abs(tm - 100 * c1.mean()) < 1e-9 and lom <= tm <= him
    z = paired(c1, c1, signer, .99, P)
    assert z["delta"] == 0 and z["lo"] == 0 and z["hi"] == 0 and z["worse_share"] == 0
    r = paired(correct_ens(Sg, 1, ids, gloss), c1, signer, bonf(8), P, B=B_MAIN)
    assert r["lo"] > 0 and verdict(r["lo"], r["hi"]) == "лучше", r
    assert abs(bonf(8) - 0.99875) < 1e-12 and len(MAIN_PAIRS) == 8
    t2, lo2, hi2 = acc_ci(c1, signer, bonf(8), P, B=B_MAIN)
    assert lo2 < lo and hi2 > hi, "ДИ 99,875 % шире 99 %"
    # смешанный: N = 1 у одной модели = её одиночная; выбор по валидации предпочитает сильную модель
    m3 = [A, Sg, load_model(PRED / "Slovo/ST-GCN/shared/scratch")]
    assert np.allclose(correct_mixed(m3, (0, 1, 0), ids, gloss), correct_ens(Sg, 1, ids, gloss))
    V = [load_model(PRED / "Slovo" / n / MAIN_RECIPE.get(n, "native") / "scratch", "val") for n in ("A1", "SL-GCN", "ST-GCN")]
    assert choose_mixed(V, 1) == (0, 1, 0)
    assert len(list(compositions(5, 5, 5))) == 126
    # 0078 п. 3: X − Sohn = +2 [+1; +3] — после сдвига −2,71 → [−1,71; +0,29] — не держится; +4 [+3; +5] — держится
    assert sohn_verdict("A1", "Sohn", dict(lo=1.0, hi=3.0)).startswith("лучше: A1 лучше нашей реализации Sohn")
    assert sohn_verdict("A1", "Sohn", dict(lo=3.0, hi=5.0)) == "лучше"
    assert sohn_verdict("Sohn", "A1", dict(lo=-5.0, hi=-3.0)) == "хуже" and sohn_verdict("Sohn", "A1", dict(lo=-3.0, hi=-1.0)).startswith("хуже: A1 лучше нашей")
    assert sohn_verdict("Sohn", "A1", dict(lo=1.0, hi=3.0)) == "лучше"  # Sohn лучше — правило 0078 не касается
    cn, nm = correct_mixed_nested(m3, ["A1", "SL-GCN", "ST-GCN"], 1, ids, gloss, signer)
    assert nm == "SL-GCN" and np.allclose(cn, correct_ens(Sg, 1, ids, gloss)), (nm, "вложенный выбор: сильнейшая при N = 1")
    # формат сводки — проверка их же загрузчиком (0062), если клон рядом
    summary(0.99)
    rows = list(csv.DictReader(open(OUT / "turnir_summary.csv", encoding="utf-8")))
    assert rows and list(rows[0]) == FMT and {r["kind"] for r in rows} == {"single", "ens_homo", "ens_mixed"}
    pz = os.environ.get("PARETO_TURNIR")
    if pz and Path(pz).is_file():
        import importlib.util
        spec = importlib.util.spec_from_file_location("pareto_turnir", pz)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.load(str(OUT / "turnir_summary.csv"))
        print("формат принят загрузчиком pareto_turnir.load")
    print("\n".join(pairs_table([("Slovo", "A1", "ST-GCN")], True)))
    print("самопроверка пройдена;", len(rows), "строк сводки (синтетика, не результаты)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["inventory", "export", "check", "summary", "stoprule"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", help="модель (A1, SL-GCN, …)")
    ap.add_argument("--dataset", choices=list(FOLDS))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        OUT = Path(__import__("tempfile").mkdtemp()) / "TURNIR"
        selftest()
    elif a.cmd == "inventory":
        inventory()
    elif a.cmd == "export":
        for c in C:
            if (a.only is None or c["model"] == a.only) and (a.dataset is None or c["ds"] == a.dataset):
                export_cell(c, a.threads)
    elif a.cmd == "check":
        check()
    elif a.cmd == "stoprule":
        stoprule()
    elif a.cmd == "summary":
        summary(0.99)  # ДИ одиночек/ансамблей в сводке — 99 % (FORMAT.md: 99 для главных строк), 20 000 повторов
        (OUT / "pairs.md").write_text("\n".join(["# Э1 — главные пары (одиночки, родной рецепт; корпус — перенос)", "",
                                                 f"Затравок в ячейках: см. turnir_summary.csv (n_seeds). Пока Э1 не завершён (нет runs/TURNIR_E1.done) — "
                                                 f"**прогон стенда на готовых прогонах, не результат турнира**; корпус — предварительно до проверки меток.", ""] +
                                                pairs_table(MAIN_PAIRS, True) + ["", "## Справочно (95 %)", ""] +
                                                pairs_table(REF_PAIRS, False)) + "\n", encoding="utf-8")
    else:
        ap.print_help()
