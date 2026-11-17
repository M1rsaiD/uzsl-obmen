"""Факты общего рецепта для рычага № 3 (EMA с прогревом) и порта группы 3 (к3 0042 разд. 3–4). к2, 17.11.2026.

Только ЦП и только чтение. Не открывает runs/TURNIR/**, runs/M13/**, runs/UzSohn/** (тест турнира и источники Sohn);
из прогонов — только config.yaml шаблонов группы 1 и ключ batch_share в metrics.json предобучений S+B (теста там нет).
Названия фолдов корпуса и исполнителей не печатаются.
Запуск (из OM): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 PYTHONPATH=src:../uzsign_e1/src:../uzsign_ai_v2/src \
  /home/error_404/.local/share/mise/installs/python/3.14.7/bin/python <этот файл>
"""
import glob
import json
import re
import sys
from pathlib import Path

import torch
import yaml

torch.set_num_threads(4)
OM = Path("/home/error_404/uzsign/osnovnaya_model")
V2 = OM.parent / "uzsign_ai_v2"
sys.path.insert(0, str(OM / "scripts"))
from mpk import sohn  # noqa: E402
from mpk.train import ema_avg_fn, ema_decay_at, shares  # noqa: E402
import turnir_eval as te  # noqa: E402  (только список ячеек C; на импорте ничего не читает)

COMMON = yaml.safe_load((OM / "configs/common.yaml").read_text())["train"]


def where(path: Path, pat: str, n: int = 9) -> None:
    hits = [(i, s.rstrip()) for i, s in enumerate(path.read_text().splitlines(), 1) if re.search(pat, s)]
    assert hits, (path, pat)
    for i, s in hits[:n]:
        print(f"  {path.relative_to(OM.parent)}:{i}: {s.strip()[:150]}")


def decay_of(steps: int, tc: dict) -> float:  # = mpk.train.run, train.py:343 (и m13_train.train_recipe_a1)
    return tc["ema_decay"] if steps >= tc["ema_short_below"] else tc["ema_decay_short"]


def t_star(d: float) -> int:  # первое t, где прогрев уже не действует: (1 + t)/(10 + t) ≥ d
    t = 0
    while ema_decay_at(d, t) < d:
        t += 1
    return t


print("== 0. Где задано (файл:строка)")
where(OM / "configs/common.yaml", r"^\s+ema")
for p in (r"^def shares", r"^def ema_decay_at", r"return min\(decay", r"^def ema_avg_fn", r"_foreach_copy_", r"decay = tc\[\"ema_decay\"\]",
          r"AveragedModel\(", r"evaluator = model if ema", r"^\s+opt\.step\(\)", r"ema\.update_parameters", r"share = shares\(",
          r"n / bs \* seg_loss"):
    where(OM / "src/mpk/train.py", p, 2)
where(OM / "src/mpk/sohn.py", r"^CFG = dict")
where(OM / "src/mpk/sohn.py", r"_foreach_sub_\(params, d\)|^\s+opt\.step\(lr, wd\)|^\s+model\.commit_bn\(\)")
for p in (r"ema_warmup=False", r"spe = n_epoch //", r"total = a\.epochs \* spe", r"train = tr \+ va if mode", r"run_loss \+= sohn\.train_step",
          r"^\s+model\.commit_bn\(\)", r"^\s+ema\.update_parameters"):
    where(OM / "scripts/m13_train.py", p, 2)
where(OM / "reports/TURNIR_prereg.md", r"EMA 0,999")
where(OM / "reports/M9_report.md", r"\(1 \+ t\)/\(10 \+ t\)", 3)
where(OM / "reports/m9_tables.md", r"\(1\+t\)/\(10\+t\)")
where(V2 / "scripts/loso.py", r"ema_warmup: bool = True|if ema_warmup:")
where(OM / "src/mpk/baselines.py", r"self\.(heads|emb|protos) = ")

print("\n== 1. EMA: формула прогрева, смысл t, целые буферы")
for d in (0.999, 0.99):
    assert all(ema_decay_at(d, t) == min(d, (1 + t) / (10 + t)) for t in range(30000))
    print(f"  decay {d}: прогрев действует при t < {t_star(d)} (decay_t(0) = {ema_decay_at(d, 0):.4f}, decay_t(100) = {ema_decay_at(d, 100):.4f})")
# AveragedModel: 1-е обновление — копия; k-е (k ≥ 2) — lerp с decay_t, t = n_averaged = k − 1
for d, warm in ((0.999, True), (0.99, True), (0.999, False)):
    m = torch.nn.Linear(1, 1, bias=False).double()
    ema = torch.optim.swa_utils.AveragedModel(m, multi_avg_fn=ema_avg_fn(d, warm), use_buffers=True)
    ref, err = None, 0.0
    for k in range(1, 3001):
        with torch.no_grad():
            m.weight.fill_(float(k) ** 0.5)
        ema.update_parameters(m)
        w = float(k) ** 0.5
        dt = ema_decay_at(d, k - 1) if warm else d
        ref = w if ref is None else dt * ref + (1 - dt) * w
        err = max(err, abs(float(ema.module.weight.detach()) - ref))
    print(f"  AveragedModel decay {d}, прогрев {warm}: |EMA − рекурсия (t = k − 1, k = 1 — копия)| ≤ {err:.1e} на 3 000 шагах")
x = torch.arange(0, 20001)
for d in (0.999, 0.99):
    for warm in (False, True):
        e = x.clone()
        ema_avg_fn(d, warm)([e], [x.clone()], torch.tensor(5))
        print(f"  целые 0…20 000, одно обновление, decay {d}, прогрев {warm}: изменилось {int((e != x).sum())} значений"
              + (f" (первые: {x[e != x][:5].tolist()})" if (e != x).any() else ""))
for flags in ({}, {"parts": True}, {"cos_head": True}, {"parts": True, "cos_head": True}):
    net = sohn.SohnNet(10, 192, **flags)
    ints = [(n, b) for n, b in net.named_buffers() if not torch.is_floating_point(b)]
    flt = sum(1 for _, b in net.named_buffers() if torch.is_floating_point(b))
    bad = {}
    for d in (0.999, 0.99):
        es = [b.clone() for _, b in ints]
        if es:
            torch.optim.swa_utils.get_ema_multi_avg_fn(d)(es, [b.clone() for _, b in ints], None)
        bad[d] = sum(int((e != b).sum()) for e, (_, b) in zip(es, ints))
    print(f"  SohnNet{flags or '(по умолчанию)'}: буферов float {flt}, целых {len(ints)} {[n for n, _ in ints][:4]}{'…' if len(ints) > 4 else ''};"
          f" без прогрева испортилось бы элементов: 0,999 → {bad[0.999]}, 0,99 → {bad[0.99]}")

print("\n== 2. Шаги Sohn = epoch × ⌊n / batch⌋ (m13_train: fold — только train; all — train + val; sb — train pretrain_all)")
E, B = sohn.CFG["epoch"], sohn.CFG["batch_size"]
print(f"  sohn.CFG: epoch {E}, batch_size {B}; порог EMA {COMMON['ema_short_below']} шагов → {COMMON['ema_decay']} / {COMMON['ema_decay_short']}")
sizes = json.loads((OM / "splits/sizes.json").read_text())
for ds in ("slovo", "bukva"):
    f = json.loads((OM / f"splits/{ds}_p5.json").read_text())["folds"]
    rows = []
    for k in sorted(x for x in f if x.startswith("fold")):
        n = len(f[k]["train"])
        assert n == sizes[f"{ds}_p5"][k]["train"][0], (ds, k)
        s = E * (n // B)
        d = decay_of(s, COMMON)
        rows.append((k, n, s, d, min(t_star(d), s) / s))
    print(f"  {ds}: " + "; ".join(f"{k} n {n} → {s} шагов, decay {d}, прогрев {p:.0%} шагов" for k, n, s, d, p in rows)
          + f"; Σ {sum(r[2] for r in rows)}")
f = json.loads((V2 / "splits/uzsl_loso.json").read_text())["folds"]
st = sorted(E * ((len(v["train"]) + len(v["val"])) // B) for v in f.values())
print(f"  корпус (11 фолдов, train + val; без названий): шаги {st}, Σ {sum(st)}; decay 0,999 у {sum(s >= 3000 for s in st)} из {len(st)};"
      f" прогрев (0,99) — {min(t_star(0.99) / s for s in st):.0%}–{max(t_star(0.99) / s for s in st):.0%} шагов")
nsb = sum(len(json.loads((OM / f"splits/{d}_pretrain_all.json").read_text())["folds"]["all"]["train"]) for d in ("slovo", "bukva"))
s = E * (nsb // B)
print(f"  S+B: n {nsb} → {s} шагов, decay {decay_of(s, COMMON)}, прогрев {t_star(decay_of(s, COMMON)) / s:.0%} шагов")
print(f"  горизонт 1/(1 − decay): 0,999 → {1 / (1 - 0.999):.0f} шагов, 0,99 → {1 / (1 - 0.99):.0f} шагов")

print("\n== 3. Общий рецепт: шаблоны группы 1 (config.yaml; ключа нет → configs/common.yaml, как turnir_train.with_defaults)")
for c in te.C:
    if c["kind"] != "run":
        continue
    srcs = [g for g in c["src"] if not g.startswith(("TURNIR/", "M13/", "UzSohn/"))]
    cfgs = [yaml.safe_load(Path(p).read_text()) for g in srcs for p in sorted(glob.glob(str(te.RUNS / g / "config.yaml")))]
    if not cfgs:
        continue
    tcs = [{**COMMON, **x["train"]} for x in cfgs]
    stp = [t["steps"] for t in tcs]
    val = lambda f: sorted({str(f(x, t)) for x, t in zip(cfgs, tcs)})  # noqa: E731
    print(f"  {c['ds']}/{c['model']}/{c['recipe']}/{c['cond']} ({', '.join(g.split('/*')[0].replace('../../uzsign_ai_v2/', 'V2:') for g in srcs)};"
          f" {len(cfgs)} конф.): шаги {min(stp)}–{max(stp)}; ema {val(lambda x, t: t['ema'])}; прогрев {val(lambda x, t: t['ema_warmup'])};"
          f" decay {val(lambda x, t: decay_of(t['steps'], t))}; T {val(lambda x, t: x['model'].get('T'))}; пакет {val(lambda x, t: t['batch_size'])};"
          f" голов {val(lambda x, t: len(x['data'].get('sets', {})) or 1)}; select {val(lambda x, t: t['select'])}; merge_val {val(lambda x, t: t['merge_val'])}")

print("\n== 4. Предобучение S+B общим рецептом (pre_tpl): доли пакета (train.shares) — по разбиению и по batch_share прогона")
for p in ("M8/m83_pretrain_sb_a1", "M9/pretrain_slovo_bukva_a2", "M8/m83_pretrain_sb_a1_big", "M11P/pretrain_sb_slgcn_orig"):
    d = te.RUNS / p / "all/seed0"
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    tc = {**COMMON, **cfg["train"]}
    sets = cfg["data"]["sets"]
    n = [len(json.loads((OM / v["split"]).read_text())["folds"][v["fold"]]["train"]) for v in sets.values()]
    got = json.loads((d / "metrics.json").read_text()).get("batch_share")
    print(f"  {p}: головы {list(sets)}; train {n} → shares {shares(n, tc['batch_size'])} (прогон: {got}); шаги {tc['steps']};"
          f" ema {tc['ema']}, прогрев {tc['ema_warmup']}, decay {decay_of(tc['steps'], tc)}; T {cfg['model'].get('T')}")

print("\n== 5. Что меняет прогрев при бюджетах Sohn: веса итераций θ_k в EMA на последнем шаге S (точный расчёт по рекурсии)")
import numpy as np  # noqa: E402


def weights(S: int, d: float, warm: bool) -> np.ndarray:
    """e_1 = θ_1; e_k = δ·e_{k−1} + (1 − δ)·θ_k, δ = decay_t(t = k − 1) (как AveragedModel + ema_avg_fn, проверено в разд. 1)."""
    dl = np.array([ema_decay_at(d, t) if warm else d for t in range(1, S)])   # δ при обновлениях k = 2…S
    tail = np.concatenate([np.cumprod(dl[::-1])[::-1], [1.0]])               # Π δ от k до S − 1 (индекс k − 1)
    w = np.empty(S)
    w[0] = tail[0]
    w[1:] = (1 - dl) * tail[1:]
    assert abs(w.sum() - 1) < 1e-9
    return w


for name, S, spe in (("Slovo", 10500, 35), ("Slovo", 10800, 36), ("S+B", 13800, 46), ("Bukva", 1500, 5), ("корпус", 1200, 4),
                     ("корпус", 1800, 6)):
    d = decay_of(S, COMMON)
    a, b = weights(S, d, True), weights(S, d, False)
    last10 = lambda w: w[-S // 10:].sum()  # noqa: E731
    init = {ep: (weights(ep * spe, d, True)[0], weights(ep * spe, d, False)[0]) for ep in (1, 10, 30) if ep * spe >= 2}
    print(f"  {name} S {S}, decay {d}: TV(с прогревом, без) = {0.5 * np.abs(a - b).sum():.2e}; вес последних 10 % шагов {last10(a):.4f} / {last10(b):.4f};"
          f" вес θ_1 на конце: {a[0]:.1e} / {b[0]:.1e}; вес θ_1 в EMA после эпохи 1/10/30 (с прогревом / без): "
          + ", ".join(f"{x:.1e}/{y:.2f}" for x, y in init.values()))
