"""Сводная таблица задачи B из risk_e2_mc.json + разбор разницы с этапом 1 (аналитически). Только ЦП."""
import json
from math import erf, sqrt
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = json.load(open(HERE / "risk_e2_mc.json"))
D = json.load(open(HERE / "noise_records.json"))
THR = 1.0081957667939179
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
rows = [("×0,5 (база)", "normal|×0.5"), ("×1 база: c вал. micro 0,44/0,40", "normal|×1.0"), ("×2 (база)", "normal|×2.0"),
        ("×1, c Sohn 0,25", "c Sohn (0,25: ALL-UZ 0,22, S+B 0,27)"),
        ("×1, c тест. macro 0,70/0,46 (тест — прокси)", "c из тест. macro старых серий (только тест — прокси)"),
        ("×1, σ по фолдам напрямую (путь V)", "σ по фолдам напрямую: вал. micro → macro (путь V, все условия)"),
        ("×1, общий сдвиг f = 0,25", "база, общий сдвиг f=0,25"), ("×1, дискретно, база c", "дискретно, база c"),
        ("×1, дискретно, c Sohn", "дискретно, c Sohn 0,25")]
out = ["| модель шума | правило | α₁ | α₅ | α₈ | α₁₃ | E[ложных] k=5/8/13 | P(≥2 ложных) k=13 | мощность, поз. 1: μ=0,5/1/2 | μ=1: поз. 5/8/13 | μ=2: поз. 5/8/13 |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for name, key in rows:
    for rule in ("б", "а"):
        r = R[f"{key}|{rule}"]; a = r["alpha_k"]; p = r["power"]
        out.append(f"| {name} | ({rule}) | {r['alpha1']:.3f} | {a['5']:.3f} | {a['8']:.3f} | {a['13']:.3f} | "
                   f"{r['E_false']['5']:.2f}/{r['E_false']['8']:.2f}/{r['E_false']['13']:.2f} | {r['P_ge2']['13']:.3f} | "
                   f"{p['pos1_mu0.5']:.2f}/{p['pos1_mu1.0']:.2f}/{p['pos1_mu2.0']:.2f} | "
                   f"{p['pos5_mu1.0']:.2f}/{p['pos8_mu1.0']:.2f}/{p['pos13_mu1.0']:.2f} | "
                   f"{p['pos5_mu2.0']:.2f}/{p['pos8_mu2.0']:.2f}/{p['pos13_mu2.0']:.2f} |")
print("\n".join(out))

# разбор: доля прохода по одному среднему и «фильтр 6/10»
print("\nРазбор (правило (б)): P(Δ̄ ≥ THR) по нормальной модели и доля, которую оставляет условие «≥ 6/10»")
sig_b = np.array(R["inputs"]["sigma_base"])
inp = json.load(open(HERE.parent.parent / "0154/risk/inputs_by_scode.json"))
sd1 = np.array([inp["by_signer"][s]["sd_seeds_pp"] for s in inp["order"]])
for name, sig, a1 in (("Э2 база ×1", sig_b, R["normal|×1.0|б"]["alpha1"]),
                      ("Э2 c Sohn", sig_b * np.sqrt(0.25 / np.array([0.443] * 5 + [0.399] * 5)),
                       R["c Sohn (0,25: ALL-UZ 0,22, S+B 0,27)|б"]["alpha1"]),
                      ("этап 1 M1", sd1, R["check_stage1_M1"]["alpha1"]),
                      ("этап 1 M2", sd1 * 0.512 / (np.sqrt((sd1 ** 2).sum()) / 10), 0.0666)):
    s = np.sqrt(2 * (sig ** 2).sum()) / 10
    share = (sig ** 2) / (sig ** 2).sum()
    pm = 1 - Phi(THR / s)
    print(f"  {name}: СКО Δ̄ {s:.3f}; THR/СКО {THR / s:.2f}; P(Δ̄ ≥ THR) {pm:.3f}; α₁ {a1:.3f} → «6/10» оставляет {a1 / pm:.0%}; "
          f"доля Var Δ̄ от 3 самых шумных ячеек {np.sort(share)[-3:].sum():.0%}")

# множитель micro → macro на валидации (биномиальная структура, p̄ исполнителя по старым сериям)
acc = {"slovo": {}, "bukva": {}}
for r in D["records"]:
    for sid, (a, n) in r["per_signer"].items():
        acc[r["ds"]].setdefault(int(sid), []).append(a)
print("\nМножитель СКО macro / СКО micro на валидации (биномиально) и доля Var macro от исполнителей с < 10 записями:")
for ds in ("slovo", "bukva"):
    xs = []
    for f in sorted(D["composition"][ds]):
        c = D["composition"][ds][f]["val"]
        ns = np.array(list(c.values()), float); ps = np.array([np.clip(np.mean(acc[ds][int(s)]), .02, .98) for s in c])
        vm = (ps * (1 - ps) / ns).sum() / len(ns) ** 2; vu = (ns * ps * (1 - ps)).sum() / ns.sum() ** 2
        small = ((ps * (1 - ps) / ns)[ns < 10]).sum() / (ps * (1 - ps) / ns).sum()
        xs.append(f"{f}: ×{np.sqrt(vm / vu):.2f}, {small:.0%} ({int((ns < 10).sum())} из {len(ns)} исп.)")
    print(f"  {ds}: " + "; ".join(xs))
