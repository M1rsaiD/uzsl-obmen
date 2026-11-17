"""Задача B (к2, 16.11): риск ложного принятия и мощность отбора Э2 при k = 5, 8, 13 рычагах. Только ЦП, Монте-Карло.
Правило — черновик поправки 4 к турниру, П22.4: Δ̄ ≥ THR = 1,0081957…; Δ_c > 0 хотя бы в 6 из 10 ячеек (5 фолдов Slovo + 5 Bukva);
вариант (а) — ещё Δ̄_Slovo ≥ 0, вариант (б) — без него. Схема — жадная, по одному рычагу (22.3), общий контроль на затравке 48 (22.5).
Счёт ячейки — macro top-1 по исполнителям валидации фолда (22.4).
Модель шума: s(j, c) = m(j, c) + ε(j, c); ε независим по плечам (ρ пары = 0, как в этапе 1; rho_check.out) и по ячейкам (доля общего
сдвига на валидации ≈ 0, analyze_noise.out); σ_c² = c_ds · (1/S²) Σ_s p_s(1 − p_s)/n_s · 10⁴ — состав валидации фолда точно
(splits/*_p5.json), c_ds — множитель «перевёрнутых записей» из разброса затравок вал. micro старых серий (noise_summary.json).
Проверка симулятора: модель M1 этапа 1 (σ по S-кодам, inputs_by_scode.json) должна дать α₁ ≈ 0,108, α₅ ≈ 0,320 (поправка 2, 3.5)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
THR, MINB = 1.0081957667939179, 6
N, CH = 200_000, 50_000
KMAX, KS, MUS = 13, (5, 8, 13), (0.5, 1.0, 2.0)
D = json.load(open(HERE / "noise_records.json"))
S = json.load(open(HERE / "noise_summary.json"))
FOLDS = [f"fold{i}" for i in range(1, 6)]


def accept(d, rule_a):
    ok = (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)
    return ok & (d[:, :5].mean(1) >= 0) if rule_a else ok


def run(gen, rule_a, seed, n=N):
    """gen(rng, m) → (m, 10) шум одного плеча. Жадная цепочка из KMAX нулевых рычагов; мощность — рычаг с эффектом μ
    на позиции 1 и на позиции k (после k − 1 нулевых, жадно)."""
    rng = np.random.default_rng(seed)
    first = np.zeros(KMAX + 1)                       # first[k] — P(≥ 1 ложного за k шагов)
    nfalse = np.zeros(KMAX + 1); ge2 = np.zeros(KMAX + 1)
    alpha1 = 0.0
    pw = defaultdict(float)
    for _ in range(n // CH):
        if hasattr(gen, "new_block"):
            gen.new_block(rng, CH)                   # дискретная модель: свои «трудные» записи на каждый блок серий
        e0 = gen(rng, CH)
        cur = e0.copy(); anyacc = np.zeros(CH, bool); cnt = np.zeros(CH)
        for j in range(1, KMAX + 1):
            # мощность на позиции j: рычаг с μ против текущей после j − 1 нулевых шагов (свой шум)
            if j in (1,) + KS:
                ej = gen(rng, CH)
                for mu in MUS:
                    pw[(j, mu)] += accept(ej + mu - cur, rule_a).sum()
            e = gen(rng, CH)
            ok = accept(e - cur, rule_a)
            if j == 1:
                alpha1 += ok.sum()
            anyacc |= ok; cnt += ok
            cur = np.where(ok[:, None], e, cur)
            first[j] += anyacc.sum(); nfalse[j] += cnt.sum(); ge2[j] += (cnt >= 2).sum()
    return {"alpha1": alpha1 / n, "alpha_k": {k: first[k] / n for k in range(1, KMAX + 1)},
            "E_false": {k: nfalse[k] / n for k in range(1, KMAX + 1)}, "P_ge2": {k: ge2[k] / n for k in range(1, KMAX + 1)},
            "power": {f"pos{j}_mu{mu}": pw[(j, mu)] / n for j in (1,) + KS for mu in MUS}}


def normal_gen(sig, f=0.0):
    sig = np.asarray(sig, float)
    def g(rng, m):
        z = rng.standard_normal((m, 10))
        if f > 0:   # общий сдвиг плеча внутри набора (Slovo — ячейки 0–4, Bukva — 5–9)
            c = rng.standard_normal((m, 2))
            z = np.sqrt(1 - f) * z + np.sqrt(f) * np.repeat(c, 5, axis=1)
        return z * sig
    return g


# --- состав валидации и точность исполнителей (среднее по всем прогонам старых серий, по набору) ---
acc = {"slovo": defaultdict(list), "bukva": defaultdict(list)}
for r in D["records"]:
    for sid, (a, n) in r["per_signer"].items():
        acc[r["ds"]][int(sid)].append(a)
cells = []                                           # (набор, фолд, [n_s], [p_s])
for ds in ("slovo", "bukva"):
    for f in FOLDS:
        comp = D["composition"][ds][f]["val"]
        ns = np.array([comp[s] for s in comp], float)
        ps = np.array([np.clip(np.mean(acc[ds][int(s)]), 0.02, 0.98) for s in comp])
        cells.append((ds, f, ns, ps))


def sig_from_c(cS, cB):
    return [np.sqrt((cS if ds == "slovo" else cB) * (ps * (1 - ps) / ns).sum() / len(ns) ** 2 * 1e4) for ds, f, ns, ps in cells]


def discrete_gen(cS, cB):
    """Записи: доля «неустойчивых» u_s = 4c·p_s(1 − p_s) (Var записи по затравкам = c·p(1 − p)); у неустойчивой — Бернулли(½)
    в каждом плече; число неустойчивых U_s ~ Bin(n_s, u_s) — своё в каждой серии, общее у плеч серии (одни записи трудны всем)."""
    n = np.concatenate([c[2] for c in cells]); p = np.concatenate([c[3] for c in cells])
    cS_ = np.concatenate([[cS if c[0] == "slovo" else cB] * len(c[2]) for c in cells])
    u = np.minimum(1.0, 4 * cS_ * p * (1 - p))
    W = np.zeros((len(n), 10))
    i = 0
    for k, c in enumerate(cells):
        W[i:i + len(c[2]), k] = 100.0 / len(c[2]) / c[2]; i += len(c[2])
    state = {}
    def g(rng, m):
        return rng.binomial(state["U"], 0.5) @ W     # U — общее у всех плеч серии
    def new_block(rng, m):
        state["U"] = rng.binomial(n.astype(np.int64), u, size=(m, len(n)))
    g.new_block = new_block
    return g


res = {}
lines = []
P = lines.append
# --- 0. проверка симулятора: модель M1 этапа 1 ---
inp = json.load(open(HERE.parent.parent / "0154/risk/inputs_by_scode.json"))
sd1 = [inp["by_signer"][s]["sd_seeds_pp"] for s in inp["order"]]
chk = run(normal_gen(sd1), False, 1)
P(f"[0] Проверка: модель M1 этапа 1 (ALL-UZ, 10 V_k): α₁ = {chk['alpha1']:.4f} (поправка 2: 0,108), α₄ = {chk['alpha_k'][4]:.4f} "
  f"(0,284), α₅ = {chk['alpha_k'][5]:.4f} (0,320); σ(Δ̄) = {np.sqrt(2 * np.sum(np.square(sd1))) / 10:.3f}")
assert abs(chk["alpha1"] - 0.108) < 0.004 and abs(chk["alpha_k"][5] - 0.320) < 0.006, "симулятор не воспроизводит этап 1"
res["check_stage1_M1"] = chk

# --- 1. входы ---
cp = {ds: S[f"{ds}|все"]["c_pooled"] for ds in ("slovo", "bukva")}
models = {
    "база: c из вал. micro старых серий": (cp["slovo"]["val_micro_last"], cp["bukva"]["val_micro_last"]),
    "c Sohn (0,25: ALL-UZ 0,22, S+B 0,27)": (0.25, 0.25),
    "c из тест. macro старых серий (только тест — прокси)": (cp["slovo"]["test_macro"], cp["bukva"]["test_macro"]),
}
base = models["база: c из вал. micro старых серий"]
sig_base = np.array(sig_from_c(*base))
def sd_mean(sig):
    v = np.square(sig)
    return np.sqrt(2 * v.sum()) / 10, np.sqrt(2 * v[:5].sum()) / 5, np.sqrt(2 * v[5:].sum()) / 5
P("[1] σ ячейки (п. п., база ×1): Slovo " + " ".join(f"{x:.2f}" for x in sig_base[:5]) + " | Bukva " +
  " ".join(f"{x:.2f}" for x in sig_base[5:]) + f"; c Slovo {base[0]:.3f}, Bukva {base[1]:.3f}")
P("    СКО Δ̄ / Δ̄_Slovo / Δ̄_Bukva (база ×1): " + " / ".join(f"{x:.3f}" for x in sd_mean(sig_base)))
res["inputs"] = {"c": {k: v for k, v in models.items()}, "sigma_base": sig_base.round(3).tolist(),
                 "sd_mean_base": [round(x, 3) for x in sd_mean(sig_base)]}


def row(name, r):
    a = r["alpha_k"]
    return (f"{name}: α₁ {r['alpha1']:.3f} | α₅ {a[5]:.3f} α₈ {a[8]:.3f} α₁₃ {a[13]:.3f} | E[ложных] {r['E_false'][5]:.2f}/"
            f"{r['E_false'][8]:.2f}/{r['E_false'][13]:.2f} | P(≥2) {r['P_ge2'][5]:.3f}/{r['P_ge2'][8]:.3f}/{r['P_ge2'][13]:.3f} | "
            "мощность поз.1 μ=0,5/1/2: " + "/".join(f"{r['power'][f'pos1_mu{m}']:.2f}" for m in MUS) +
            " | μ=1 на поз. 5/8/13: " + "/".join(f"{r['power'][f'pos{k}_mu1.0']:.2f}" for k in KS))


seed = 100
# --- 2. главная сетка: чувствительность к σ × правило ---
P("[2] Нормальная модель, база c; σ × 0,5 / 1 / 2; правило (а) с Δ̄_Slovo ≥ 0 и (б) без него")
for mult in (0.5, 1.0, 2.0):
    for rule_a in (True, False):
        seed += 1
        r = run(normal_gen(sig_base * mult), rule_a, seed)
        key = f"normal|×{mult}|{'а' if rule_a else 'б'}"
        res[key] = r; P("  " + row(key, r))
# --- 3. устойчивость к модели шума (×1) ---
P("[3] Устойчивость (×1): другие c, σ по фолдам напрямую (путь V), общий сдвиг f = 0,25 внутри набора")
alt = {name: np.array(sig_from_c(*c)) for name, c in models.items() if c != base}
alt["σ по фолдам напрямую: вал. micro → macro (путь V, все условия)"] = np.array(S["slovo|все"]["E2_sd_V"] + S["bukva|все"]["E2_sd_V"])
for name, sig in alt.items():
    P(f"  σ {name}: " + " ".join(f"{x:.2f}" for x in sig) + "; СКО Δ̄ " + f"{sd_mean(sig)[0]:.3f}")
    for rule_a in (True, False):
        seed += 1
        r = run(normal_gen(sig), rule_a, seed); key = f"{name}|{'а' if rule_a else 'б'}"; res[key] = r; P("    " + row(key, r))
for rule_a in (True, False):
    seed += 1
    r = run(normal_gen(sig_base, f=0.25), rule_a, seed); key = f"база, общий сдвиг f=0,25|{'а' if rule_a else 'б'}"
    res[key] = r; P("  " + row(key, r))
# --- 4. дискретная модель (целые записи, мелкие исполнители валидации) ---
P("[4] Дискретная модель (записи валидации, ничья Δ_c = 0 — «не лучше»)")
for name, c in (("база c", base), ("c Sohn 0,25", (0.25, 0.25))):
    for rule_a in (True, False):
        seed += 1
        r = run(discrete_gen(*c), rule_a, seed); key = f"дискретно, {name}|{'а' if rule_a else 'б'}"; res[key] = r
        P("  " + row(key, r))

json.dump(res, open(HERE / "risk_e2_mc.json", "w"), ensure_ascii=False, indent=1, default=float)
print("\n".join(lines))
open(HERE / "risk_e2_mc.out", "w").write("\n".join(lines) + "\n")
