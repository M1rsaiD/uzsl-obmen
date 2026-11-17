"""к 22.4 и 22.7 поправки 4 (к3 0042 разд. 1 п. 4, разд. 2.1). Только ЦП, только разбиения и опись записей.
Читает: OM splits/{slovo,bukva}_p5.json, поле clips кэша OM cache/M13/{slovo,bukva}.npz (без точек), для сверки —
noise_records.json счёта к2 (0156/risk_e2). НЕ читает runs/**. Печатает только агрегаты: ни одного кода исполнителя.
Запуск: CUDA_VISIBLE_DEVICES= nice -n 19 python3.14.7 -I val_composition.py > val_composition.out"""
import hashlib, io, json, zipfile
from collections import Counter
from pathlib import Path
import numpy as np

OM = Path("/home/error_404/uzsign/osnovnaya_model")
K2 = Path(__file__).resolve().parents[2] / "0156/risk_e2/noise_records.json"
P = {"slovo": 0.7, "bukva": 0.9}  # допущение: одна точность p на всех исполнителей валидации набора
DS = ("slovo", "bukva")


def clips(ds):  # только опись записей: sample_id → signer_id
    z = zipfile.ZipFile(OM / f"cache/M13/{ds}.npz")
    return {c["sample_id"]: c["signer_id"] for c in json.loads(str(np.load(io.BytesIO(z.read("clips.npy")))))}


sig, sp, sizes = {}, {}, json.load(open(OM / "splits/sizes.json"))
for ds in DS:
    sig[ds] = clips(ds)
    raw = (OM / f"splits/{ds}_p5.json").read_bytes()
    sp[ds] = json.loads(raw)
    print(f"{ds}: splits sha256 {hashlib.sha256(raw).hexdigest()[:16]}…, записей в кэше M13 {len(sig[ds])}")

# --- самопроверка разбиений: части = весь кэш, исполнители не делятся, тест фолда k = группа k-1
for ds in DS:
    d = sp[ds]
    for k, f in d["folds"].items():
        ids = [i for p in ("train", "val", "test") for i in f[p]]
        assert len(ids) == len(set(ids)) and set(ids) == set(sig[ds]), (ds, k)
        S = {p: {sig[ds][i] for i in f[p]} for p in ("train", "val", "test")}
        assert not (S["train"] & S["val"] or S["train"] & S["test"] or S["val"] & S["test"]), (ds, k)
        assert S["test"] == {s for s, g in d["groups"].items() if g == int(k[4:]) - 1}, (ds, k)
        assert set(d["always_train"]) <= S["train"], (ds, k)
        for p in ("val", "test"):
            assert sizes[f"{ds}_p5"][k][p][:2] == [len(f[p]), len(S[p])], (ds, k, p)
print("самопроверка разбиений: ok (части = кэш; части не делят исполнителей; тест фолда k = группа k; размеры = sizes.json)\n")

# --- 1. состав валидации Э2 по фолдам
k2 = json.load(open(K2))["composition"]
print("1. Валидация Э2 по фолдам (протокол П). Биномиальная модель: Var macro = p(1−p)/N² · Σ 1/n_i, Var micro = p(1−p)/Σ n_i;")
print("   при одной p на всех доля Var macro от исполнителей с n < 10 и множитель СКО macro/micro от p НЕ зависят;")
print("   p (Slovo 0,7 / Bukva 0,9) входит только в абсолютные СКО. Сверка состава со счётом к2 (noise_records.json, кэш МПК).")
for ds in DS:
    for k, f in sp[ds]["folds"].items():
        n = np.array(sorted(Counter(sig[ds][i] for i in f["val"]).values()), float)
        N, p = len(n), P[ds]
        small = n < 10
        share = (1 / n[small]).sum() / (1 / n).sum()
        mult = np.sqrt(n.sum() * (1 / n).sum()) / N
        sd_ma = 100 * np.sqrt(p * (1 - p) * (1 / n).sum()) / N
        sd_mi = 100 * np.sqrt(p * (1 - p) / n.sum())
        same = sorted(k2[ds][k]["val"].values()) == n.astype(int).tolist()
        print(f"  {ds} {k}: исп. {N}, записей {int(n.sum())}; на исп. min/мед/max {int(n[0])}/{np.median(n):g}/{int(n[-1])}; "
              f"< 10 записей — {int(small.sum())} исп. ({int(n[small].sum())} зап.); доля Var macro от них {share:.0%}; "
              f"вес в micro: их {n[small].sum() / n.sum():.1%}, 3 крупнейших {n[-3:].sum() / n.sum():.0%}, N_eff = (Σn)²/Σn² = {n.sum() ** 2 / (n ** 2).sum():.1f}; ×{mult:.2f}; СКО macro/micro {sd_ma:.2f}/{sd_mi:.2f} п. п.; вес 1 записи у исп. с min — {100 / N / n[0]:.2f} п. п.; "
              f"состав = счёт к2: {same}")

for ds in DS:  # одни и те же ли люди «крупнейшие» в разных фолдах
    top = [Counter(sig[ds][i] for i in f["val"]).most_common(3) for f in sp[ds]["folds"].values()]
    u = {s for t in top for s, _ in t}
    allv = Counter(sig[ds][i] for f in sp[ds]["folds"].values() for i in f["val"])
    print(f"  {ds}: различных исполнителей среди 3 крупнейших по 5 фолдам — {len(u)} на 15 мест (5 × 3); их доля записей во всех 5 валидациях "
          f"(с повторами) {sum(allv[s] for s in u) / sum(allv.values()):.0%}")

# --- 2. тест турнира = объединение тестов 5 фолдов (все исполнители, кроме always_train); «чистые» — вне всех валидаций Э2
print("\n2. Тест набора турнира = объединение тестов 5 фолдов протокола П (каждый исполнитель — в тесте ровно одного фолда).")
V = {ds: set().union(*({sig[ds][i] for i in f["val"]} for f in sp[ds]["folds"].values())) for ds in DS}
both = V["slovo"] | V["bukva"]
print(f"  коды в обоих наборах: {len(set(sig['slovo'].values()) & set(sig['bukva'].values()))}")
for ds in DS:
    other = "bukva" if ds == "slovo" else "slovo"
    T = {sig[ds][i]: k for k, f in sp[ds]["folds"].items() for i in f["test"]}   # исполнитель → фолд теста
    rec = Counter(sig[ds][i] for f in sp[ds]["folds"].values() for i in f["test"])
    nT = sum(rec.values())
    for name, ex in ((f"(А) вне валидации {ds} (как к3)", V[ds]), (f"(Б) вне валидации Slovo и Bukva (по коду)", both)):
        cl = sorted(s for s in T if s not in ex)
        r = sum(rec[s] for s in cl)
        h = hashlib.sha256("\n".join(cl).encode()).hexdigest()[:16]
        per = "; ".join(f"{k}: {sum(T[s] == k for s in cl)}/{sum(T[s] == k for s in T)} исп., "
                        f"{sum(rec[s] for s in cl if T[s] == k)}/{len(f['test'])} зап."
                        for k, f in sp[ds]["folds"].items())
        print(f"  {ds} {name}: {len(cl)} из {len(T)} исп., {r} из {nT} записей теста = {r / nT:.1%} "
              f"(в валидации: {len(T) - len(cl)} исп., {nT - r} зап. = {(nT - r) / nT:.1%}); sha256 списка {h}…")
        print(f"      по фолдам теста: {per}")
    extra = {s for s in T if s in V[other] and s not in V[ds]}
    print(f"  {ds}: исп. теста, которые есть только в валидации {other} (разница А − Б): {len(extra)}, "
          f"записей {sum(rec[s] for s in extra)}")
