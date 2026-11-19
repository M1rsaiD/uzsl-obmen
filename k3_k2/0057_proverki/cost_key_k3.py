"""к3 0057: П33 — цена Sohn (общий) в сводке по ключу (модель, рецепт), без подстановки родного Sohn. Стенд — моя сборка
turnir_eval.py (66795c0, строки 42–43 — заглушка) + хунк turnir_eval.py диффа 0178 (git apply — чисто) → a3ce665. Синтетика —
его selftest (Sohn (общий) = копия ST-GCN). Затем свой reports/turnir_cost.json в двух видах и summary: (1) только ключ «Sohn»
(старый файл) → строки Sohn/shared — «нет данных», Sohn/native — его цена; (2) + ключ «Sohn/shared» → у shared своя цена,
у ансамблей — сумма по частям. Запуск: python cost_key_k3.py <scripts>"""
import contextlib, csv, io, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import turnir_eval as te  # noqa: E402

te.OUT = Path(tempfile.mkdtemp()) / "TURNIR"
with contextlib.redirect_stdout(io.StringIO()):
    te.selftest()
root = Path(tempfile.mkdtemp()); (root / "reports").mkdir()
te.ROOT = root
base = {"Slovo": {m: {"gflops": g, "latency_ms": 10 * g, "params_m": g / 10} for m, g in
                  (("Sohn", 1.0), ("A1", 3.0), ("SL-GCN", 5.0), ("ST-GCN", 7.0), ("A2", 2.0), ("MPK_large", 9.0), ("E1", 4.0))}}

def rows(cost):
    (root / "reports/turnir_cost.json").write_text(json.dumps(cost))
    with contextlib.redirect_stdout(io.StringIO()):
        te.summary(0.99)
    R = list(csv.DictReader(open(te.OUT / "turnir_summary.csv", encoding="utf-8")))
    return [(r["model"], r["recipe"], r["kind"], r["members"], r["gflops"]) for r in R if r["dataset"] == "Slovo" and "Sohn" in r["model"]]

r1 = rows(base)
print("(1) старый turnir_cost.json — ключа «Sohn/shared» нет:")
for x in r1: print("   ", x)
r2 = rows({"Slovo": {**base["Slovo"], "Sohn/shared": {"gflops": 0.5, "latency_ms": 5.0, "params_m": 0.05}}})
print("(2) + ключ «Sohn/shared» (gflops 0,5):")
for x in r2: print("   ", x)
sh1 = [x for x in r1 if x[1] == "shared"]; sh2 = [x for x in r2 if x[1] == "shared"]
nat1 = [x for x in r1 if x[1] == "native" and x[0] == "Sohn"]; nat2 = [x for x in r2 if x[1] == "native" and x[0] == "Sohn"]
assert len(sh1) == 5 and all(x[4] == "нет данных" for x in sh1), sh1
assert nat1 == nat2 and all(x[4] != "нет данных" for x in nat1), (nat1, nat2)
k = {int(x[3]): float(x[4]) for x in sh2}
assert sorted(k) == [1, 2, 3, 4, 5] and all(abs(v - 0.5 * m) < 1e-9 for m, v in k.items()), k
print("ВСЁ СОВПАЛО")
