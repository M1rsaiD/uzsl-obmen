"""22.7 (Б), к1: e2_outside() стенда 964ecba (строки 695–707) — исполняю дословно на синтетике (коды s01…; файлы во временном каталоге).
Запуск: python3 -I vne_val_k1.py <восстановленный scripts/turnir_eval.py> (k1_k2/0151_skripty/rec_964ecba.sh)"""
import hashlib, json, re, sys, tempfile, textwrap
from pathlib import Path
import numpy as np
src = open(sys.argv[1], encoding="utf-8").read()
fn = re.search(r"^def e2_outside\(\) -> set:\n(?:    .*\n|\n)+?    return set\(out\)\n", src, re.M).group(0)
assert fn.count("\n") == 13, fn.count("\n")   # 695–707
def run(sl, bk, sha):
    with tempfile.TemporaryDirectory() as t:
        R = Path(t); (R / "cache/M13").mkdir(parents=True); (R / "splits").mkdir()
        for d, folds in (("slovo", sl), ("bukva", bk)):
            clips = [{"sample_id": f"{d}{i}", "signer_id": s} for f in folds for part in f.values() for i, s in part]
            np.savez(R / f"cache/M13/{d}.npz", clips=np.array(json.dumps(clips)))
            sp = {"folds": {f"fold{k}": {p: [f"{d}{i}" for i, _ in v] for p, v in f.items()} for k, f in enumerate(folds, 1)}}
            (R / f"splits/{d}_p5.json").write_text(json.dumps(sp))
        ns = {"np": np, "json": json, "hashlib": hashlib, "ROOT": R, "E2_OUT_SHA": sha}
        exec(fn, ns)
        return ns["e2_outside"]()
sha = lambda xs: hashlib.sha256("\n".join(sorted(xs)).encode("utf-8")).hexdigest()  # noqa: E731
# Slovo: s01 — только тест; s02 — тест фолда 1 и валидация фолда 2; s03 — тест Slovo и валидация Bukva; s04 — только валидация
sl = [{"test": [(1, "s01"), (2, "s02"), (3, "s03")], "val": [(4, "s04")]}, {"test": [(5, "s05")], "val": [(6, "s02")]}]
bk = [{"test": [(1, "s09")], "val": [(2, "s03")]}]
got = run(sl, bk, sha({"s01", "s05"}))
print("вне валидаций:", sorted(got)); assert got == {"s01", "s05"}
try:
    run(sl, bk, sha({"s01"})); raise SystemExit("!!! sha не проверен")
except AssertionError as e:
    print("чужой sha → AssertionError:", e)
print("22.7 (Б): определение исполнено — валидации обоих наборов, по коду исполнителя; список сверяется с sha до печати")
