"""к3 0053 (сверка Б9 0172 по 4.7): какой каталог прогона берёт turnir_val_m13.export (блоб 7718975) у пары (фолд, затравка),
если каталогов два (у ячейки Sohn на Slovo источники «UzSohn/slovo/…» и «M13/slovo/…», turnir_eval.py строка 94), и совпадает ли он
с каталогом теста export_cell (первый с probs.npz, строки 353–357). Синтетика: файлы-пустышки, logits_fn подменён; стенд 964ecba
(+ a1deebc) со стабом строк 42–43, как у к1 0152. Запуск (не -I): PYTHONPATH=<scripts> python b9_dir_k3.py"""
import json, tempfile
from pathlib import Path
import numpy as np
import turnir_eval as te
import turnir_val_m13 as b9

root = Path(tempfile.mkdtemp())
te.ROOT, te.OUT = root / "osn", root / "osn/runs/TURNIR"
te.PRED = te.OUT / "preds"
cell = next(c for c in te.C if c["kind"] == "m13" and c["ds"] == te.P and c["model"] == "Sohn")
print("ячейка Sohn на Slovo, источники:", cell["src"])
d1, d2 = (te.ROOT / "runs" / r / "slovo/fold1/seed42" for r in ("UzSohn", "M13"))
te.C, te.src_dirs = [cell], (lambda c: [d1, d2])
saved = []
te.save = lambda out, **kw: saved.append(kw["run"])

def case(name, d1_files, d1_ok):
    for d in (d1, d2):
        d.mkdir(parents=True, exist_ok=True)
        for f in ("probs.npz", "metrics.json", "best.pt", "log.csv"):
            (d / f).unlink(missing_ok=True)
            if d is d2 or f in d1_files:
                (d / f).write_text("x")
    for f in te.cell_dir(cell).glob("*"):
        f.unlink()
    saved.clear()
    def logits_fn(d, threads):   # d1: сверка с log.csv не сходится (если d1_ok = False); d2 — сходится
        ok = d != d1 or d1_ok
        n = 100   # допуск сверки — max(2; 0,5 %·n) записей: при n = 100 расхождение 0,5 — это 50 записей
        return dict(ids=np.array([f"r{k}" for k in range(n)]), logits=np.zeros((n, 3)), names=["g0", "g1", "g2"],
                    signer=np.full(n, "S1"), gloss=np.full(n, "g0"), top1=0.5, top1_log=0.5 if ok else 0.0, epoch=1)
    orig_save = te.save
    def save_and_touch(out, **kw):
        orig_save(out, **kw); Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text("x")
    te.save = save_and_touch
    bad = b9.export(1, None, logits_fn)
    te.save = orig_save
    test_dir = d1 if (d1 / "probs.npz").is_file() else d2   # export_cell: первый каталог с probs.npz (и metrics.json)
    got = [Path(r).parts[-4] for r in saved]
    print(f"{name}: тест (export_cell) — {test_dir.parts[-4]}; валидация (Б9) — {got or 'нет'}; не сошлось {bad} → "
          f"{'тот же прогон' if got == [test_dir.parts[-4]] else 'ДРУГОЙ прогон' if got else 'файла нет'}")

case("1) оба полные, d1 сходится     ", ("probs.npz", "metrics.json", "best.pt", "log.csv"), True)
case("2) d1 не сходится с log.csv    ", ("probs.npz", "metrics.json", "best.pt", "log.csv"), False)
case("3) у d1 нет best.pt            ", ("probs.npz", "metrics.json", "log.csv"), True)
