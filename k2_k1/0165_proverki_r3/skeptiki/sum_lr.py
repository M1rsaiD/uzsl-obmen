"""К-4 (|Δs| ≤ Σ lr) на бюджетах Э2: Σ lr по расписанию Sohn (one_cycle_lr из sohn.py ветки d7b0d68, только чтение, ЦП).
Запуск: PYTHONPATH=<wt_e2>/src python3.14.7 -I? — нет: -I сбрасывает PYTHONPATH, поэтому путь — через sys.path из argv[1]."""
import sys
sys.path.insert(0, sys.argv[1])
from mpk import sohn  # noqa: E402
for name, total in (("ALL-UZ (≈ 14 шагов/эп.)", 300 * 14), ("Slovo", 10620), ("Bukva", 1500), ("S+B", 13800)):
    s = sum(sohn.one_cycle_lr(t, sohn.CFG["lr"], total, sohn.CFG["lr_min"]) for t in range(total))
    print(f"{name}: шагов {total}, Σ lr = {s:.2f}; 16 + Σ lr = {16 + s:.1f}")
