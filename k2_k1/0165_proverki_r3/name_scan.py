"""Сканер имён (задание 0159, п. 4): имена фолдов из всех splits/*.json OM и V2 (кроме fold\\d, all, main, other, S\\d+ —
без учёта регистра) и имена источников корпуса (список — файлом вне папки письма, argv[2], чтобы в proverki их не было) —
ищутся во всех файлах Г/0159 (argv[1]). Печатает только счёт и путь:строка, не сами имена. ЦП, только чтение.
Запуск: nice -n 19 python3.14.7 -I name_scan.py <Г/0159> <файл имён источников> > name_scan.out"""
import json, re, sys
from pathlib import Path

ROOTS = [Path("/home/error_404/uzsign/osnovnaya_model/splits"), Path("/home/error_404/uzsign/uzsign_ai_v2/splits")]
SKIP = re.compile(r"^(fold\d+|all|main|other|S\d+)$", re.I)
SRC = Path(sys.argv[2]).read_text(encoding="utf-8").split()


def fold_keys(o):
    out = set()
    if isinstance(o, dict):
        f = o.get("folds")
        if isinstance(f, dict):
            out |= set(f)
        elif isinstance(f, list):
            out |= {x.get("name") for x in f if isinstance(x, dict) and isinstance(x.get("name"), str)}
    return out


names, nfiles = set(), 0
for r in ROOTS:
    for p in sorted(r.glob("*.json")):
        nfiles += 1
        try:
            names |= fold_keys(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"!!! {p.name}: {type(e).__name__}")
names = {n for n in names if n and not SKIP.match(n)}
print(f"splits-файлов: {nfiles}; имён фолдов к поиску (после исключений): {len(names)}; источников: {len(SRC)}")
pats = [(n, re.compile(re.escape(n), re.I)) for n in sorted(names)] + [(s, re.compile(r"(?<![a-z])" + s, re.I)) for s in SRC]
base = Path(sys.argv[1])
hits = 0
for p in sorted(base.rglob("*")):
    if not p.is_file():
        continue
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        continue
    for i, line in enumerate(lines, 1):
        for n, rx in pats:
            if rx.search(line):
                hits += 1
                print(f"  совпадение ({'источник' if n in SRC else 'фолд'}): {p.relative_to(base)}:{i}")
print(f"совпадений: {hits}")
