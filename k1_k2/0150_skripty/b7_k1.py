"""Б7, к1: исполняю строки пометок 22.8 / 22.7 дословно из k2_k1/0168_proverki/b7_stroki.txt (turnir_eval.py 0b22f9b).
Строки берутся из файла по номерам (не переписаны руками): 550–551 (add), 632 (mark), 686–688 (not_nested).
Запуск: python3 -I b7_k1.py <путь к b7_stroki.txt>"""
import json, re, sys, tempfile, textwrap
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
def lines(a, b):  # строки turnir_eval.py с номерами a..b (первая секция файла, где они есть)
    got = {}
    for s in src:
        m = re.match(r"\s*(\d+)\t(.*)$", s)
        if m and a <= int(m.group(1)) <= b and int(m.group(1)) not in got:
            got[int(m.group(1))] = m.group(2)
        if s.startswith("## tests/"):
            break  # дальше — номера строк теста, не стенда
    assert sorted(got) == list(range(a, b + 1)), (a, b, sorted(got))
    return [got[i] for i in range(a, b + 1)]

# --- 22.7: not_nested (686–688), RUNS — временный каталог
body = textwrap.dedent("\n".join(lines(686, 688)))
ns = {"json": json}
exec("def not_nested():\n" + textwrap.indent(body, "    "), ns)
STEPS = ["U02-1", "U02-2", "U01a", "U04", "U01b", "r2", "r3", "r4"]
def case(files):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t) / "UZSOHN"; d.mkdir()
        for name, acc, wg in files:
            (d / f"decide_e2_{name}.json").write_text(json.dumps({"accepted": acc, "without_group": wg}))
        ns["RUNS"] = Path(t)
        return ns["not_nested"]()
eq = lambda acc: {str(j): {"accepted": acc} for j in range(1, 6)}  # noqa: E731
res = {
    "нет файлов": case([]),
    "8 шагов, все j совпали": case([(s, s == "U01a", eq(s == "U01a")) for s in STEPS]),
    "8 шагов, у r3 j=3 иначе": case([(s, s == "U01a", eq(s == "U01a") | ({"3": {"accepted": True}} if s == "r3" else {}))
                                      for s in STEPS]),
    "7 шагов (нет r3, он был бы иначе)": case([(s, s == "U01a", eq(s == "U01a")) for s in STEPS if s != "r3"]),
}
for k, v in res.items():
    print(f"not_nested — {k}: {v}")
assert res == {"нет файлов": False, "8 шагов, все j совпали": False, "8 шагов, у r3 j=3 иначе": True,
               "7 шагов (нет r3, он был бы иначе)": False}

# --- 22.7: пометка в паре (632)
expr = re.search(r'mark = (.*)$', lines(632, 632)[0]).group(1)
P = "Slovo"  # протокол П — Slovo (тест Slovo пересекается с валидациями, 22.7)
marks = {}
for nn in (False, True):
    for ds in ("Slovo", "UzSL"):
        for a, b in (("UzSohn", "A1"), ("UzSohn", "Sohn"), ("A1", "SL-GCN"), ("Sohn", "UzSohn")):
            marks[(nn, ds, a, b)] = eval(expr, {"nn": nn, "ds": ds, "P": P, "a": a, "b": b})
on = sorted(k for k, v in marks.items() if v)
print("пометка «не вложенный» стоит у:", on)
assert on == sorted((True, "Slovo", a, b) for a, b in (("UzSohn", "A1"), ("UzSohn", "Sohn"), ("Sohn", "UzSohn")))

# --- 22.8: подпись и источник (550–551) — для каждой строки сводки
blk = textwrap.dedent("\n".join(lines(550, 551)))
uz = "UzSohn (Э2: U01a, № 3)"
out = {}
for st in ("Э2", "≡", ""):
    for model in ("UzSohn", "Sohn", "наша реализация Sohn", "A1", "SL-GCN"):
        loc = {"st": st, "model": model, "source": "M13", "uz": uz}
        exec(blk, {}, loc)
        out[(st, model)] = (loc["model"], loc["source"])
for k, v in out.items():
    if v != (k[1], "M13"):
        print("изменена строка", k, "→", v)
assert {k for k, v in out.items() if v != (k[1], "M13")} == {("Э2", "UzSohn")}
assert out[("Э2", "UzSohn")] == (uz, "M13; общий рецепт UzSohn не обучался (поправка 4, 22.8)")
print("Б7: все проверки пройдены")
