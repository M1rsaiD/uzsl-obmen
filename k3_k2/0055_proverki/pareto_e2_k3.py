"""к3 0055: подписи UzSohn после Э2 в сводке стенда → загрузчик и рисунок FORMAT 0062 (к2 0171: «не проверено: загрузчик FORMAT
0062 на новых подписях»). Стенд — моя сборка turnir_eval.py (блоб 66795c0 = a3d1247 + 0172/0174 вне стенда; строки 42–43 —
заглушка), синтетика его же selftest: после неё runs/UzSohn_E2.done = «u01b,r2», Э1 UzSohn — копия SL-GCN. Затем summary ещё
раз (теперь e2_state = «Э2») → turnir_summary.csv → pareto_turnir.load и plot. Запуск: python pareto_e2_k3.py <scripts> <pareto_turnir.py>"""
import contextlib, csv, importlib.util, io, sys, tempfile
from pathlib import Path

scr, pz = sys.argv[1:3]
sys.path.insert(0, scr)
import turnir_eval as te  # noqa: E402

te.OUT = Path(tempfile.mkdtemp()) / "TURNIR"
with contextlib.redirect_stdout(io.StringIO()):
    te.selftest()
print("e2_state после селфтеста:", te.e2_state())
with contextlib.redirect_stdout(io.StringIO()):
    te.summary(0.99)
p = te.OUT / "turnir_summary.csv"
rows = list(csv.DictReader(open(p, encoding="utf-8")))
uz = [r for r in rows if "UzSohn" in r["model"]]
print(f"строк сводки {len(rows)}; с UzSohn {len(uz)}:")
for r in uz:
    print(f"  {r['dataset']} | {r['model']!r} | {r['kind']} | source: {r['source']}")
assert uz and all("UzSohn (Э2: U01b, № 2)" in r["model"] for r in uz)
spec = importlib.util.spec_from_file_location("pareto_turnir", pz)
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
L = pt.load(str(p))
print("pareto_turnir.load: принято строк", len(L))
for r in L:   # цены в синтетике — «нет данных»; чтобы точки и подписи попали на рисунок, задаю условные (не из стенда)
    r["gflops"] = 1.0 + 0.1 * L.index(r)
out = Path(tempfile.mkdtemp()) / "pareto_e2"   # plot пишет <out>.png и <out>.pdf
pt.plot(L, str(out), recipe="native", condition="scratch")
out = out.with_suffix(".png")
print("plot: файл", "есть" if out.is_file() and out.stat().st_size > 0 else "НЕТ", f"({out.stat().st_size} байт)")
print("ВСЁ СОВПАЛО")
