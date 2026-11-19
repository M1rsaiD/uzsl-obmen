"""к3 0057: П30 и В3, В4 — раздел группы 3 в pairs.md на моей сборке стенда (a3ce665 = 66795c0 + хунк turnir_eval.py 0178),
синтетика selftest (данные есть только у Slovo; Bukva и корпус — «ждёт»).
Печать g3_tail() целиком и проверка по П30 (0165 разд. 5): какие пары, где «↳ +2,71», уровень 95 %, корпус с нуля — отдельно.
Запуск: python pairs_g3_k3.py <scripts>"""
import contextlib, io, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import turnir_eval as te  # noqa: E402

te.OUT = Path(tempfile.mkdtemp()) / "TURNIR"
with contextlib.redirect_stdout(io.StringIO()):
    te.selftest()
g = te.g3_tail()
print("\n".join(g))
print("\nG3_PAIRS:", [(d, a, b) for d, a, b in te.G3_PAIRS])
body = [l for l in g if l.startswith("| ") and not l.startswith("| набор")]
arrow = [l for l in body if "↳" in l]
assert all("Sohn (общий)" in l for l in arrow) and not any("Sohn (родной) −" in l and "↳" in l for l in body)
assert all("ДИ 95 %" in l for l in g if l.startswith("| набор"))
assert g.index("Корпус с нуля:") > 0 and [(d, a, b) for d, a, b in te.G3_PAIRS] == \
    [("Slovo", ("Sohn", "native"), ("Sohn", "shared")), ("Bukva", ("Sohn", "native"), ("Sohn", "shared")),
     ("UzSL", ("Sohn", "native"), ("Sohn", "shared")), ("Slovo", ("Sohn", "shared"), ("SL-GCN", "shared")),
     ("UzSL", ("Sohn", "shared"), ("SL-GCN", "shared"))]
print("sohn_verdict, Sohn (общий) − SL-GCN, lo < 0 < hi:", te.sohn_verdict("Sohn", "SL-GCN", dict(lo=-3.0, hi=1.0)))
print("sohn_verdict, Sohn (общий) − SL-GCN, hi + 2,71 < 0 (держится):", te.sohn_verdict("Sohn", "SL-GCN", dict(lo=-6.0, hi=-4.0)))
print("sohn_verdict, Sohn (общий) − SL-GCN, hi < 0 < hi + 2,71 (В3):", te.sohn_verdict("Sohn", "SL-GCN", dict(lo=-2.0, hi=-1.0)))
print("ВСЁ СОВПАЛО")
