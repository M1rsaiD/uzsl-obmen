"""к3 0056: полнота G3_CODE (П32, к2 0176 разд. 2.3) — своим исполнением. Порт turnir_sohn_shared.py (fc5d15a; в c9a9c2a →
ae84ddb изменились только G3_CODE и документация — исполняемый путь тот же) на дереве рычагов (sohn.py 7f1ac1f, m13_train.py
5f83fec) с настоящими mpk.sohn, mpk.augment, mpk.train (моя копия 7563b5e), m13_build и uzsign_e1.train — заглушки.
Установка (синтетика, cfg) — строки 1–46 моего 0054_proverki/g3_lev_k3.py, дословно. Под sys.setprofile («call») — предобучение
и дообучение с init_from, заморозкой и разморозкой; печать: какие файлы дерева исполнялись, есть ли они в новом G3_CODE.
Запуск (не -I): PYTHONPATH=<стабы>:<src> python g3_cov_k3.py <корень дерева> <0054_proverki/g3_lev_k3.py>"""
import os, sys
from pathlib import Path
root, lev = Path(sys.argv[1]).resolve(), sys.argv[2]
src = "".join(open(lev, encoding="utf-8").readlines()[:46])
sys.argv = [lev, str(root)]
g = {"__name__": "g3_setup", "__file__": lev}
exec(compile(src, lev, "exec"), g)
P, cfg, dS, FT0, tmp, cf, pre_set = (g[k] for k in ("P", "cfg", "dS", "FT0", "tmp", "cf", "pre_set"))
G3_NEW = ("src/mpk/sohn.py", "scripts/m13_train.py", "scripts/turnir_sohn_shared.py", "src/mpk/train.py", "src/mpk/baselines.py",
          "src/mpk/augment.py", "scripts/m13_build.py", "../uzsign_e1/src/uzsign_e1/train.py")   # к2 0176 разд. 2.3 = дифф 0176:568–571
seen = set()
sys.setprofile(lambda fr, ev, arg: ev == "call" and seen.add(fr.f_code.co_filename))
try:
    P.run(cfg(pre_set(dS), 24, 8, FT0), 0, tmp / "pre", None, 0, cf)
    FT = dict(init_from=str(tmp / "pre/best.pt"), freeze_steps=6, unfreeze_every=4, backbone_lr_mult=0.1)
    P.run(cfg(dS, 20, 7, FT), 1, tmp / "ft", None, 0, cf)
finally:
    sys.setprofile(None)
stubs = Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()
ran = sorted({os.path.realpath(f) for f in seen if os.path.isfile(f)})
mine = [f for f in ran if f.startswith((str(root) + os.sep, str(stubs) + os.sep))]
# соответствие моей сборки дереву ПК: заглушки stubs/m13_build.py, stubs/uzsign_e1/train.py — scripts/m13_build.py,
# ../uzsign_e1/src/uzsign_e1/train.py; e1util.py копии 7563b5e — то, что на ПК mpk.train берёт из uzsign_e1 (к2 0176 разд. 3.2)
MAP = {"stubs/m13_build.py": "scripts/m13_build.py", "stubs/uzsign_e1/train.py": "../uzsign_e1/src/uzsign_e1/train.py",
       "stubs/uzsign_e1/__init__.py": "(пакет uzsign_e1, __init__)", "src/mpk/e1util.py": "../uzsign_e1/src/uzsign_e1/train.py (на ПК)"}
print("исполнялись файлы дерева (под профилем, событие call):")
bad = []
for f in mine:
    rel = os.path.relpath(f, stubs.parent if f.startswith(str(stubs)) else root)
    pc = MAP.get(rel, rel)
    ok = pc.split(" (")[0] in G3_NEW
    bad += [] if ok or pc.startswith("(") else [rel]
    print(f"  {rel:32s} → {pc:45s} {'в G3_CODE' if ok else 'НЕТ в G3_CODE'}")
print("вне G3_CODE:", bad or "нет")
print("в G3_CODE, но не исполнялись:", sorted(set(G3_NEW) - {MAP.get(os.path.relpath(f, stubs.parent if f.startswith(str(stubs)) else root), os.path.relpath(f, root)).split(" (")[0] for f in mine}) or "нет")
