"""к3 0053 (сверка Б9 0172 по 4.7): val_logits готовит recs так же, как строки 0172 (recs.tokens = tokens), и зовёт m13_train.predict.
Вопрос: что будет с прогоном UzSohn в Э1, если № 2 (part_tokens) принят? В main m13_train ставит recs.tokens, recs.part_tokens
(5f83fec, строка 708), а Б9 — только recs.tokens. Синтетика, ЦП; m13_train 5f83fec, sohn.py 7f1ac1f (дерево a3d1247), m13_build заглушён.
Запуск (не -I: нужен PYTHONPATH): PYTHONPATH=<src>:<scripts> python b9_pt_k3.py"""
import sys, types
sys.modules["m13_build"] = types.ModuleType("m13_build")
import numpy as np, torch
import m13_train as mt
from mpk import sohn

class R(list):   # как Recs: элементы (x [T, 122, 2], absent [T, 4]) + aspect
    pass
rng = np.random.default_rng(0)
recs = R((rng.random((40, sohn.NUM_NODES_POSE, 2)).astype(np.float32), np.zeros((40, 4), bool)) for _ in range(6))
recs.aspect = np.ones(6, np.float32)
tokens = ("u01a", "u04")
flags = {k: False for k in mt.IDEA.values()} | {"part_tokens": True}   # как metrics.json["flags"] прогона с № 2
torch.manual_seed(0)
model = sohn.SohnNet(5, sohn.CFG["dim"], channels=sohn.input_channels(tokens), tokens=tokens, **flags)
labels, va = np.array([0, 1, 2, 3, 4, 0]), [0, 1, 2]
for name, set_pt in (("как Б9 0172 (только recs.tokens)", False), ("+ recs.part_tokens = flags['part_tokens'] (как main)", True)):
    recs.tokens = tokens
    if set_pt:
        recs.part_tokens = flags["part_tokens"]
    try:
        z, _ = mt.predict(model, recs, labels, va, torch.device("cpu"), 0.1)
        print(f"{name}: predict прошёл, логиты {z.shape}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {str(e)[:160]}")
