"""Скептик ред. 3 (линза «код»): принимает ли m13_train (ветка d7b0d68) команду пробы памяти 22.10 п. 3
`m13_train.py --set slovo --input u01a,u04,u01b --variant 9 --ema --balanced` и что она делает. Записи — синтетика (как
tests/test_sohn.py::_tiny_run_inp), кэш не читается; проверка фолда — по ключам splits/slovo_p5.json (как records, стр. 173).
Выход — только в папку скептика. Аргументы: <worktree> <out>"""
import json
import os
import sys
from pathlib import Path

WT, OUT = sys.argv[1], Path(sys.argv[2])
sys.path[:0] = [WT + "/src", WT + "/scripts", os.path.expanduser("~/uzsign/uzsign_e1/src")]
os.environ["WEEKEND_GATE_NOW"] = "2026-11-10 10:00"   # будний день: weekend_gate.sh не ждёт
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpk import sohn  # noqa: E402
import m13_train as T  # noqa: E402

FOLDS = json.loads((Path(WT) / "splits/slovo_p5.json").read_text())["folds"]
seen = []


def records(set_, fold, pose=False, inp=False):
    seen.append((set_, fold, pose, inp))
    _ = FOLDS[fold]                                     # как records(): folds[fold] — KeyError, если такого фолда нет
    rng = np.random.default_rng(0)
    off = np.concatenate([[0], np.cumsum(rng.integers(8, 40, 96))])
    xy = rng.uniform(0.2, 0.8, (off[-1], sohn.NUM_NODES_POSE, 2)).astype(np.float32)
    clips = [{"sample_id": f"s{i}", "signer_id": f"p{i % 6}", "gloss": f"g{i % 4}"} for i in range(96)]
    r = T.Recs(xy[:, :118].copy(), np.zeros((off[-1], 4), bool), off, xy[:, 118:].copy() if (pose or inp) else None,
               np.full(96, 0.5625, np.float32) if inp else None)
    return r, [c["gloss"] for c in clips], clips, (list(range(64)), list(range(64, 80)), list(range(80, 96)))


T.records = records
PROBE = ["--set", "slovo", "--input", "u01a,u04,u01b", "--variant", "9", "--ema", "--balanced"]


def attempt(name, argv):
    try:
        r = T.main(argv)
        print(f"[{name}] main вернул: {type(r).__name__}{'' if r is None else ' ключи ' + str(sorted(r)[:6])}")
        return r
    except SystemExit as e:
        print(f"[{name}] SystemExit({e.code}) — argparse отказал")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] {type(e).__name__}: {e}")


attempt("1: команда 22.10 п. 3 как есть (+ --plan-only)", PROBE + ["--plan-only"])
attempt("2: + --seed 48 (фолд по умолчанию)", PROBE + ["--seed", "48", "--plan-only", "--out", str(OUT / "p2" / "seed48")])
print("   фолд по умолчанию:", seen[-1][1] if seen else "records не вызван", "| фолды slovo_p5:", list(FOLDS))
p = attempt("3: + --seed 48 --fold fold1 --plan-only", PROBE + ["--seed", "48", "--fold", "fold1", "--plan-only", "--out",
                                                                  str(OUT / "p3" / "seed48")])
if p:
    print(f"   план: n_train {p['n_train']}, шагов {p['steps']}, AWP с шага {p['awp_start_step']}, тестовых записей {len(p['test_ids'])}")
# 4: полный крошечный прогон (все три рычага + полный вход) — что пишет в конце
for k, v in (("batch_size", 64), ("awp_start_epoch", 1), ("dropout_start_epoch", 2)):
    sohn.CFG[k] = v
out = OUT / "p4" / "seed48"
attempt("4: + --fold fold1 --seed 48 --epochs 4 --max-steps 3 (ЦП)", PROBE + ["--seed", "48", "--fold", "fold1", "--epochs", "4",
                                                                              "--max-steps", "3", "--workers", "1", "--out", str(out)])
if (out / "metrics.json").is_file():
    m = json.loads((out / "metrics.json").read_text())
    print("   файлы:", sorted(f.name for f in out.iterdir()))
    print("   metrics: top1 (тест) есть:", "top1" in m, "| n_test", m.get("n_test"), "| ema", m.get("ema"), "| balanced", m.get("balanced"),
          "| flags.part_tokens", m["flags"]["part_tokens"], "| flags.cos_head", m["flags"]["cos_head"], "| channels", m.get("channels"))
    sd = torch.load(out / "last.pt")["model"]
    print("   last.pt: tok.* ", sum(k.startswith("tok.") for k in sd), "ключей; stem.weight —", "stem.weight" in sd,
          "| cos_scale", float(sd["cos_scale"]))
# куда лёг бы прогон без --out и --tag
a = T.argparse.Namespace(tag=None, set="slovo", fold="fold1", seed=48, out=None)
tag = a.tag or a.set
print("без --out/--tag каталог:", T.ROOT / "runs/UzSohn" / tag / a.fold / f"seed{a.seed}")
