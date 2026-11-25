"""Турнир, группа 2 Э1: родные рецепты оригинальных SL-GCN (SAM-SLR @de6c53a6) и ST-GCN (yysijie @221c0e15) ИХ кодом на
всех наборах (решение автора, письмо к1 0064). Код авторов не меняется; запуск — тем же способом, что M11 Г/Г3
(scripts/m11_samslr/run.py, scripts/m11_stgcn/run.py: их main.py / REC_Processor, те же шаблоны рецептов и подмены).

cd osnovnaya_model && PYTHONPATH=src:../uzsign_e1/src python scripts/turnir_their.py one --arch slgcn|stgcn \
    --ds slovo|bukva|uzsl|sb --fold F --seed K [--init путь_весов] --out КАТАЛОГ
Наборы: slovo, bukva — данные M11 (runs/M11/<их каталог>/data/<фолд>, те же, что у прогонов Г/Г3); uzsl — LOSO корпуса
  (uzsign_ai_v2/cache/uzsl.npz, splits/uzsl_loso.json), sb — предобучение Slovo + Bukva (slovo_pretrain_all +
  bukva_pretrain_all, метки «slovo:…»/«bukva:…», 1 034 класса — как M13). Данные корпуса/S+B строятся их же функцией clip
  (build_data.py M11) в runs/TURNIR/their/<arch>/data/<набор>/<фолд>/ (персональные данные, не в git).
Выбор эпохи — как у них: SAM-SLR — лучшая по валидации (Slovo/Bukva, предобучение S+B); ST-GCN — последняя. Корпус: валидация
  LOSO — 1 запись, поэтому train + val и последняя эпоха (как A1 шага 1 — merge_val/last и M13 — fold 'all'); оговорка.
Затравки: у авторов затравка зашита (SAM-SLR init_seed(·) → 1; наш запуск ST-GCN — 0). Затравка k турнира = их затравка + 1000·k
  (подмена torch/numpy/random .seed со сдвигом); k = 0 — ровно прежние прогоны M11 (Г/Г3).
Перенос: веса предобучения S+B своей затравки, классификатор новый (SAM-SLR ignore_weights fc.weight/fc.bias, ST-GCN fcn),
  дальше — их полный рецепт (250 / 50 эпох).
Выход (формат стенда): predictions.csv, probs.npz (ids, probs float16 softmax, classes), summary.json; оборванный каталог —
  в _oborvan_* (перезапуск целиком).
"""
import argparse
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent / "uzsign_ai_v2"
HERE = Path(__file__).resolve().parent
M11DIR = {"slgcn": {"slovo": "samslr", "bukva": "samslr_bukva"}, "stgcn": {"slovo": "stgcn", "bukva": "stgcn_bukva"}}
EPOCHS = {"slgcn": 250, "stgcn": 50}


def seed_offset(k: int) -> None:
    """Затравка k: их seed(·) + 1000·k во всех генераторах (k = 0 — без изменений)."""
    import torch
    off = 1000 * k
    for mod, name in ((torch, "manual_seed"), (torch.cuda, "manual_seed_all"), (np.random, "seed"), (random, "seed")):
        f = getattr(mod, name)
        # воркер DataLoader зовёт np.random.seed(список) из уже сдвинутой базы — пропускаем как есть
        setattr(mod, name, lambda s=None, f=f: f(s) if isinstance(s, (list, tuple, np.ndarray)) else f((s or 0) + off))


def build(arch: str, ds: str, fold: str) -> Path:
    """Каталог данных их формата: {train,val,test}_data[_joint].npy, *_label.pkl, *_meta.json, labels.json."""
    if ds in ("slovo", "bukva"):
        d = ROOT / "runs/M11" / M11DIR[arch][ds] / "data" / fold
        assert (d / "test_meta.json").is_file(), f"нет данных M11: {d}"
        return d
    sys.path.insert(0, str(HERE / ("m11_samslr" if arch == "slgcn" else "m11_stgcn")))
    import build_data  # их clip() M11 (SL-GCN: пиксели 512, 150 кадров; ST-GCN: доли − 0,5, 300 кадров)
    from mpk import data
    o = ROOT / "runs/TURNIR/their" / arch / "data" / ds / fold
    if (o / "test_meta.json").is_file():
        return o
    if ds == "uzsl":
        c = data.load(UZ / "cache/uzsl.npz")
        f = json.loads((UZ / "splits/uzsl_loso.json").read_text())["folds"][fold]
        parts = {"train": [(c, s) for s in f["train"] + f["val"]], "val": [(c, s) for s in f["val"]],
                 "test": [(c, s) for s in f["test"]]}
        name = lambda c, i: c["clips"][i]["gloss"]  # noqa: E731
    else:  # sb: предобучение Slovo + Bukva
        parts = {"train": [], "val": [], "test": []}
        for d_ in ("slovo", "bukva"):
            c = data.load(ROOT / f"cache/{d_}.npz")
            c["_ds"] = d_
            f = json.loads((ROOT / f"splits/{d_}_pretrain_all.json").read_text())["folds"]["all"]
            parts["train"] += [(c, s) for s in f["train"]]
            parts["val"] += [(c, s) for s in f["val"]]
        parts["test"] = parts["val"]
        name = lambda c, i: f"{c['_ds']}:{c['clips'][i]['gloss']}"  # noqa: E731
    pos = {}
    for r in parts.values():
        for c, s in r:
            if id(c) not in pos:
                pos[id(c)] = {cl["sample_id"]: i for i, cl in enumerate(c["clips"])}
    labels = sorted({name(c, pos[id(c)][s]) for r in parts.values() for c, s in r})
    lab = {g: k for k, g in enumerate(labels)}
    o.mkdir(parents=True, exist_ok=True)
    suf = "_joint" if arch == "slgcn" else ""
    for part, rows in parts.items():
        ids = [s for _, s in rows]
        x = np.stack([build_data.clip(c, pos[id(c)][s]) for c, s in rows]).transpose(0, 3, 1, 2)[..., None]
        np.save(o / f"{part}_data{suf}.npy", x.astype(np.float32))
        with open(o / f"{part}_label.pkl", "wb") as fh:
            pickle.dump((ids, [lab[name(c, pos[id(c)][s])] for c, s in rows]), fh)
        meta = {s: {"signer_id": c["clips"][pos[id(c)][s]]["signer_id"], "gloss": name(c, pos[id(c)][s])} for c, s in rows}
        (o / f"{part}_meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    (o / "labels.json").write_text(json.dumps(labels, ensure_ascii=False))
    return o


def write_out(out: Path, ds: str, score: dict, data: Path, extra: dict) -> None:
    labels = json.loads((data / "labels.json").read_text())
    meta = json.loads((data / "test_meta.json").read_text())
    ids = list(score)
    lg = np.stack([np.asarray(score[k], np.float32) for k in ids])
    e = np.exp(lg - lg.max(1, keepdims=True))
    p = (e / e.sum(1, keepdims=True)).astype(np.float16)
    np.savez(out / "probs.npz", ids=np.array(ids), probs=p, classes=np.array(labels))
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample_id", "signer_id", "gloss", "pred", "correct", "top5"])
        for k, s in zip(ids, lg):
            t5 = np.argsort(-s)[:5]
            w.writerow([ds, k, meta[k]["signer_id"], meta[k]["gloss"], labels[t5[0]], int(labels[t5[0]] == meta[k]["gloss"]),
                        "|".join(labels[q] for q in t5)])
    top1 = float(np.mean([labels[int(np.argmax(s))] == meta[k]["gloss"] for k, s in zip(ids, lg)]))
    (out / "summary.json").write_text(json.dumps({**extra, "top1": top1, "n_test": len(ids)}, indent=1, ensure_ascii=False))
    print("top1", round(top1, 4), flush=True)


def slgcn(ds, fold, data, out, init, last_epoch):
    sys.path.insert(0, str(HERE / "m11_samslr"))
    import run as R  # их main.py и подмены M11 (yaml.load, ReduceLROnPlateau, input, fork)
    import yaml
    sys.path[:0] = [str(HERE / "m11_samslr"), str(R.SLGCN)]
    import os
    os.chdir(out)
    n_class = len(json.loads((data / "labels.json").read_text()))
    name = f"sam_slr_joint_{fold}"
    tpl = (R.HERE / "train_joint_mpk.yaml").read_text()
    cfg = yaml.safe_load(tpl.replace("{fold}", fold).replace("{data}", str(data)))
    cfg["model_args"]["num_class"] = n_class
    if ds != "slovo":  # только железо, как M11 Г2-3 (Bukva): 8 воркеров; Slovo — их 32 (затравка k = 0 = прежний прогон)
        cfg["num_worker"] = 8
    if init:
        cfg |= {"weights": init, "ignore_weights": ["fc.weight", "fc.bias"]}
    (out / "save_models").mkdir(exist_ok=True)
    (out / "train.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    R.their_main(out / "train.yaml")
    done = R.epochs_done(out / "work_dir" / name)
    assert sorted(done) == list(range(250)), len(done)
    best = 249 if last_epoch else max(sorted(done), key=lambda e: done[e])
    t = yaml.safe_load(tpl.replace("{fold}", fold).replace("{data}", str(data)))
    t["model_args"]["num_class"] = n_class
    t["Experiment_name"] = name + "_test"
    t["test_feeder_args"] |= {"data_path": str(data / "test_data_joint.npy"), "label_path": str(data / "test_label.pkl"),
                              "debug": False}
    t |= {"phase": "test", "weights": str(out / f"save_models/{name}-{best}.pt"), "start_epoch": best, "save_score": True}
    (out / "test.yaml").write_text(yaml.safe_dump(t, allow_unicode=True, sort_keys=False))
    R.their_main(out / "test.yaml")
    (pk,) = (out / "work_dir" / (name + "_test") / "eval_results").glob(f"epoch_{best}_*.pkl")
    return pickle.loads(pk.read_bytes()), {"epoch": best, "val_top1": done[best], "weights": f"save_models/{name}-{best}.pt"}


def stgcn(ds, fold, data, out, init, last_epoch):
    sys.path.insert(0, str(HERE / "m11_stgcn"))
    import run as R  # REC4GPU (эмуляция 4 ГП), their(); stgcn_mpk — пути, граф mpk27
    import yaml
    labels = json.loads((data / "labels.json").read_text())
    wd = out / "work_dir"
    cfg = yaml.safe_load((R.HERE / "train_kinetics_mpk.yaml").read_text().replace("{work_dir}", str(wd))
                         .replace("{data}", str(data)).replace("{num_class}", str(len(labels))))
    cfg["num_epoch"] = EPOCHS["stgcn"]
    if init:
        cfg |= {"weights": init, "ignore_weights": ["fcn"]}
    R.their(cfg, out / "train.yaml")
    final = wd / f"epoch{EPOCHS['stgcn']}_model.pt"
    assert final.is_file(), final
    test = {"work_dir": str(out / "work_dir_test"), "feeder": cfg["feeder"], "model": cfg["model"],
            "model_args": cfg["model_args"], "phase": "test", "device": [0], "use_gpu": True, "test_batch_size": 64,
            "weights": str(final), "save_result": True,
            "test_feeder_args": {"data_path": str(data / "test_data.npy"), "label_path": str(data / "test_label.pkl")}}
    R.their(test, out / "test.yaml")
    v = R.val_log(wd)
    return pickle.loads((out / "work_dir_test/test_result.pkl").read_bytes()), \
        {"epoch": EPOCHS["stgcn"], "val_top1_last": v.get(EPOCHS["stgcn"] - 1), "weights": str(final.relative_to(out))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["one", "build"])
    ap.add_argument("--arch", choices=["slgcn", "stgcn"], required=True)
    ap.add_argument("--ds", choices=["slovo", "bukva", "uzsl", "sb"], required=True)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = build(a.arch, a.ds, a.fold)
    if a.cmd == "build":
        print(data)
        sys.exit(0)
    out = Path(a.out).resolve()
    if (out / "summary.json").is_file():
        print("готово", out)
        sys.exit(0)
    if out.exists():
        out.rename(out.parent / f"_oborvan_{out.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True)
    seed_offset(a.seed)
    t0 = time.time()
    fn = slgcn if a.arch == "slgcn" else stgcn
    score, extra = fn(a.ds, a.fold, data, out, a.init, last_epoch=a.ds == "uzsl")
    write_out(out, a.ds, score, data, {**extra, "arch": a.arch, "ds": a.ds, "fold": a.fold, "seed": a.seed, "init": a.init,
                                       "seed_rule": "их затравка + 1000·k", "wall_time_s": time.time() - t0})
    # маркер успеха стенда/раннера — metrics.json (как у mpk.train / m13_train)
    (out / "metrics.json").write_text((out / "summary.json").read_text())
