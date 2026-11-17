"""Задача B (к2, 16.11): входы модели шума Э2 — разброс между затравками у одной модели на протоколе П (slovo_p5, bukva_p5).
Читает только готовые metrics.json/log.csv старых серий (runs/M1…M9, M9_7, E1F) и предобучение Sohn S+B (runs/UzSohn/sb/all).
НЕ читает runs/TURNIR/**, runs/M13/**, runs/UzSohn/{slovo,bukva,…}. Пишет только сюда: noise_records.json (без ID исполнителей).
Только ЦП."""
import io, json, os, sys, zipfile, csv
from collections import Counter
from pathlib import Path
import numpy as np

OM = Path("/home/error_404/uzsign/osnovnaya_model")
OUT = Path(__file__).with_name("noise_records.json")
SERIES = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M9_7", "E1F"]
FORBID = ("TURNIR", "M13")


def clips(name):
    z = zipfile.ZipFile(OM / f"cache/{name}.npz")
    return json.loads(str(np.load(io.BytesIO(z.read("clips.npy")), allow_pickle=False)))


def composition(ds):
    sig = {c["sample_id"]: c["signer_id"] for c in clips(ds)}
    sp = json.load(open(OM / f"splits/{ds}_p5.json"))["folds"]
    out = {}
    for f, parts in sp.items():
        out[f] = {p: Counter(sig[s] for s in parts[p]) for p in ("val", "test")}
    return out


def last_val(log):
    try:
        rows = list(csv.DictReader(open(log)))
        v = [float(r["val_top1"]) for r in rows if r.get("val_top1") not in (None, "")]
        return v[-1] if v else None
    except (OSError, KeyError, ValueError):
        return None


def main():
    comp = {ds: composition(ds) for ds in ("slovo", "bukva")}
    sid = {}   # signer hash -> анонимный номер (в вывод — только номера)
    recs = []
    for ser in SERIES:
        for m in sorted((OM / "runs" / ser).rglob("metrics.json")):
            assert not any(x in m.parts for x in FORBID)
            try:
                d = json.load(open(m))
            except (OSError, ValueError):
                continue
            cfg = d.get("config", {})
            split = cfg.get("data", {}).get("split", "")
            ds = {"splits/slovo_p5.json": "slovo", "splits/bukva_p5.json": "bukva"}.get(split)
            if ds is None or "macro_top1_by_signer" not in d:
                continue
            fold = cfg["data"]["fold"]
            ps = d["macro_top1_by_signer"].get("per_signer", {})
            tc = comp[ds][fold]["test"]
            if set(ps) != set(tc):          # тест прогона не совпал с разбиением — пропуск
                continue
            per = {sid.setdefault(k, len(sid)): [v, tc[k]] for k, v in ps.items()}
            rel = m.relative_to(OM / "runs").parts
            recs.append({"series": ser, "cond": "/".join(rel[:-3]), "fold": fold, "seed": d.get("seed", cfg.get("seed")),
                         "ds": ds, "val_best": d.get("best_val_top1"), "val_last": last_val(m.with_name("log.csv")),
                         "test_micro": d.get("top1"), "test_macro": d["macro_top1_by_signer"]["mean"],
                         "select": cfg.get("train", {}).get("select"), "ema": cfg.get("train", {}).get("ema"),
                         "arch": cfg.get("model", {}).get("arch"), "per_signer": per})
    # состав валидации/теста по фолдам (номера исполнителей — те же анонимные)
    compo = {ds: {f: {p: {str(sid.setdefault(k, len(sid))): n for k, n in c.items()} for p, c in parts.items()}
                  for f, parts in comp[ds].items()} for ds in comp}
    # Sohn, предобучение S+B (одна валидация, 4 затравки) — только поля валидации
    sohn = []
    for s in (42, 43, 44, 48):
        p = OM / f"runs/UzSohn/sb/all/seed{s}"
        md = json.load(open(p / "metrics.json"))
        rows = list(csv.DictReader(open(p / "log.csv")))
        be = int(md["best_epoch"])
        sohn.append({"seed": s, "n_val": md["n_val"], "best_epoch": be,
                     "val_top1_best": float(rows[be - 1]["val_top1"]), "val_top1_last": float(rows[-1]["val_top1"]),
                     "val_loss_best": float(md["best_val_loss"])})
    json.dump({"records": recs, "composition": compo, "sohn_sb": sohn}, open(OUT, "w"), ensure_ascii=False)
    print(len(recs), "прогонов протокола П;", Counter((r["series"], r["ds"]) for r in recs))
    print("Sohn S+B:", sohn)


if __name__ == "__main__":
    main()
