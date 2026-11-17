"""к 22.7 (печать к тесту, находка скептика 0159): полный sha256 списков (А) и (Б) — правило сериализации как
val_composition.py:82: коды исполнителей теста Slovo вне валидаций, по возрастанию (str), через «\\n», UTF-8, без
завершающего «\\n». Только ЦП, только разбиения и поле clips кэша M13; кодов не печатает. runs/** не читает."""
import hashlib, io, json, zipfile
from pathlib import Path
import numpy as np

OM = Path("/home/error_404/uzsign/osnovnaya_model")


def clips(ds):
    z = zipfile.ZipFile(OM / f"cache/M13/{ds}.npz")
    return {c["sample_id"]: c["signer_id"] for c in json.loads(str(np.load(io.BytesIO(z.read("clips.npy")))))}


sig = {ds: clips(ds) for ds in ("slovo", "bukva")}
sp = {ds: json.loads((OM / f"splits/{ds}_p5.json").read_bytes()) for ds in sig}
V = {ds: {sig[ds][i] for f in sp[ds]["folds"].values() for i in f["val"]} for ds in sig}
T = {sig["slovo"][i] for f in sp["slovo"]["folds"].values() for i in f["test"]}
for name, ex in (("(А) вне валидации Slovo", V["slovo"]), ("(Б) вне валидаций Slovo и Bukva", V["slovo"] | V["bukva"])):
    cl = sorted(s for s in T if s not in ex)
    s = "\n".join(cl)
    assert not s.endswith("\n") and all(isinstance(x, str) for x in cl)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    print(f"{name}: {len(cl)} из {len(T)} исп.; sha256 {h}")
