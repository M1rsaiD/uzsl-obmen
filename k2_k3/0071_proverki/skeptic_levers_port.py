"""Скептик определений № 2–4 и порта группы 3 (0159). Только ЦП, OM только чтение; runs/TURNIR/**, runs/M13/** не читаются.
Читает: OM splits/{slovo,bukva}_pretrain_all.json, splits/sizes.json, поле clips кэша cache/M13/{slovo,bukva}.npz (без точек),
config.yaml шаблонов runs/M7/{slovo,bukva}_a_ema (бюджет шагов МПК). Печатает только агрегаты.
Запуск: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 <python 3.14.7> -B skeptic_levers_port.py > skeptic_levers_port.out
  А — № 4 на предобучении S+B (поток Sohn sb — один на 1 034 класса): доля Bukva и no_event в потоке, естественный против p ∝ 1/n.
  Б — № 3: горизонт EMA в долях обучения, Sohn (пакет 512) против МПК (пакет 128); порог 3 000 в шагах и в примерах.
  В — № 3: EMA не меняет траекторию обучения (train_step с AWP, MaskedBN, Lookahead) — бит в бит с прогоном без EMA.
"""
import io, json, os, sys, zipfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "src"), str(OM.parent / "uzsign_e1/src")]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(4)


def clips(ds):
    z = zipfile.ZipFile(OM / f"cache/M13/{ds}.npz")
    return {c["sample_id"]: c["gloss"] for c in json.loads(str(np.load(io.BytesIO(z.read("clips.npy")))))}


print("=== А. № 4 на S+B (records('sb'): один поток, одна голова «slovo:…/bukva:…», m13_train.py:144–156)")
cnt = Counter()
for ds in ("slovo", "bukva"):
    g = clips(ds)
    tr = json.loads((OM / f"splits/{ds}_pretrain_all.json").read_text())["folds"]["all"]["train"]
    cnt.update(f"{ds}:{g[s]}" for s in tr)
n = sum(cnt.values())
C = len(cnt)
nb = sum(v for k, v in cnt.items() if k.startswith("bukva:"))
cb = sum(k.startswith("bukva:") for k in cnt)
ne = {k: v for k, v in cnt.items() if "no_event" in k}
print(f"записей обучения {n}, классов {C} (Bukva {cb}); Bukva в потоке: естественно {nb / n:.4f}, при p ∝ 1/n_класса {cb / C:.4f}")
for k, v in ne.items():
    print(f"  {k}: {v} записей; доля в потоке {v / n:.4f} → {1 / C:.5f}; одна запись тянется в {(1 / C / v) / (1 / n):.3f} раза чаще")
vb = [v for k, v in cnt.items() if k.startswith("bukva:")]
print(f"  запись Bukva (медиана {int(np.median(vb))} на класс): чаще в {(1 / C / np.median(vb)) / (1 / n):.3f} раза; "
      f"запись Slovo (медиана {int(np.median([v for k, v in cnt.items() if k.startswith('slovo:') and 'no_event' not in k]))})")

print("\n=== Б. № 3: горизонт EMA 1/(1 − decay) в долях обучения")
sz = json.loads((OM / "splits/sizes.json").read_text())


def mpk_steps(name):
    out = {}   # только бюджет из config.yaml шаблона (metrics.json не читается)
    for p in sorted((OM / f"runs/M7/{name}").glob("fold*/seed*/config.yaml")):
        out.setdefault(p.parts[-3], yaml.safe_load(p.read_text())["train"]["steps"])
    assert out, name
    return list(out.values())


for ds, tpl in (("slovo", "slovo_a_ema"), ("bukva", "bukva_a_ema")):
    ntr = [sz[f"{ds}_p5"][f"fold{k}"]["train"][0] for k in range(1, 6)]
    s_sohn = [300 * (x // 512) for x in ntr]
    s_mpk = mpk_steps(tpl)
    for lab, st, bs in (("Sohn", s_sohn, 512), ("МПК (шаблон M7)", s_mpk, 128)):
        d = [0.999 if s >= 3000 else 0.99 for s in st]
        h = [1 / (1 - x) for x in d]
        ex = [0.999 if s * bs >= 3000 * 128 else 0.99 for s in st]
        hx = [1 / (1 - x) for x in ex]
        print(f"{ds:5s} {lab:16s} шагов {min(st)}–{max(st)}: порог в шагах → decay {sorted(set(d))}, горизонт "
              f"{min(a / b for a, b in zip(h, st)):.3f}–{max(a / b for a, b in zip(h, st)):.3f} обучения; "
              f"порог в примерах (3 000 × 128) → {sorted(set(ex))}, горизонт {min(a / b for a, b in zip(hx, st)):.3f}–"
              f"{max(a / b for a, b in zip(hx, st)):.3f}")

print("\n=== В. № 3: траектория с EMA = без EMA (ЦП, 6 шагов, AWP с 3-го, MaskedBN, Lookahead)")
from mpk import sohn  # noqa: E402
from mpk.train import ema_avg_fn  # noqa: E402


def parts_of(step):
    g = torch.Generator().manual_seed(1000 + step)
    return [(torch.randn(4, 32, 708, generator=g), torch.ones(4, 32, dtype=torch.bool), torch.randint(0, 7, (4,), generator=g))
            for _ in range(2)]


def run(with_ema):
    torch.manual_seed(48)
    model = sohn.SohnNet(7)
    opt = sohn.RAdamLookahead(model.parameters())
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn(0.999, True), use_buffers=True) if with_ema else None
    for step in range(6):
        model.train()
        model.late_dropout_on = step >= 4
        sohn.train_step(model, opt, parts_of(step), 1e-3, 1e-4, awp_on=step >= 2)
        if ema is not None:
            ema.update_parameters(model)
            with torch.no_grad():                    # оценка EMA между шагами (как val каждую эпоху)
                ema.module.eval()
                ema.module(*parts_of(99)[0][:2])
    return model, ema


a, _ = run(False)
b, e = run(True)
same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
diff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(b.state_dict().values(), e.module.state_dict().values()))
print(f"модель с EMA = модель без EMA бит в бит: {same}; EMA ≠ модели: max|Δ| = {diff:.3g}")
assert same and diff > 0
print("\nвсе проверки пройдены")
