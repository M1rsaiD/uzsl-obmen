"""ЦП-проверка кода рычагов 0059 № 2–4 для Sohn поверх (а′) — OM HEAD (только чтение).

Запуск (ЦП, ≤ 4 потоков, без .pyc в OM):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 /home/error_404/.local/share/mise/installs/python/3.14.7/bin/python -B \
      levers_code.py | tee levers_code.out
А — как есть в OM: parts=True с базовым входом и с токенами; где отказывает m13_train; cos_head + AWP.
Б — предлагаемые правки, имитация подклассами здесь (OM не меняется): части по столбцам (а′), cos_head без смещения,
    выборка p ∝ 1/n_класса, EMA после полного шага. Только маленькие тензоры.
"""
import os
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
OM = "/home/error_404/uzsign/osnovnaya_model"
sys.path[:0] = [f"{OM}/src", f"{OM}/../uzsign_e1/src", f"{OM}/../uzsign_ai_v2/src", f"{OM}/scripts"]

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

torch.set_num_threads(4)
from mpk import sohn  # noqa: E402
from mpk.train import ema_avg_fn, ema_decay_at  # noqa: E402
import m13_train  # noqa: E402

git = lambda *a: subprocess.run(["git", "-C", OM, *a], capture_output=True, text=True).stdout.strip()
print("OM HEAD", git("rev-parse", "HEAD"), "| незакоммичено в sohn.py/m13_train.py/test_sohn.py:",
      repr(git("status", "--porcelain", "src/mpk/sohn.py", "scripts/m13_train.py", "tests/test_sohn.py")))
ASP = 0.5625
L = sohn.LOCAL
PTS = {"губы": [L[k] for k in sohn.LIP], "кисть Л": sohn.LH_LOC, "кисть П": sohn.RH_LOC,
       "нос+глаза": [L[k] for k in sohn.NOSE + sohn.REYE + sohn.LEYE], "поза": list(range(118, 122))}
NAMES = list(PTS)


def rec(T, seed):
    r = np.random.default_rng(seed)
    x = r.uniform(0.3, 0.7, (T, sohn.NUM_NODES_POSE, 2)).astype(np.float32)
    x[:, 118], x[:, 119] = [0.62, 0.8], [0.38, 0.8]
    x[:, 118:120] += r.normal(0, 0.01, (T, 2, 2)).astype(np.float32)   # плечи не постоянны: dx плеч ≠ 0
    return x


RECS = [rec(17, 1), rec(23, 2)]


def owners(tok):
    """Независимо от раскладки в коде: столбец c «принадлежит» точке p, если при NaN у p (все кадры) он становится нулём."""
    x = RECS[1]
    ref = sohn.preprocess_u(x, ASP, tok)
    nz = (ref != 0).any(0)
    own = [set() for _ in range(ref.shape[1])]
    for p in range(sohn.NUM_NODES_POSE):
        y = x.copy()
        y[:, p] = np.nan
        z = nz & (sohn.preprocess_u(y, ASP, tok) == 0).all(0)
        for c in np.flatnonzero(z):
            own[c].add(p)
    return own, nz


def audit(groups, tok, names):
    """groups — списки столбцов частей; → нарушения (столбец части, чья точка-владелец вне части), без владельца, нули."""
    own, nz = owners(tok)
    W = len(own)
    bad = {n: sum(1 for c in g if c < W and own[c] and not own[c] <= set(PTS[n])) for n, g in zip(names, groups)}
    used = set(int(c) for g in groups for c in g)
    return {"W": W, "нарушений": bad, "не в частях": W - len(used & set(range(W))),
            "без владельца (не нули)": [c for c in range(W) if nz[c] and not own[c]], "нули": [c for c in range(W) if not nz[c]]}


def grad_cols(model, W):
    """Столбцы входа, до которых доходит градиент выхода (случайный вход, маска — все шаги)."""
    torch.manual_seed(5)
    x = torch.randn(2, 16, W, requires_grad=True)
    model.eval()(x, torch.ones(2, 16, dtype=torch.bool)).sum().backward()
    return int((x.grad.abs().sum((0, 1)) > 0).sum())


def batch(tok):
    return sohn.collate([sohn.preprocess_u(r, ASP, tok) for r in RECS])


# ================================================================ А. КАК ЕСТЬ
print("\n=== А1. SohnNet(parts=True), базовый вход 708 ===")
torch.manual_seed(0)
base = sohn.SohnNet(7).eval()
torch.manual_seed(0)
mp = sohn.SohnNet(7, parts=True).eval()
r = mp.load_state_dict(base.state_dict(), strict=False)
assert not r.unexpected_keys and all(k.startswith("part_mlp.") for k in r.missing_keys)
x, m = batch(())
assert x.shape[2] == 708
with torch.no_grad():
    y0, y1 = base(x, m), mp(x, m)
assert y1.shape == (2, 7) and torch.allclose(y0, y1, atol=1e-5)
gb = [getattr(mp, f"part_base{i}").tolist() for i in range(mp.n_parts)]
print("частей", mp.n_parts, "| столбцов на часть", dict(zip(NAMES, map(len, gb))), "| форма выхода", tuple(y1.shape),
      "| в начале = Sohn: max|Δ| =", f"{(y0 - y1).abs().max():.1e}")
print("остаточная MLP части:", mp.part_mlp[0], "| выход MLP в начале — нули:",
      bool((mp.part_mlp[0][2].weight == 0).all() and (mp.part_mlp[0][2].bias == 0).all()))
print("аудит раскладки (базовый вход):", audit(gb, (), NAMES[:4]))

print("\n=== А2. SohnNet(parts=True) с токенами (а′): падает ли и что делает ===")
for tok in (("u01a",), ("u04",), ("u01b",), ("u02s", "u02c"), ("u01a", "u04", "u01b"), sohn.INPUT_TOKENS):
    W = sohn.input_channels(tok)
    x, m = batch(tok)
    assert x.shape[2] == W
    torch.manual_seed(0)
    bt = sohn.SohnNet(7, channels=W).eval()
    torch.manual_seed(0)
    pt = sohn.SohnNet(7, channels=W, parts=True).eval()
    pt.load_state_dict(bt.state_dict(), strict=False)
    cut = __import__("copy").deepcopy(bt)
    with torch.no_grad():
        cut.stem.weight[:, 708:] = 0
        yb, yp, yc = bt(x, m), pt(x, m), cut(x, m)
    assert yp.shape == (2, 7)
    gp = [getattr(pt, f"part_base{i}").tolist() for i in range(pt.n_parts)]
    a = audit(gp, tok, NAMES[:4])
    print(f"{','.join(tok):28s} W={W:4d}: не падает, выход {tuple(yp.shape)}; ≠ без частей (max|Δ| {(yb - yp).abs().max():.2e}),"
          f" = stem без столбцов ≥ 708 (max|Δ| {(yc - yp).abs().max():.1e}); градиент доходит до {grad_cols(pt, W)} из {W}"
          f" (без частей — {grad_cols(bt, W)}); вне частей {a['не в частях']}; не в своей части: {a['нарушений']}")

print("\n=== А3. m13_train: где отказывает (--plan-only, без обучения и без файлов) ===")
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m13_plan_tmp", "seed48")   # --out: замок не в OM, если assert не сработает
for argv in (["--input", "u01a,u04,u01b", "--variant", "1"], ["--input", "u01b", "--variant", "7"], ["--input", "u01b"]):
    full = ["--set", "slovo", "--fold", "fold1", "--seed", "48", *argv, "--plan-only", "--out", TMP]
    try:
        p = m13_train.main(full)
        print(" ".join(argv), "→ проходит; шагов", p["steps"], "=", p["steps_per_epoch"], "× 300; n_train", p["n_train"]) if p else None
    except AssertionError as e:
        print(" ".join(argv), f"→ AssertionError m13_train.py:{traceback.extract_tb(e.__traceback__)[-1].lineno}: {e}")

print("\n=== А4. cos_head (идея 7) + AWP в sohn.train_step ===")


def parts_of(W, n=7, seed=3):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(3, 20, W, generator=g), torch.rand(3, 20, generator=g) > 0.1, torch.randint(0, n, (3,), generator=g))
            for _ in range(2)]


for tok in ((), ("u01b",)):
    W = sohn.input_channels(tok)
    torch.manual_seed(0)
    mc = sohn.SohnNet(7, channels=W, cos_head=True).train()
    with torch.no_grad():
        yc = mc.eval()(*batch(tok))
    mc.train()
    print(f"вход {W}: форма {tuple(yc.shape)}, max|логит| {yc.abs().max():.2f} ≤ cos_scale {float(mc.cos_scale.detach()):.0f};",
          "classifier.bias в выходе не участвует")
    opt = sohn.RAdamLookahead(mc.parameters())
    sohn.train_step(mc, opt, parts_of(W), 1e-3, 1e-4, awp_on=False)
    print("  шаг без AWP — ок;", end=" ")
    try:
        sohn.train_step(mc, opt, parts_of(W), 1e-3, 1e-4, awp_on=True)
        print("шаг с AWP — ок")
    except RuntimeError as e:
        tb = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.endswith("sohn.py")][-1]
        print(f"шаг с AWP — RuntimeError sohn.py:{tb.lineno} ({tb.line.strip()}): {str(e).splitlines()[0][:140]}")
        named = [(n, p) for n, p in mc.named_parameters() if p.requires_grad]   # порядок params в train_step
        x, m = batch(tok)
        g = torch.autograd.grad(F.cross_entropy(mc(x, m).float(), torch.tensor([0, 1])), [p for _, p in named], allow_unused=True)
        print("  не участвует в графе:", [(i, n) for i, ((n, _), q) in enumerate(zip(named, g)) if q is None])

print("\n=== А5. выборка и EMA в цикле Sohn (m13_train.main) ===")
src = open(f"{OM}/scripts/m13_train.py", encoding="utf-8").read().splitlines()
hits = [i + 1 for i, s in enumerate(src) if any(w in s for w in ("WeightedRandomSampler", "balanced", "1 / cnt", "AveragedModel"))]
print("строки m13_train.py с AveragedModel/весами выборки:", hits, "→ EMA только в train_recipe_a1 (U33), выборки ∝ 1/n нет;",
      "Sohn-поток — Stream (перестановка + буфер 32768)")

# ================================================================ Б. ПРЕДЛАГАЕМЫЕ ПРАВКИ (имитация)
print("\n=== Б1. № 2: части по столбцам (а′) — предлагаемая правка ===")


def part_cols(tokens):
    """Столбцы preprocess_u(tokens) по частям: база 6·P (P = 122 при u01a), затем u04 [Л 225 | П 225], затем u01b [Л 42 | П 42]."""
    P = sohn.NUM_NODES_POSE if "u01a" in tokens else sohn.NUM_NODES
    cols = lambda pts: [k * 2 * P + 2 * p + a for k in range(3) for p in pts for a in (0, 1)]   # = sohn._cols при P = 118
    o, Lc, Rc = 6 * P, [], []
    if "u04" in tokens:
        Lc += range(o, o + 225)
        Rc += range(o + 225, o + 450)
        o += 450
    if "u01b" in tokens:
        Lc += range(o, o + 42)
        Rc += range(o + 42, o + 84)
        o += 84
    g = [cols(PTS["губы"]), cols(PTS["кисть Л"]) + Lc, cols(PTS["кисть П"]) + Rc, cols(PTS["нос+глаза"])]
    if "u01a" in tokens:
        g.append(cols(PTS["поза"]))
    assert o == sohn.input_channels(tokens) and sorted(sum(g, [])) == list(range(o))
    return g


class SohnNetPT(sohn.SohnNet):
    """Имитация правки в sohn.SohnNet: parts=True + tokens. Прежние модули — в прежнем порядке (part_mlp — после них, как в коде)."""

    def __init__(self, n, tokens=(), dim=192, **kw):
        super().__init__(n, dim, channels=sohn.input_channels(tokens), **kw)
        g = part_cols(tokens)
        for i, c in enumerate(g):
            self.register_buffer(f"part_base{i}", torch.tensor(c, dtype=torch.long), persistent=False)
            self.register_buffer(f"part_ext{i}", torch.tensor([], dtype=torch.long), persistent=False)
        self.n_parts, self.parts = len(g), True
        self.part_mlp = nn.ModuleList([nn.Sequential(sohn.dense(dim, dim, True), nn.SiLU(), sohn._zero(nn.Linear(dim, dim)))
                                       for _ in g])

    def stem_in(self, x):
        h = 0
        for i, mlp in enumerate(self.part_mlp):
            b = getattr(self, f"part_base{i}")
            hp = F.linear(x[..., b], self.stem.weight[:, b])     # правка: x вместо x[..., :708]
            h = h + hp + mlp(hp)
        return h


torch.manual_seed(0)
old = sohn.SohnNet(7, parts=True).eval()
torch.manual_seed(0)
new = SohnNetPT(7, ()).eval()
x, m = batch(())
with torch.no_grad():
    same = torch.equal(old(x, m), new(x, m))
assert same and all(torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values()))
assert all(torch.equal(a, b) for (_, a), (_, b) in zip(sorted(old.named_buffers()), sorted(new.named_buffers())))
print("без токенов: веса, буферы частей и выход = прежний parts=True бит в бит:", same)
n_sohn = sum(p.numel() for p in base.parameters())
combos = [tuple(t for t, on in zip(("u01a", "u04", "u01b"), bits) if on) for bits in np.ndindex(2, 2, 2)]
for tok in combos[1:] + [sohn.INPUT_TOKENS]:
    W = sohn.input_channels(tok)
    names = NAMES[:5] if "u01a" in tok else NAMES[:4]
    g = part_cols(tok)
    a = audit(g, tok, names)
    torch.manual_seed(0)
    cur = sohn.SohnNet(7, channels=W).eval()
    torch.manual_seed(0)
    pt = SohnNetPT(7, tok).eval()
    x, m = batch(tok)
    with torch.no_grad():
        d = (cur(x, m) - pt(x, m)).abs().max()
    gc = grad_cols(pt, W)
    dp = sum(p.numel() for p in pt.parameters()) - sum(p.numel() for p in cur.parameters())
    assert d < 1e-4 and gc == W and not any(a["нарушений"].values()) and not a["не в частях"] and not a["без владельца (не нули)"]
    z = [c - (W - 84) for c in a["нули"]] if "u01b" in tok else a["нули"]
    print(f"{','.join(tok):28s} W={W:4d} частей {len(g)}: {dict(zip(names, map(len, g)))}; не в своей части 0, вне частей 0;"
          f" в начале = текущая модель (max|Δ| {d:.1e}); градиент — все {gc}; +{dp} парам. (+{dp / n_sohn:.1%} к Sohn);"
          f" нули{' (от начала u01b)' if 'u01b' in tok else ''}: {z}")

print("\n=== Б2. № 4: cos_head поверх частей и токенов; правка — classifier.bias без градиента ===")
tok = ("u01a", "u04", "u01b")
W = sohn.input_channels(tok)
torch.manual_seed(0)
mc = SohnNetPT(7, tok, cos_head=True).train()
mc.classifier.bias.requires_grad_(False)          # правка: смещение не участвует в косинусной голове → не в params AWP
opt = sohn.RAdamLookahead(mc.parameters())
for s in range(3):
    sohn.train_step(mc, opt, parts_of(W), 1e-3, 1e-4, awp_on=s > 0)
with torch.no_grad():
    yc = mc.eval()(*batch(tok))
assert yc.shape == (2, 7) and yc.abs().max() <= float(mc.cos_scale.detach()) + 1e-4 and torch.all(mc.classifier.bias == 0)
print(f"3 шага (2 с AWP) — ок; cos_scale {float(mc.cos_scale.detach()):.4f} (старт 16, обучаемый); bias = 0; выход {tuple(yc.shape)}")

print("\n=== Б3. № 4: выборка p ∝ 1/n_класса (поток Stream, шаги не меняются) ===")


class StreamW(m13_train.Stream):
    """Имитация правки Stream: w — вероятности записей train (∝ 1/n_класса); w=None — прежний поток бит в бит."""

    def __init__(self, idx, steps, seed, bs, buf, w=None):
        super().__init__(idx, steps, seed, bs, buf)
        self.w = w

    def __iter__(self):
        if self.w is None:
            yield from super().__iter__()
            return
        rng = np.random.default_rng([self.seed, 7])
        for k in range(self.steps):      # i.i.d. с возвращением; позиция (затравка аугментации) — как в Stream: 0, 1, 2, …
            yield [(int(i), k * self.bs + j) for j, i in enumerate(rng.choice(self.idx, self.bs, p=self.w))]


rg = np.random.default_rng(0)
lab = np.concatenate([np.repeat(np.arange(1000), rg.integers(12, 21, 1000)), np.full(360, 1000)])   # как Slovo: 12–20 + no_event
train = list(range(len(lab)))
cnt = np.bincount(lab[train])
w = 1.0 / cnt[lab[train]]
w /= w.sum()
spe = len(train) // 512
s0, s1 = list(m13_train.Stream(train, 3, 48, 512, 32768)), list(StreamW(train, 3, 48, 512, 32768))
assert s0 == s1
nb = 200
draw = [b for b in StreamW(train, nb, 48, 512, 32768, w)]
pos = [p for b in draw for _, p in b]
idx = np.array([i for b in draw for i, _ in b])
f = np.bincount(lab[idx], minlength=1001) / len(idx)
assert pos == list(range(nb * 512)) and all(len(b) == 512 for b in draw)
print(f"w=None = Stream бит в бит: {s0 == s1}; n_train {len(train)} → шагов 300 × {spe} (как без правки: spe = n // 512);"
      f" позиции 0…{nb * 512 - 1} без повторов; доля класса: мин {f.min():.5f} / макс {f.max():.5f} (цель {1 / 1001:.5f});"
      f" no_event {f[1000]:.4f} (было {cnt[1000] / len(train):.4f})")

print("\n=== Б4. № 3: EMA после полного шага (AWP снят, Lookahead синхронизирован, commit_bn) ===")
tr = yaml.safe_load(open(f"{OM}/configs/common.yaml", encoding="utf-8"))["train"]
sz = json.load(open(f"{OM}/splits/sizes.json", encoding="utf-8"))
for ds in ("slovo_p5", "bukva_p5"):
    st = [300 * (v["train"][0] // 512) for v in sz[ds].values()]
    d = [tr["ema_decay"] if s >= tr["ema_short_below"] else tr["ema_decay_short"] for s in st]
    cap = next(t for t in range(10 ** 5) if ema_decay_at(d[0], t) >= d[0])
    print(f"{ds}: шагов {st} → decay {sorted(set(d))} (common.yaml: {tr['ema_decay']}/{tr['ema_decay_short']} при < "
          f"{tr['ema_short_below']}; ema_warmup по умолчанию {tr['ema_warmup']}); прогрев (1+t)/(10+t) до t = {cap}"
          f" ({cap / st[0]:.0%} шагов), затем окно ≈ {1 / (1 - d[0]):.0f} шагов")
tok = ("u01a", "u04", "u01b")
W = sohn.input_channels(tok)
torch.manual_seed(0)
model = SohnNetPT(7, tok).train()
opt = sohn.RAdamLookahead(model.parameters())
ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn(0.999, True), use_buffers=True)
fl = lambda mod: [t.detach().clone() for t in list(mod.parameters()) + [b for b in mod.buffers() if b.is_floating_point()]]
ok = []
for step in range(1, 6):
    prev = fl(ema.module)
    sohn.train_step(model, opt, parts_of(W, seed=step), 1e-3, 1e-4, awp_on=step > 1)   # AWP снят и commit_bn — внутри
    if step == 5:
        ok.append(("Lookahead: быстрые = медленные после шага 5", all(torch.equal(p, s) for p, s in zip(opt.params, opt.slow))))
    ema.update_parameters(model)                                                    # ← место обновления
    cur, e = fl(model), fl(ema.module)
    if step == 1:
        ok.append(("шаг 1: EMA = копия весов и буферов", all(torch.equal(a, b) for a, b in zip(e, cur))))
    else:
        dk = ema_decay_at(0.999, step - 1)
        ok.append((f"шаг {step}: d = {dk:.4f}", all(torch.allclose(a, p + (c - p) * (1 - dk), atol=1e-6) for a, p, c in zip(e, prev, cur))))
bn = model.stem_bn
ok.append(("running_var MaskedBN модели ≠ 1 (commit_bn прошёл) и EMA ≠ модели", bool((bn.running_var != 1).any()) and
           not torch.equal(ema.module.stem_bn.running_var, bn.running_var)))
ok.append(("целые буферы частей скопированы", all(torch.equal(getattr(ema.module, f"part_base{i}"), getattr(model, f"part_base{i}"))
                                                  for i in range(model.n_parts))))
ok.append(("ключи state_dict EMA = модели (best.pt грузится в SohnNet)", list(ema.module.state_dict()) == list(model.state_dict())))
with torch.no_grad():
    ye = ema.module.eval()(*batch(tok))
ok.append(("вывод EMA-модели в eval (бегущие средние EMA)", bool(torch.isfinite(ye).all()) and ye.shape == (2, 7)))
for k, v in ok:
    print(f"  {'ok ' if v else 'НЕТ'} {k}")
assert all(v for _, v in ok)
print("\nвсе проверки пройдены")
