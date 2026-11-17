"""Проверка находок F1/F5 (correctness) и счёт параметров SohnNet (ЦП). Код — рабочее дерево wt_e2 (PYTHONPATH=src)."""
import json, math, sys
import numpy as np, torch, torch.nn.functional as F
from mpk import sohn

torch.manual_seed(0)
# --- классы Slovo / Bukva: как m13_train (names = глоссы записей train+val+test фолда; фолд — первый в split)
ROOT = "/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/wt_e2"
C = {}
for s in ("slovo", "bukva"):
    with np.load(f"{ROOT}/cache/M13/{s}.npz", allow_pickle=False) as z:
        clips = json.loads(str(z["clips"]))
    g = {c["sample_id"]: c["gloss"] for c in clips}
    folds = json.loads(open(f"{ROOT}/splits/{s}_p5.json").read())["folds"]
    ns = {len({g[i] for k in ("train", "val", "test") for i in f[k]}) for f in folds.values()}
    C[s] = ns.pop(); assert not ns
print("классы", C)

# --- F1 (correctness): LateDropout(0,8) до нормировки косинусной головы
m = sohn.SohnNet(C["slovo"], cos_head=True).eval()
x = torch.randn(64, 40, sohn.CHANNELS); msk = torch.ones(64, 40, dtype=torch.bool)
with torch.no_grad():
    f = m.features(x, msk)
    r = torch.stack([F.cosine_similarity(F.dropout(f, 0.8, True), f, dim=-1) for _ in range(20)])
print(f"cos(dropout(f), f): среднее {r.mean():.3f}, мин {r.min():.3f}, макс {r.max():.3f}; sqrt(0,2) = {math.sqrt(0.2):.3f}")
rho = float(r.mean())
for nm, c in (("Slovo", C["slovo"]), ("Bukva", C["bukva"]), ("S+B", C["slovo"] + C["bukva"])):
    p = math.exp(16 * rho) / (math.exp(16 * rho) + c - 1)
    print(f"  {nm} C={c}: потолок p(верный) при обучении, s=16: {p:.3f}; s для p=0,9 при ρ: {math.log(9 * (c - 1)) / rho:.1f}")

# --- F5 (correctness): dtype логитов под autocast bf16 (ЦП; политика matmul та же, что CUDA)
for cos in (False, True):
    net = sohn.SohnNet(C["slovo"], cos_head=cos).eval()
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        print("autocast bf16, cos_head", cos, "→ логиты", net(x[:2], msk[:2]).dtype)

# --- параметры: без рычагов / № 2 / № 4; вход базовый и U04 + U01b
def cnt(n, tok, **fl):
    net = sohn.SohnNet(n, channels=sohn.input_channels(tok), tokens=tok, **fl)
    return sum(p.numel() for p in net.parameters()), sum(p.numel() for p in net.parameters() if p.requires_grad)
rows = {}
for s in ("slovo", "bukva"):
    for tok in ((), ("u04", "u01b")):
        b, bt = cnt(C[s], tok)
        p2, p2t = cnt(C[s], tok, part_tokens=True)
        p4, p4t = cnt(C[s], tok, cos_head=True)
        rows[f"{s} {'+'.join(tok) or 'база'}"] = dict(ch=sohn.input_channels(tok), base=b, n2=p2, d2=p2 - b, pct2=round(100 * (p2 - b) / b, 2),
                                                      n4_all=p4, n4_train=p4t)
        assert b == bt and p2 == p2t
for k, v in rows.items():
    print(k, v)
json.dump({"classes": C, "rows": rows}, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
