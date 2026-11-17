"""к3: исполнение кусков m13_train.py ветки turnir-e2-levers (блоб 8ac5f0f = 5d9aebb из 0165 + хунк 0163) на ЦП.
1) sohn_ema: decay 0,99 при бюджете < 3 000 шагов, 0,999 при ≥ 3 000; прогрев M9.7; буферы усредняются, int — копия;
2) Stream с balanced_p: частоты классов внутри набора поровну, позиции k·bs + j, шагов ровно steps.
m13_build заглушён (нужен только для импорта); mpk.train — из локальной копии mpe-slovo-bukva 7563b5e (ema_avg_fn, ema_decay_at)."""
import sys, types
sys.modules["m13_build"] = types.ModuleType("m13_build")
import numpy as np, torch
import m13_train as M
from mpk.train import ema_decay_at

torch.manual_seed(0)
for total, d in ((1500, 0.99), (2999, 0.99), (3000, 0.999), (10_620, 0.999)):
    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    e = M.sohn_ema(net, total)
    ref_w, ref_b = None, None
    for t in range(60):
        with torch.no_grad():
            net[0].weight.add_(torch.randn_like(net[0].weight)); net[1].running_mean.add_(torch.randn(3))
            net[1].num_batches_tracked += 1
        e.update_parameters(net)
        w, b = net[0].weight.detach().clone(), net[1].running_mean.clone()
        a = 1 - ema_decay_at(d, t)
        ref_w = w if ref_w is None else torch.lerp(ref_w, w, a)
        ref_b = b if ref_b is None else torch.lerp(ref_b, b, a)
    assert torch.allclose(e.module[0].weight, ref_w) and torch.allclose(e.module[1].running_mean, ref_b), total
    assert int(e.module[1].num_batches_tracked) == int(net[1].num_batches_tracked) == 60       # int-буфер — копия
print("1) sohn_ema: decay по бюджету (1 500 / 2 999 → 0,99; 3 000 / 10 620 → 0,999), прогрев M9.7, буферы BN усредняются, int — копия")

rng = np.random.default_rng(3)
lab, gl = [], []
for c in range(50):
    n = 300 if c == 0 else int(rng.integers(5, 21)); lab += [c] * n; gl += [f"slovo:{c}"] * n
for c in range(10):
    n = int(rng.integers(60, 95)); lab += [50 + c] * n; gl += [f"bukva:{c}"] * n
lab = np.array(lab); train = list(range(len(lab)))
p = M.balanced_p(train, lab, gl, "sb")
st = M.Stream(train, 4000, 48, 64, 32768, p)
seen, pos = [], []
for k, b in enumerate(st):
    assert len(b) == 64
    seen += [i for i, _ in b]; pos += [q for _, q in b]
assert k + 1 == 4000 and pos == list(range(4000 * 64))
seen = np.array(seen); L = lab[seen]; G = np.array([gl[i].split(":")[0] for i in seen])
share = np.mean(G == "slovo"); exp = np.mean(np.array([g.split(":")[0] for g in gl]) == "slovo")
cs = np.bincount(L[G == "slovo"], minlength=50) / np.sum(G == "slovo"); cb = np.bincount(L[G == "bukva"] - 50, minlength=10) / np.sum(G == "bukva")
assert abs(share - exp) < 0.005 and np.abs(cs - 1 / 50).max() < 0.004 and np.abs(cb - 1 / 10).max() < 0.006
print(f"2) Stream + balanced_p: 4 000 шагов × 64, позиции 0…{len(pos) - 1}; доля Slovo {share:.4f} (ожид. {exp:.4f}); "
      f"частоты классов: Slovo 0,020 ± {np.abs(cs - 0.02).max():.4f} (класс с 300 записями — {cs[0]:.4f}), Bukva 0,100 ± {np.abs(cb - 0.1).max():.4f}")
print("ВСЁ СОВПАЛО")
