"""к3 0048: свои проверки исполнением (ЦП, torch 2.5.1+cpu) решений исполнителя 1, 3 и «не ошибки» bf16 (0163 разд. 3–4).
sohn.py — блоб 7a87d90 (ветка d7b0d68), восстановлен в 0046. Логика загрузки --init — дословно из m13_train.py 8ac5f0f, строки 751–757."""
import torch
from mpk import sohn

TOK = ("u01a", "u04", "u01b")
ch = sohn.input_channels(TOK)

# Реш. 3: stem создаётся и снимается → прочие слои при той же затравке = сеть без № 2
def net(pt, seed=48, **kw):
    torch.manual_seed(seed)
    return sohn.SohnNet(1001, 192, channels=ch, tokens=TOK, part_tokens=pt, **kw)
a, b = net(False), net(True)
sa, sb = a.state_dict(), b.state_dict()
shared = [k for k in sa if k in sb]
diff = [k for k in shared if not torch.equal(sa[k], sb[k])]
only_a = sorted({k.split(".")[0] for k in sa if k not in sb}); only_b = sorted({k.split(".")[0] for k in sb if k not in sa})
assert not diff and only_a == ["stem"] and only_b == ["tok"], (diff[:3], only_a, only_b)
print(f"реш. 3: при затравке 48 все {len(shared)} общих тензоров сети с № 2 = сети без № 2 бит в бит; различаются только stem / tok")
# контроль: без «создать и снять» (stem не создаётся) прочие слои разошлись бы
torch.manual_seed(48); _ = torch.nn.Linear(10, 10)   # лишний расход ГСЧ до сети
c = sohn.SohnNet(1001, 192, channels=ch, tokens=TOK)
n_mis = sum(not torch.equal(sa[k], c.state_dict()[k]) for k in sa)
print(f"   контроль: лишний расход ГСЧ до сети меняет {n_mis} из {len(sa)} тензоров — тест чувствителен")

# Реш. 1: --init с --balanced → s = 16 из конструктора; без --balanced (идея 7) — s из весов; classifier.* — никогда
pre = net(False, cos_head=True); 
with torch.no_grad(): pre.cos_scale.fill_(23.5)
sd = pre.state_dict()
for balanced, want in ((True, 16.0), (False, 23.5)):
    model = net(False, seed=7, cos_head=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 682)   # корпус: другие классы
    model.classifier.bias.requires_grad_(False)
    w0 = model.classifier.weight.clone()
    keep = {k: v for k, v in sd.items() if not k.startswith("classifier.") and not (balanced and k == "cos_scale")}
    miss = model.load_state_dict(keep, strict=False)
    new = [k for k in miss.missing_keys if k.startswith(("stem_extra.", "gcn.", "part_mlp.", "cross.", "pool_score.", "cos_scale"))
           or ".dw_ms." in k]
    assert set(miss.missing_keys) - set(new) == {"classifier.weight", "classifier.bias"} and not miss.unexpected_keys, miss
    assert float(model.cos_scale) == want and torch.equal(model.classifier.weight, w0)
    assert torch.equal(model.stem.weight, pre.stem.weight)                    # энкодер загружен
    print(f"реш. 1: --balanced={balanced}: s = {float(model.cos_scale)} (ожид. {want}); classifier — новый; энкодер — из S+B")

# «Не ошибка» bf16 (0163 разд. 4): под autocast логиты обеих голов — bf16, смещения между плечами по типу нет
x = torch.randn(2, 40, ch); m = torch.ones(2, 40, dtype=torch.bool)
for cos in (False, True):
    n_ = net(False, cos_head=cos).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = n_(x, m)
    print(f"bf16: cos_head={cos}: dtype логитов под autocast = {y.dtype}")
    assert y.dtype == torch.bfloat16
print("ВСЁ СОВПАЛО")
