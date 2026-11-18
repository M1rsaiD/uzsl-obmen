"""к3 0052: поток ГСЧ torch после создания сети с № 2 и без него (затравка 48) — сверка довода к2 (ред. 3.3, 22.11: «поток расходится»)
против моего 0048 («в PartTokens нет dropout, Stream от модели не зависит»). sohn.py — блоб 7a87d90 (0046); ЦП, torch 2.5.1+cpu.
Запуск (не -I: нужен PYTHONPATH): PYTHONPATH=<src> python rng_k3.py"""
import torch
from mpk import sohn
TOK = ("u01a", "u04", "u01b"); ch = sohn.input_channels(TOK)
st = {}
for pt in (False, True):
    torch.manual_seed(48); sohn.SohnNet(1001, 192, channels=ch, tokens=TOK, part_tokens=pt); st[pt] = torch.get_rng_state()
    print(f"part_tokens={pt}: первые 3 числа torch.rand после создания сети —", [round(float(v), 6) for v in torch.rand(3)])
print("состояние ГСЧ torch после создания сети одинаково:", torch.equal(st[False], st[True]))
