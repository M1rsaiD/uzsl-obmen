"""Скептик ред. 3: состояние ГСЧ torch после создания SohnNet с № 2 и без (22.11 «расход ГСЧ тот же»), ЦП. Аргумент: <worktree>"""
import sys
sys.path[:0] = [sys.argv[1] + "/src"]
import torch  # noqa: E402
from mpk import sohn  # noqa: E402
tok = ("u04", "u01b")
st = {}
for pt in (False, True):
    torch.manual_seed(48)
    sohn.SohnNet(1001, channels=sohn.input_channels(tok), tokens=tok, part_tokens=pt)
    st[pt] = torch.get_rng_state()
    print(f"part_tokens={pt}: первый torch.rand после создания сети = {float(torch.rand(1)):.6f}")
print("состояние ГСЧ после создания сети одинаково:", torch.equal(st[False], st[True]))
