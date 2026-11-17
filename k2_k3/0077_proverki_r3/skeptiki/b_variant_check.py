"""Скептик ред. 3: тест (Б) test_e2_long_tail_cos_head_dropout_after_norm — на sohn.py d7b0d68 и на прежнем b766571 (до (Б));
потолок p при s = 16 и s варианта (А) = ln(1 + 9C) / ρ при ρ = 0,447 и √0,2. Только чтение (git show), ЦП.
Аргументы: <worktree>"""
import math
import subprocess
import sys
import types

WT = sys.argv[1]
sys.path[:0] = [WT + "/src"]
import torch  # noqa: E402


def mod_at(commit):
    src = subprocess.run(["git", "-C", WT, "show", f"{commit}:src/mpk/sohn.py"], capture_output=True, text=True, check=True).stdout
    m = types.ModuleType(f"sohn_{commit}")
    m.__file__ = WT + "/src/mpk/sohn.py"
    exec(compile(src, f"sohn_{commit}.py", "exec"), m.__dict__)
    return m


for c in ("b766571", "d7b0d68"):
    sohn = mod_at(c)
    torch.manual_seed(0)
    m = sohn.SohnNet(5, cos_head=True)
    f = torch.nn.functional.silu(torch.randn(1, 384) * 1.5)
    with torch.no_grad():
        m.classifier.weight[0] = f[0]
    m.features = lambda x, mk: f.expand(4000, -1)
    m.late_dropout_on = True
    with torch.no_grad():
        ev = m.eval()(None, None)[0]
        tr = m.train()(None, None).mean(0)
    print(f"{c}: логит оценки {float(ev[0]):.4f}, средний логит обучения {float(tr[0]):.4f}, отношение {float(tr[0] / ev[0]):.4f}")
for rho_name, rho in (("0,447", 0.447), ("√0,2", math.sqrt(0.2))):
    for name, C in (("Slovo", 1001), ("S+B", 1034), ("Bukva", 33)):
        s16 = 16 * rho
        p_zero = math.exp(s16) / (math.exp(s16) + (C - 1))                              # прочие косинусы 0
        p_simp = math.exp(s16) / (math.exp(s16) + (C - 1) * math.exp(-s16 / (C - 1)))  # прочие — симплекс
        print(f"ρ={rho_name} {name:5s} C={C:5d}: ln(1+9C)={math.log(1 + 9 * C):.4f}, s(А)={math.log(1 + 9 * C) / rho:.3f}, "
              f"потолок p при s=16: {p_zero:.4f} (прочие 0) / {p_simp:.4f} (симплекс)")
