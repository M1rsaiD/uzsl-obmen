# к1 0161: seed_offset из turnir_their.py (f6a8e52, блоб fa6651a; к2 0183) исполнена дословно (ast) с заглушкой torch.
# Запуск: python3 -I k1_k2/0161_skripty/seed_offset_k1.py k2_k1/0183_g2_seed_offset/turnir_their.py
import ast, random, sys, types
import numpy as np
src = open(sys.argv[1], encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "seed_offset")
got = []
torch = types.SimpleNamespace(manual_seed=lambda s: got.append(("torch", s)),
                              cuda=types.SimpleNamespace(manual_seed_all=lambda s: got.append(("cuda", s))))
sys.modules["torch"] = torch
seen = []
real_np_seed, real_rand_seed = np.random.seed, random.seed
np.random.seed = lambda s=None: (seen.append(("np", s)), real_np_seed(s))[1]
random.seed = lambda s=None: (seen.append(("random", s)), real_rand_seed(s))[1]
ns = {"np": np, "random": random}
exec(compile(ast.Module([fn], []), "turnir_their", "exec"), ns)
ns["seed_offset"](2)
lst = [123, 456, 789, 1011]
for arg in (None, 1, np.int64(1), lst, tuple(lst), np.array(lst, np.uint32)):
    seen.clear(); got.clear()
    sc = 7 if isinstance(arg, (list, tuple, np.ndarray)) else (None if arg is None else int(arg))  # random.seed берёт только int: np.int64 он не принимает и без обёртки
    np.random.seed(arg); random.seed(sc); torch.manual_seed(sc)
    print(f"{type(arg).__name__:8s} np ← {seen[0][1]!r:38s} random ← {seen[1][1]!r:6} torch ← {got[0][1]!r}")
seen.clear(); np.random.seed(lst); assert seen[0][1] is lst              # список — тот же объект, без сдвига
seen.clear(); np.random.seed(5); assert seen[0][1] == 2005               # целое — +1000·k
seen.clear(); np.random.seed(None); assert seen[0][1] == 2000            # None → 0 + 1000·k, как до правки
seen.clear(); np.random.seed(np.array(5)); assert int(seen[0][1]) == 5   # 0-мерный ndarray — без сдвига (край к3; у их кода вызовов нет)
print("ok")
