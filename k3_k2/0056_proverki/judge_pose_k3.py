"""к3 0056 (обещано в 0055 разд. 4): judge (uzsohn_sel.py 49b93bd, блоб d47d0c6) и pose_of (uzsohn_pose.py, b7532dc) — функции
взяты из файлов к2 0176_fajly дословно (ast, без импорта модулей: они читают файлы ПК при импорте), исполнены на краевых случаях.
Запуск из корня обмена: python3 -I judge_pose_k3.py"""
import ast
import numpy as np

def take(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names) or
            (isinstance(n, ast.Assign) and any(getattr(x, "id", "") in names for t in n.targets for x in ast.walk(t)))]
    g = {"np": np}
    exec(compile(ast.Module(keep, []), path, "exec"), g)
    return g

S = take("k2_k1/0176_fajly/uzsohn_sel_49b93bd.py", {"DELTA", "THR", "MIN_BETTER", "judge", "full_budget", "EPOCHS"})
judge, THR = S["judge"], S["THR"]
print(f"THR = {THR!r}, MIN_BETTER = {S['MIN_BETTER']}")
a = {f"c{k}": 50.0 for k in range(10)}
cases = {"Δ = THR у всех 10 (ничья — принят)": {c: v + THR for c, v in a.items()},
         "Δ = 1,008 у всех (< THR — нет)": {c: v + 1.008 for c, v in a.items()},
         "Δ̄ ≥ THR, но лучше 5 из 10 (нет)": {c: v + (3 * THR if k < 5 else -0.0) for k, (c, v) in enumerate(a.items())},
         "Δ̄ ≥ THR, лучше 6 из 10 (принят)": {c: v + (2 * THR if k < 6 else -0.2) for k, (c, v) in enumerate(a.items())},
         "Δ = 0 в ячейке не «лучше» (9/10, принят)": {c: v + (2 * THR if k else 0.0) for k, (c, v) in enumerate(a.items())}}
for name, b in cases.items():
    d, diff, better, ok = judge(a, b)
    print(f"  {name}: Δ̄ = {diff:.6f}, лучше {better}/10 → {ok}")
assert [judge(a, b)[3] for b in cases.values()] == [True, False, False, True, True] and judge(a, list(cases.values())[4])[2] == 9
fb = S["full_budget"]
m = dict(steps_run=100, steps=100, limit_train=0, epochs=300)
assert fb(m) and not fb({**m, "stop_epoch": 17}) and not fb({**m, "epochs": 299}) and not fb({**m, "steps_run": 99})
print("full_budget: полный — True; --stop-epoch, --epochs 299, steps_run ≠ steps — False")

P = take("k2_k1/0176_fajly/uzsohn_pose_49b93bd.py", {"NP", "pose_of"})
x85 = np.random.default_rng(0).uniform(0.1, 0.9, (5, 85, 2)).astype(np.float32)
pres = np.ones((5, 4), bool); pres[1, 0] = False            # кадр 1: позы нет
x85[2, 3] = 0.0                                             # кадр 2: точка 3 (MediaPipe 14) = (0, 0)
x85[3, 2, 0] = 0.0                                          # кадр 3: точка 2, только x = 0 — точка есть
p = P["pose_of"](x85, pres)
print("pose_of: NaN по (кадр, точка):", [[int(np.isnan(p[t, k]).all()) for k in range(4)] for t in range(5)])
assert np.isnan(p[1]).all() and np.isnan(p[2, 3]).all() and np.isfinite(p[3, 2]).all() and np.isfinite(p[[0, 4]]).all()
print("ВСЁ СОВПАЛО")
