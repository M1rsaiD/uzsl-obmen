# к1 0157: функции uzsohn_sel.py и uzsohn_pose.py на 49b93bd (блобы d47d0c6, b7532dc; к2 0176_fajly) исполнены дословно (ast),
# без данных ПК. Запуск: python3 -I judge_elbow_k1.py <uzsohn_sel_49b93bd.py> <uzsohn_pose_49b93bd.py>
import ast, sys, types
import numpy as np

def take(path, names, ns):
    tree = ast.parse(open(path, encoding="utf-8").read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), names
    exec(compile(ast.Module(body, []), path, "exec"), ns)

S = {"np": np, "THR": max(1.0, 1.0081957667939179), "MIN_BETTER": 6, "EPOCHS": 300}
take(sys.argv[1], ["judge", "full_budget", "e2_cells", "e2_row"], S)
judge = S["judge"]
cells = [f"{s}/fold{k}" for s in ("slovo", "bukva") for k in range(1, 6)]
a = {c: 50.0 for c in cells}
def run(name, b, want):
    d, diff, better, ok = judge(a, b)
    print(f"{name:42s} Δ̄ = {diff:.10f}  лучше {better}/10  принят: {ok}")
    assert ok is want, name
run("+1,00 у всех (< THR)", {c: 51.0 for c in cells}, False)
run("+1,01 у всех", {c: 51.01 for c in cells}, True)
run("+1,4: лучше у 6 (+3) / хуже у 4 (−1)", {c: 50 + (3.0 if i < 6 else -1.0) for i, c in enumerate(cells)}, True)
run("+2,0: лучше у 5 (+5) / хуже у 5 (−1)", {c: 50 + (5.0 if i < 5 else -1.0) for i, c in enumerate(cells)}, False)
run("+1,6: лучше у 6, у 4 ничья (Δ = 0)", {c: 50 + (16 / 6 if i < 6 else 0.0) for i, c in enumerate(cells)}, True)
run("+1,6: лучше у 5, ничья у 5", {c: 50 + (3.2 if i < 5 else 0.0) for i, c in enumerate(cells)}, False)
# ровно на пороге: Δ_k = THR в каждой ячейке — принят ли (разность float)?
d, diff, better, ok = judge(a, {c: 50 + S["THR"] for c in cells})
print(f"Δ_k = THR у всех: Δ̄ − THR = {diff - S['THR']:.3e}, принят: {ok}  (на счёте из целых записей ровно THR не бывает)")
# micro против macro (e2_cells): исполнитель с 90 записями и с 10
R = {"slovo/fold1": ({}, [(f"r{i}", "A", int(i < 45)) for i in range(90)] + [(f"q{i}", "B", 1) for i in range(10)])}
mi, ma = S["e2_cells"](R)["slovo/fold1"], S["e2_cells"](R, macro=True)["slovo/fold1"]
print(f"micro {mi:.2f} % (55/100), macro {ma:.2f} % ((50 + 100)/2)"); assert (round(mi, 6), round(ma, 6)) == (55.0, 75.0)
assert S["e2_cells"](R, drop={"A"})["slovo/fold1"] == 100.0 and S["e2_cells"](R, drop={"A", "B"}) == {}
base = dict(steps_run=10620, steps=10620, limit_train=0, epochs=300, stop_epoch=None, max_steps=None)
assert S["full_budget"](base) and not any(S["full_budget"]({**base, **x}) for x in
       ({"steps_run": 10619}, {"limit_train": 64}, {"epochs": 17}, {"stop_epoch": 17}, {"max_steps": 5}))
print("full_budget: полный — да; урезанный любым из 5 способов — нет")

# uzsohn_pose: pose_of и elbow_check
sohn = types.SimpleNamespace(LH_LOC=[0] + list(range(1, 21)), RH_LOC=[21] + list(range(22, 42)))
sys.modules["mpk"] = types.SimpleNamespace(sohn=sohn); sys.modules["mpk.sohn"] = sohn
P = {"np": np, "NP": 4}
take(sys.argv[2], ["pose_of", "elbow_check"], P)
x85 = np.zeros((3, 85, 2), np.float32); x85[:, :4] = 0.5; x85[1, 2] = 0.0
pres = np.array([[True], [True], [False]] * 1).repeat(85, 1)
po = P["pose_of"](x85, pres)
print("pose_of: кадр без позы →", np.isnan(po[2]).all(), "; точка (0, 0) →", np.isnan(po[1, 2]).all(), "; прочие целы:", np.isfinite(po[0]).all())
assert np.isnan(po[2]).all() and np.isnan(po[1, 2]).all() and np.isfinite(po[0]).all() and np.isfinite(po[1, [0, 1, 3]]).all()
rng = np.random.default_rng(0)
def rec(n, mirror=False, hole=False):
    sx = np.full((n, 42, 2), np.nan, np.float32); pose = np.full((n, 4, 2), np.nan, np.float32)
    wl, wr = np.array([0.6, 0.6]), np.array([0.4, 0.6])          # Л-запястье справа на кадре (зеркало камеры)
    sx[:, 0], sx[:, 21] = wl + rng.normal(0, .01, (n, 2)), wr + rng.normal(0, .01, (n, 2))
    el, er = np.array([0.62, 0.5]), np.array([0.38, 0.5])
    pose[:, 2], pose[:, 3] = (er, el) if mirror else (el, er)
    if hole: pose[: n // 2, 2:] = np.nan                         # NaN-локти: кадры вне фильтра
    return pose, sx
def check(recs):
    pose = np.concatenate([p for p, _ in recs]); sx = np.concatenate([s for _, s in recs])
    off = np.concatenate([[0], np.cumsum([len(p) for p, _ in recs])])
    return P["elbow_check"](pose, sx, off)
print("elbow_check, поза и кисти отражены одинаково:", check([rec(30), rec(20, hole=True), rec(25)]))
try:
    check([rec(30, mirror=True), rec(20, mirror=True)]); print("!!! зеркальная поза не поймана"); sys.exit(1)
except AssertionError as e:
    print("поза отражена иначе, чем кисти → отказ:", str(e)[:70])
print("ok")
