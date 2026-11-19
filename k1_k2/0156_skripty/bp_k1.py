# к1: balanced_p (m13_train 5f83fec) исполнена дословно на синтетике S+B: доля набора = как без № 4, классы внутри набора поровну
import ast, sys, numpy as np
src = open(sys.argv[1], encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "balanced_p")
ns = {"np": np}; exec(compile(ast.Module([fn], []), "m13_train", "exec"), ns)
rng = np.random.default_rng(0)
gl = [f"slovo:c{k}" for k in rng.integers(0, 7, 900)] + [f"bukva:b{k}" for k in rng.integers(0, 3, 300)]
names = sorted(set(gl)); lab = np.array([names.index(g) for g in gl]); train = list(range(len(gl)))
p = ns["balanced_p"](train, lab, gl, "sb")
for s in ("slovo", "bukva"):
    m = np.array([g.startswith(s) for g in gl])
    per = [p[(lab == names.index(c))].sum() for c in names if c.startswith(s)]
    print(s, "доля", round(p[m].sum(), 6), "ожид.", round(m.mean(), 6), "классы", np.round(per, 6))
    assert abs(p[m].sum() - m.mean()) < 1e-12 and np.allclose(per, per[0])
q = ns["balanced_p"](train, lab, gl, "uzsl")   # не sb — один набор: все классы поровну
assert np.allclose([q[lab == k].sum() for k in range(len(names))], 1 / len(names))
try:
    ns["balanced_p"](train, lab, [g.split(":")[1] for g in gl], "sb"); print("!!! без префикса не упало")
except AssertionError:
    print("без префикса у sb — assert (как задумано)")
print("ok")
