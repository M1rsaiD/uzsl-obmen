# к3: resolve_init (ast, дословно из turnir_their.py eb7f1db) + маршрут пулов из диффа d07e1e3 (turnir_train.py:301–304)
import ast, json, os, sys, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
src = open(sys.argv[1], encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "resolve_init")
g = {"Path": Path, "json": json}; exec(compile(ast.Module([fn], []), "resolve_init", "exec"), g); R = g["resolve_init"]
t = Path(tempfile.mkdtemp()); os.chdir(t)
for arch, rel in (("SL-GCN", "save_models/sam_slr_joint_all-7.pt"), ("ST-GCN", "work_dir/epoch50_model.pt")):
    d = Path("runs/pretrain") / arch / "seed0"; (d / rel).parent.mkdir(parents=True); (d / rel).write_bytes(b"x")
    (d / "summary.json").write_text(json.dumps({"weights": rel}))
    r = R(str(d)); os.chdir("/")  # их код потом делает chdir(out): путь должен остаться годным
    print(arch, "каталог (относительный) →", Path(r).relative_to(t), "абсолютный:", Path(r).is_absolute(), "файл есть после chdir:", Path(r).is_file())
    os.chdir(t); print(arch, "файл как есть:", R(str(d / rel)) == str((t / d / rel).resolve()))
    (d / rel).unlink()
    try: R(str(d)); print(arch, "пропавший файл: НЕТ assert")
    except AssertionError as e: print(arch, "пропавший файл → assert:", str(e)[:40])
try: R(str(t / "net")); print("несуществующий путь: НЕТ assert")
except AssertionError as e: print("несуществующий путь → assert:", str(e)[:40])
for jobs, small_n in ((1, 0), (1, 2), (2, 3)):  # строки диффа, как есть
    jobs_list = [{"small": s} for s in (True, False, True)]
    with ThreadPoolExecutor(jobs) as big, ThreadPoolExecutor(max(1, small_n)) as small:
        route = ["small" if (small if j["small"] and small_n else big) is small else "big" for j in jobs_list]
    print(f"jobs={jobs} jobs_small={small_n}: малое, большое, малое →", route)
