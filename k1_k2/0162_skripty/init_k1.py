# к1 0162: resolve_init и выбор пула из диффа d07e1e3 (к2 0184) — исполнены дословно, без torch.
# Запуск: python3 -I k1_k2/0162_skripty/init_k1.py k2_k1/0184_g2_init/d07e1e3.diff
import ast, json, sys, tempfile
from pathlib import Path
d = open(sys.argv[1], encoding="utf-8").read().split("\n")
i = d.index("+def resolve_init(init: str) -> str:")
src = "\n".join(l[1:] for l in d[i:i + 7])
ns = {"Path": Path, "json": json}; exec(compile(ast.parse(src), "turnir_their", "exec"), ns); R = ns["resolve_init"]
with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    for arch, rel in (("SL-GCN", "save_models/sam_slr_joint_all-7.pt"), ("ST-GCN", "work_dir/stgcn/epoch50_model.pt")):
        pre = t / arch / "all" / "seed2"; (pre / rel).parent.mkdir(parents=True); (pre / rel).write_bytes(b"x")
        (pre / "summary.json").write_text(json.dumps({"weights": rel}))
        assert R(str(pre)) == str((pre / rel).resolve()) and R(str(pre / rel)) == str((pre / rel).resolve())
        print(arch, "каталог →", Path(R(str(pre))).relative_to(pre.resolve()), "; файл → как есть")
    ab = t / "abs.pt"; ab.write_bytes(b"x"); q = t / "q"; q.mkdir(); (q / "summary.json").write_text(json.dumps({"weights": str(ab)}))
    assert R(str(q)) == str(ab.resolve()); print("абсолютный путь в summary.json → он же")
    for bad, why in ((t / "нет.pt", "файла нет"), (q, "summary.json указывает на пропавший файл")):
        if bad is q: ab.unlink()
        try: R(str(bad)); print("!!! не упало:", why); sys.exit(1)
        except AssertionError: print("assert:", why)
    e = t / "e"; e.mkdir()
    try: R(str(e)); print("!!! каталог без summary.json не упал"); sys.exit(1)
    except FileNotFoundError: print("каталог без summary.json → FileNotFoundError (громко, до сборки данных)")
# выбор пула: строка 304 дословно
for jobs_small in (0, 2):
    for small in (True, False):
        a = type("A", (), {"jobs_small": jobs_small})(); j = {"small": small}
        pool = "small" if (j["small"] and a.jobs_small) else "big"
        print(f"--jobs-small {jobs_small}, малое={small}: пул {pool}, размер малого пула {max(1, a.jobs_small)}")
        assert pool == ("small" if small and jobs_small else "big")
print("ok")
