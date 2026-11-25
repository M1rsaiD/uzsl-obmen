# к3: seed_offset из turnir_their.py (ast, дословно), настоящий torch; потоки воркеров при k
import ast, sys, random, numpy as np, torch, multiprocessing as mp
src = open(sys.argv[1], encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "seed_offset")
if sys.argv[2] == "old":  # строка до f6a8e52 (дифф 0095, «-»)
    old = ast.parse("setattr(mod, name, lambda s=None, f=f: f((s or 0) + off))").body[0]
    loop = next(n for n in fn.body if isinstance(n, ast.For)); loop.body[-1] = old
g = {"np": np, "random": random}; exec(compile(ast.Module([fn], []), "seed_offset", "exec"), g)
class D(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return torch.tensor([np.random.rand(), random.random(), torch.rand(1).item()])
if __name__ == "__main__":
    torch.multiprocessing.set_start_method(sys.argv[3], force=True)
    k = int(sys.argv[4]); g["seed_offset"](k)
    random.seed(0), np.random.seed(0), torch.manual_seed(0)  # как m11_stgcn/run.py:75 (по 0095)
    try:
        out = [round(x, 6) for b in torch.utils.data.DataLoader(D(), batch_size=1, num_workers=2) for x in b[0].tolist()]
        print(f"{sys.argv[2]} {sys.argv[3]} k={k} initial_seed={torch.initial_seed()} worker_np/random/torch={out[:3]}")
    except Exception as e:
        print(f"{sys.argv[2]} {sys.argv[3]} k={k} ОШИБКА {type(e).__name__}: {str(e).splitlines()[0][:90]}")
