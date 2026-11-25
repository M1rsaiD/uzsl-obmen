import sys, random, numpy as np, torch
sys.path.insert(0, "scripts"); import turnir_their as T
mode = sys.argv[1]
if mode == "fix":
    def seed_offset(k):
        off = 1000 * k
        for mod, name in ((torch, "manual_seed"), (torch.cuda, "manual_seed_all"), (np.random, "seed"), (random, "seed")):
            f = getattr(mod, name)
            setattr(mod, name, lambda s=None, f=f: f(s) if isinstance(s, (list, tuple, np.ndarray)) else f((s or 0) + off))
    T.seed_offset = seed_offset
T.seed_offset(int(sys.argv[2])); torch.manual_seed(1)
dl = torch.utils.data.DataLoader(list(range(8)), batch_size=2, num_workers=2, multiprocessing_context="fork")
print(mode, sys.argv[2], "ok", [b.tolist() for b in dl], torch.initial_seed())
