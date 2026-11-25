# Проверка: затравка k турнира доходит до потоков воркеров DataLoader (путь как у ST-GCN: run.py:75 seed(0), без worker_init_fn)
import sys, random, multiprocessing as mp, numpy as np, torch
sys.path.insert(0, "scripts"); import turnir_their as T
class D(torch.utils.data.Dataset):
    def __len__(self): return 4
    def __getitem__(self, i): return torch.tensor([np.random.rand(), random.random(), torch.rand(1).item()])
if __name__ == "__main__":
    k = int(sys.argv[1]); T.seed_offset(k)
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    dl = torch.utils.data.DataLoader(D(), batch_size=1, num_workers=2)
    out = [round(x, 6) for b in dl for x in b[0].tolist()]
    print(f"k={k} start_method={mp.get_start_method()} initial_seed={torch.initial_seed()} worker_np/random/torch={out[:3]}")
