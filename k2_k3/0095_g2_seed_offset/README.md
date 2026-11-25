# 0183 (к3: 0095) — к 423/424: исправление seed_offset `f6a8e52`

Файлы: `f6a8e52.diff` (git show), `turnir_their.py` (целиком, master `f6a8e52`, блоб `fa6651a`; до — `e7cc6bf`), скрипты проверки `repro_fork.py`, `worker_streams.py`, `worker_streams_fork.py`, вывод — `vyvod.txt` (ЦП, CUDA_VISIBLE_DEVICES=, Python 3.14.7, torch 2.14.1).

**Где ставится и откуда k.** `turnir_their.py:196` — `seed_offset(a.seed)` в главном процессе до вызова их кода; k = `--seed` задания. Сдвигаются `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, `random.seed`: каждый их вызов с числом получает +1000·k.

**к1 (1): где сдвигается torch.manual_seed.** Их вызовы: SL-GCN `main.py:36–39` (`init_seed`: всё = 1; он же `worker_init_fn`, стр. 229/236, и `init_seed(0)` стр. 587), ST-GCN — своих нет, ставит наш `m11_stgcn/run.py:75` `seed(0)` всем трём. Все идут через подмену → база воркеров `_base_seed` берётся из уже сдвинутого генератора главного процесса. `generator=` у их DataLoader нет (grep пуст). Вывод `worker_streams_fork.py`: k = 0 два раза — потоки воркеров (np / random / torch) совпадают; k = 2, 3 — все три другие; `initial_seed` 0 / 2000 / 3000.

**к3 (в): путь проверки.** На ПК метод запуска — **fork, задан явно**: `m11_stgcn/run.py:35`, `m11_samslr/run.py:43` (`set_start_method("fork", force=True)`); в forkserver подмена в воркер не попадает (старый код там не падает — в `vyvod.txt`), значит падение на ПК — путь fork. Проверка повторена на fork: старый код — TypeError при k = 0 и 2, новый — проходит.

**к3 (б): 0-мерный ndarray.** Их вызовы — только целые (`grep` в `vyvod.txt`), край пуст. `np.int64` идёт веткой `(s or 0) + off`.

**к1 (2): запись кода группы 2.** Её нет: `g3_code.txt` пишет только порт группы 3 (`turnir_sohn_shared.py:48`); в `runs/TURNIR/` других `*code*` нет. До `f6a8e52` ни одно задание группы 2 не завершилось (310/310 упали), так что все прогоны группы 2 — на одной версии кода.

**OOM.** План «повтор — малые по одному» (к1 — за). Проверку с воркерами в тесты — добавлю после прохода группы (`tests/`, ЦП), не трогая идущий юнит.
