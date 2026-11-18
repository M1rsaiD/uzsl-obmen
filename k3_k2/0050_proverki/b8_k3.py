"""к3 0050, Б8 (сверка к1 0151 разд. 2 по 4.7). Строки берутся из файлов ДОСЛОВНО (поиск по тексту, не по номерам) и исполняются:
имя вложенного состава (стенд 964ecba, строка 475), разбор состава в части (master f6a5a72 — 578–579; стенд — 599–600), цена (548–549).
Запуск: python3 -I b8_k3.py <стенд 964ecba>/scripts/turnir_eval.py <обмен>/k2_k1/0168_fajly/turnir_eval_master_f6a5a72_strk42-43_maskirovany.py"""
import sys
S, M = (open(p, encoding="utf-8").read().splitlines() for p in sys.argv[1:3])
def line(src, *keys):
    hit = [(i + 1, s.strip()) for i, s in enumerate(src) if all(k in s for k in keys)]
    assert len(hit) == 1, (keys, hit)
    return hit[0]
n_name, NAME = line(S, 'name = "+".join(', "len(set(chosen))")
n_p1s, P1S = line(S, 'parts = [p for p in comp.split("+")')
n_p1m, P1M = line(M, 'parts = [p for p in comp.split("+")')
P2S, P2M = line(S, 'parts = [p.split("×")[0] for p in parts]')[1], line(M, 'parts = [p.split("×")[0] for p in parts]')[1]
G, SM = line(S, "g = [cost.get(ds, {}).get(p, {}) for p in parts]")[1], line(S, "sm = lambda k:")[1]
print(f"стенд {n_name}: {NAME}\nстенд {n_p1s}: {P1S}\nmaster {n_p1m}: {P1M}\n")
cost = {"uzsl": {"A1": {"gflops": 1.5}, "SL-GCN": {"gflops": 2.5}, "Sohn": {"gflops": 0.25}, "UzSohn": {"gflops": 0.5}}}
def gfl(comp, p1, p2, b=False):
    ns = {"comp": comp, "cost": cost, "ds": "uzsl"}
    try:
        exec(p1, ns); exec(p2, ns)
    except ValueError as e:
        return f"ValueError ({e})"
    if b:
        ns["parts"] = [] if comp.endswith("*") else ns["parts"]   # правка (б) к1 0151
    exec(G, ns); exec(SM, ns)
    return ns["sm"]("gflops")
print(f"{'состав (строка 475)':22s} | {'master':45s} | {'стенд 964ecba':12s} | (б)")
for names, chosen in ((["A1", "SL-GCN"], [(2, 1)] * 3), (["A1", "SL-GCN"], [(2, 1), (2, 1), (1, 2)]), (["A1", "SL-GCN"], [(1, 2), (1, 2), (2, 1)]),
                      (["Sohn", "UzSohn"], [(2, 1), (2, 1), (1, 1)]), (["Sohn", "UzSohn"], [(1, 2), (1, 2), (3, 0)]), (["Sohn", "UzSohn"], [(3, 0), (3, 0), (1, 2)])):
    ns = {"names": names, "chosen": chosen, "top": max(set(chosen), key=chosen.count)}
    exec(NAME, ns)
    comp = ns["name"]
    print(f"{comp:22s} | {str(gfl(comp, P1M, P2M)):45s} | {str(gfl(comp, P1S, P2S)):12s} | {gfl(comp, P1S, P2S, b=True)}")
