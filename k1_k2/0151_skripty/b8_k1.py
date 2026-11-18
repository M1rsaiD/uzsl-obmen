"""Б8, к1: строки стенда turnir_eval.py (964ecba) исполняю дословно, по номерам — 475 (имя вложенного состава),
548–549 (цена), 599–600 (разбор состава в части); 513 — подпись acc_ci.
Запуск: python3 -I b8_k1.py <восстановленный scripts/turnir_eval.py> <k1_k2/0062_pareto/pareto_turnir.py>"""
import importlib.util, math, re, sys, textwrap
L = open(sys.argv[1], encoding="utf-8").read().splitlines()
ln = lambda a, b=None: textwrap.dedent("\n".join(L[a - 1:(b or a)]))  # noqa: E731

# 1) где ставится «*»: один раз, в конце всего имени (строка 475)
assert '+ ("*" if len(set(chosen)) > 1 else "")' in L[474], L[474]
name_expr = re.search(r"name = (.*)$", L[474]).group(1)
def nested_name(names, top, chosen):  # дословное выражение строки 475
    return eval(name_expr, {}, {"names": names, "top": top, "chosen": chosen})
# 2) цена (548–549) и разбор состава (599–600)
assert L[547].strip().startswith("g = [cost.get(ds") and L[548].strip().startswith("sm = lambda k")
assert L[598].strip().startswith("parts = [p for p in comp.split") and L[599].strip() == 'parts = [p.split("×")[0] for p in parts]'
cost = {"UzSL": {n: {"gflops": g, "latency_ms": 10 * g, "params_m": g / 10} for n, g in
                 (("A1", 1.0), ("A2", 0.5), ("MPK_large", 2.0), ("ST-GCN", 3.0), ("SL-GCN", 4.0), ("Sohn", 5.0), ("UzSohn", 6.0))}}
def price(comp):
    loc = {"comp": comp, "cost": cost, "ds": "UzSL"}  # одно пространство имён: lambda в 549 видит g
    exec(ln(599, 600), loc)
    exec(ln(548, 549), loc)
    return loc["parts"], loc["sm"]("gflops")
cases = [  # (names, top — самый частый состав, chosen — составы по исполнителям)
    (["A1", "SL-GCN"], (2, 1), [(2, 1), (2, 1)]),   # состав один у всех — «*» нет
    (["A1", "SL-GCN"], (2, 1), [(2, 1), (1, 2)]),   # различается, последний без ×k → «Имя*»
    (["A1", "SL-GCN"], (1, 2), [(1, 2), (2, 1)]),   # различается, последний с ×k → «Имя×k*»
    (["Sohn", "UzSohn"], (2, 1), [(2, 1), (3, 0)]), # толпа Sohn, «UzSohn*»
    (["Sohn", "UzSohn"], (1, 2), [(1, 2), (3, 0)]), # толпа Sohn, «UzSohn×2*»
]
res = []
for names, top, chosen in cases:
    comp = nested_name(names, top, chosen)
    parts, g = price(comp)
    res.append((comp, parts, g)); print(f"{comp:22s} части={parts} gflops={g}")
assert [r[2] for r in res] == [6.0, "нет данных", 9.0, "нет данных", 17.0]
# 3) числа строки (top1, ДИ) от частей не зависят: acc_ci (513) частей не получает
assert L[512].startswith("def acc_ci(c, signer, level, ds, macro=True, B=None)") and "parts" not in L[512]
assert "acc_ci(c, signer, level_main, ds, B=B_MAIN)" in L[546]
# 4) но цена решает, попадёт ли строка на рисунок Парето (0062): «нет данных» → NaN → строка отброшена
spec = importlib.util.spec_from_file_location("p", sys.argv[2]); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
src = open(sys.argv[2], encoding="utf-8").read()
assert "r[c] = float(r[c]) if r[c] not in ('', 'нет данных') else math.nan" in src and "and not math.isnan(r[xcol])]" in src
print("Б8: «Имя*» → нет цены → строки нет на панели UzSL рисунка Парето; «Имя×k*» → цена самого частого состава → строка есть")
