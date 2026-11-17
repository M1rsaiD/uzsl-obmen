"""ЦП, только чтение: (1) классы в обучении каждого фолда П (рычаг 4); (2) доля записей валидации Э2, лежащих в тесте турнира."""
import csv, json, collections, sys
U = sys.argv[1]  # /home/error_404/uzsign
for ds, ann in (("slovo", "datasets/slovo/slovo/annotations.csv"), ("bukva", "datasets/bukva/bukva/annotations.csv")):
    lab = {r["attachment_id"]: r["text"] for r in csv.DictReader(open(f"{U}/{ann}", encoding="utf-8"), delimiter="\t")}
    F = json.load(open(f"{U}/osnovnaya_model/splits/{ds}_p5.json"))["folds"]
    alltest = set().union(*[set(f["test"]) for f in F.values()])
    for k, f in F.items():
        tr = collections.Counter(lab[i] for i in f["train"])
        big = {c: n for c, n in tr.items() if c == "no_event"}
        v = sorted(n for c, n in tr.items() if c != "no_event")
        val = set(f["val"])
        print(f"{ds} {k}: обучение {len(f['train'])}, классов {len(tr)}, на класс min/мед/max {v[0]}/{v[len(v)//2]}/{v[-1]}, no_event {big}; "
              f"валидация {len(val)}, из них в тесте турнира {len(val & alltest)}")
