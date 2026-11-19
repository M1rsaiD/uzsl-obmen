"""к2 0176 (к1 0156 разд. 3 п. 3): в splits/slovo_p5.json есть и `folds` (e2_outside), и `groups` (uzsohn_sel.e2, «без группы j»);
у каждого фолда k исполнители теста — ровно группа k − 1. Печать — только счётчики, без кодов исполнителей и записей.
Запуск: python -I p5_groups_k2.py <osnovnaya_model>/splits/slovo_p5.json <datasets>/slovo/slovo/annotations.csv"""
import csv, json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("ключи:", sorted(d), "| группы:", sorted(set(d["groups"].values())))
rows = list(csv.DictReader(open(sys.argv[2], encoding="utf-8"), delimiter="\t"))
who = {r["attachment_id"]: r["user_id"] for r in rows}
for k, f in sorted(d["folds"].items()):
    users = {who[a] for a in f["test"]}
    gs = sorted({d["groups"].get(u, "нет") for u in users}, key=str)
    print(f"{k}: тест {len(f['test'])} зап., исполнителей {len(users)}, их группы {gs}")
    assert gs == [int(k[4:]) - 1], k
print("ok: фолд = группа + 1 у всех 5")
