# счёт записей на класс (проверка довода «длинного хвоста нет» у рычага 4)
import csv, sys, collections
for path in sys.argv[1:]:
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096); f.seek(0)
        rd = csv.DictReader(f, dialect=csv.Sniffer().sniff(sample, delimiters=",\t;"))
        col = next(c for c in rd.fieldnames if c in ("text", "label", "gloss", "class"))
        cnt = collections.Counter(r[col] for r in rd)
    v = sorted(cnt.values())
    print(path, "колонка", col, "классов", len(cnt), "записей", sum(v), "min", v[0], "медиана", v[len(v)//2], "max", v[-1],
          "топ-2", cnt.most_common(2)[1][1] if len(cnt) > 1 else None)
