"""к3 0051 (сверка к1 0152 разд. 1 по 4.7): точки частей МПК — из КОДА МПК (data.py, model.py), не из SPEC; против констант sohn.py
и part_points диффа 0163. Запуск: python3 -I chasti_k3.py <mpe-slovo-bukva>/src/mpk <sohn.py 7a87d90> <обмен>/k2_k1/0163_e2_levers_diff_ot_f6a5a72.diff"""
import ast, re, sys
mpk, sohn, diff = sys.argv[1:4]
d, m, s = (open(p, encoding="utf-8").read() for p in (f"{mpk}/data.py", f"{mpk}/model.py", sohn))
def lst(src, n):  # список/кортеж после «n =», может идти на несколько строк
    i = re.search(rf"^{n}\s*=\s*[\[(]", src, re.M).end() - 1
    return ast.literal_eval(src[i:src.index({"[": "]", "(": ")"}[src[i]], i) + 1])
P, F = lst(d, "POSE_MP"), lst(d, "FACE_MP")
pp = re.search(r"^PART_POINTS = (.+?)  #", m, re.M).group(1)
print("model.py PART_POINTS:", pp, "→ длины", [len(x) for x in eval(pp)], "(порядок токенов: кисть Л, кисть П, тело, лицо)")
print("data.py: POSE_MP", P, "| FACE_MP", len(F), "точек")
c = {n: lst(s, n) for n in ("LIP", "NOSE", "REYE", "LEYE", "POSE_MP4")}
face = c["LIP"] + c["NOSE"] + c["REYE"] + c["LEYE"]
print({n: len(v) for n, v in c.items()}, "| лицо № 2:", len(face), "разных", len(set(face)))
print("лицо: общих с МПК", len(set(F) & set(face)), "| у № 2 нет", len(set(F) - set(face)), sorted(set(F) - set(face)))
print("тело: у № 2 нет", sorted(set(P) - set(c["POSE_MP4"])))
ret = next(l for l in open(diff, encoding="utf-8") if l.startswith("+    return [LH_LOC"))
print("0163 part_points:", ret[1:].strip())
