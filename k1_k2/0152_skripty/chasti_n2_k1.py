"""Части № 2, к1: точки токенов рычага № 2 (0163, sohn.part_points) против точек МПК (SPEC_MPK.md, разд. 2.1; диссертация 3.1.1: K = 85).
Запуск: python3 -I chasti_n2_k1.py <dissertatsiya>/06-experiments/SPEC_MPK.md k2_k1/0163_e2_levers_diff_ot_f6a5a72.diff k2_k3/0055_u01a_sintetika/code/src/mpk/sohn.py"""
import re, sys
spec, diff, sohn = (open(p, encoding="utf-8").read() for p in sys.argv[1:4])
row = lambda name: [int(x) for x in re.search(r"^\| " + name + r"[^|]*\| ([0-9, –-]+) \|", spec, re.M).group(1).replace(" ", "").split(",")]  # noqa: E731
mpk_pose, mpk_face = row("Поза"), row("Лицо")
ns = {}
for name in ("NOSE", "LIP", "REYE", "LEYE", "POSE_MP4"):
    exec(re.search(r"^" + name + r" = \[[^\]]*\]", sohn, re.M).group(0), ns)
assert "return [LH_LOC, RH_LOC, *body, [LOCAL[k] for k in LIP + NOSE + REYE + LEYE]]" in diff   # 0163: part_points, лицо
assert 'body = [list(range(NUM_NODES, NUM_NODES_POSE))] if "u01a" in tokens else []' in diff    # тело = точки U01a (118–121)
face = set(ns["LIP"] + ns["NOSE"] + ns["REYE"] + ns["LEYE"])
print(f"МПК: поза {len(mpk_pose)} {mpk_pose}, кисти 2 × 21, лицо {len(mpk_face)}  → всего {len(mpk_pose) + 42 + len(mpk_face)}")
print(f"№ 2: тело {len(ns['POSE_MP4'])} {ns['POSE_MP4']} (только с U01a), кисти 2 × 21, лицо {len(face)} (губы 40 + нос 4 + глаза 16 + 16)")
print(f"тело: общих {len(set(mpk_pose) & set(ns['POSE_MP4']))}, только у МПК {sorted(set(mpk_pose) - set(ns['POSE_MP4']))}")
print(f"лицо: общих {len(set(mpk_face) & face)}, только у МПК {len(set(mpk_face) - face)} {sorted(set(mpk_face) - face)}, только у № 2 {len(face - set(mpk_face))}")
assert (len(mpk_pose), len(mpk_face), len(face), len(set(mpk_face) & face)) == (6, 37, 76, 20)
