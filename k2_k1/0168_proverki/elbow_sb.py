# Зеркальность позы и кистей Slovo / Bukva (досылка к реш. 2, к1 0148): та же elbow_check, что у корпуса (uzsohn_pose.py ветки
# turnir-e2-levers 49b93bd), на готовых кэшах cache/UZSOHN/{slovo,bukva}_pose.npz и cache/M13/{slovo,bukva}.npz; печать — только счёты
import sys, json
import numpy as np
sys.path[:0] = ["scripts", "src"]
import uzsohn_pose, m13_build
for name in ("bukva", "slovo"):
    S = m13_build.load(name)
    with np.load(f"cache/UZSOHN/{name}_pose.npz", allow_pickle=False) as z:
        assert np.array_equal(z["offsets"], S["offsets"]), name
        pose = z["xy"]
    print(name, json.dumps(uzsohn_pose.elbow_check(pose, S["xy"], S["offsets"]), ensure_ascii=False), flush=True)
    del S, pose
