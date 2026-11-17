import json, sys
r = json.load(open(sys.argv[1]))
for k, v in r.items():
    if not isinstance(v, dict) or "alpha_k" not in v or not k.endswith("|б"):
        continue
    a, e = v["alpha_k"], v["E_false"]
    print(f"{k[:58]:58s} " + " ".join(f"a{j} {a[str(j)]:.3f}" for j in (5, 6, 7, 8, 13)) + " | " + " ".join(f"E{j} {e[str(j)]:.2f}" for j in (5, 7, 8, 13)))
