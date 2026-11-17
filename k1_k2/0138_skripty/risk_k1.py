# к1 0138: α₁, α₅ и P(A₅ | не A₁…A₄) своим кодом — непрерывные модели M1–M3 поправки 2, 3.5 (целые записи не повторял)
import json, numpy as np
J = json.load(open('k2_k1/0154_risk_inputs_by_scode.json')); S = J['order']; B = J['by_signer']
sd = np.array([B[k]['sd_seeds_pp'] for k in S]); p = np.array([B[k]['top1_mean_pp'] for k in S]) / 100
nV = np.array([B[k]['n_as_Vk'] for k in S]); nT = np.array([B[k]['n_test'] for k in S])
mac = np.array(J['macro_by_seed_pp']); THR = max(1.0, 1.0081957)
print(f"СКО macro трёх затравок = {mac.std(ddof=1):.3f} (в поправке — 0,662)")
def run(sig, N=400_000, arms=5, seed=138):
    r = np.random.default_rng(seed); A = np.zeros((N, arms), bool)
    e0 = r.standard_normal((N, 10)) * sig
    for j in range(arms):
        d = r.standard_normal((N, 10)) * sig - e0
        A[:, j] = (d.mean(1) >= THR) & ((d > 0).sum(1) >= 6)
    a1 = A.mean(0).mean(); a5 = A.any(1).mean(); cond = A[~A[:, :4].any(1), 4].mean()
    return a1, A[:, :4].any(1).mean(), a5, cond
for name, sig in [("M1", sd), ("M2", sd * 0.512 / 0.662),
                  ("M3 (n на V_k)", np.sqrt(0.2199 * p * (1 - p) / nV * 1e4)),
                  ("M3 (n теста)", np.sqrt(0.2199 * p * (1 - p) / nT * 1e4))]:
    a1, a4, a5, c = run(sig)
    print(f"{name:<14} α₁ = {a1:.3f}  α₄ = {a4:.3f}  α₅ = {a5:.3f}  P(A₅ | не A₁…A₄) = {c:.3f}")
