# аудит к1 0145 N4: свой генератор (seed 2026), d = 384, keep 0,2; (в) dropout до нормировки, (Б) после; плюс «сосредоточенный» признак
import numpy as np
r = np.random.default_rng(2026); d, n, k = 384, 4000, 0.2
cos = lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
nz = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
for kind in ("gauss", "silu", "sparse8"):
    f = r.standard_normal((n, d))
    if kind == "silu": f = f / (1 + np.exp(-f))
    if kind == "sparse8": f[:, 8:] *= 0.05          # энергия в 8 координатах — оговорка к1 «не сосредоточен»
    w = f + 0.5 * r.standard_normal((n, d)); m = (r.random((n, d)) < k) / k
    rho_v = cos(f * m, w).mean() / cos(f, w).mean()                       # (в): F.normalize(dropout(f))
    rho_b = ((nz(f) * m) * nz(w)).sum(1).mean() / cos(f, w).mean()        # (Б): dropout(F.normalize(f)), голова s·f_d·ŵ
    print(f"{kind:8s} (в) ρ̂ = {rho_v:.4f}  (Б) отношение = {rho_b:.4f}  √k = {k ** .5:.4f}")
for C in (1001, 33):
    e = np.exp(16 * k ** .5); print(f"C = {C}: потолок p (в) = {e / (e + C - 1):.3f}; ln(1+9C) = {np.log(1 + 9 * C):.2f}")
