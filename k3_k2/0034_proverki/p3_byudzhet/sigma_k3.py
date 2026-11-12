"""к3, к 12 п. 3 (σ1/σ2): действующее искажение аугментации относительно входа на тесте, своим кодом по формулам
`sohn.augment` (строки 158–171: масштаб, скос ±0,15 по одной оси, поворот ±30° — в долях кадра, с вер. 0,75).
Строки — векторы, как в коде: v' = v · S · R. Пиксели p = v · D, D = diag(1, H/W) (y · H/W).
U02-1: вход = p / ширина плеч (скаляр) → карта тест → обучение = D⁻¹ S R D.
База у облака, изотропного в пикселях (круг, ρ_пикс = 1): вход = v / СКО по осям → карта = D⁻¹ S R · diag(1/√diag(MᵀM)), M = D⁻¹ S R
(на тесте круг после деления на СКО по осям снова изотропен). Печать: p50 и p90 σ1/σ2 по всем примерам (25 % без affine = 1)."""
import numpy as np
g = np.random.default_rng(7)
N = 200000
def draws():
    sh = g.uniform(-0.15, 0.15, N); which = g.random(N) < 0.5
    sx = np.where(which, 0.0, sh); sy = np.where(which, sh, 0.0)
    r = g.uniform(-30, 30, N) / 180 * np.pi
    S = np.stack([np.stack([np.ones(N), sx], 1), np.stack([sy, np.ones(N)], 1)], 1)
    R = np.stack([np.stack([np.cos(r), np.sin(r)], 1), np.stack([-np.sin(r), np.cos(r)], 1)], 1)
    aff = g.random(N) < 0.75
    return S, R, aff
S, R, aff = draws()
def ratio(M):
    sv = np.linalg.svd(M, compute_uv=False); return sv[:, 0] / sv[:, 1]
for name, hw in (("16:9", 9 / 16), ("4:3", 3 / 4), ("1:1", 1.0), ("9:16", 16 / 9)):
    D = np.diag([1.0, hw]); Di = np.diag([1.0, 1 / hw])
    M = Di @ S @ R
    u = ratio(M @ D)
    cn = np.sqrt(np.einsum('nij,nij->nj', M, M))          # √diag(MᵀM): нормы столбцов
    b = ratio(M / cn[:, None, :])
    u[~aff] = 1; b[~aff] = 1
    print(f"W:H {name}: U02-1 σ1/σ2 p50 {np.median(u):.2f} p90 {np.quantile(u, .9):.2f} | база, круг: p50 {np.median(b):.2f} p90 {np.quantile(b, .9):.2f}")
