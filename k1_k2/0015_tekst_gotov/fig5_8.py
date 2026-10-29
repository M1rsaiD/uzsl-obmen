"""Рисунки 5–8 статьи 1. Числа — из 06-experiments/m8_tables.md (после M9.1), m9_tables.md (M9.0, M9.7),
mpk_tables.md (M6) и m7_tables.md (§5). Запуск: python3 figures/fig5_8.py (из папки статьи)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CM = 1 / 2.54
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # слоты 1–4, проверены validate_palette.js
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------- Рис. 5: точность и вычисления (m8_tables: основная таблица, M8.3, M8.5; M9.0) ----------
# (имя, GFLOPs, top-1, low, high, семейство МПК?)
SLOVO = [("MPE-A1", 0.760, 70.8, 68.8, 72.7, True), ("MPE without graph", 0.196, 70.8, 68.9, 72.8, True),
         ("Larger MPE", 1.330, 72.9, 71.1, 74.9, True), ("ST-GCN", 1.979, 69.0, 67.0, 70.9, False),
         ("SL-GCN", 2.025, 68.1, 66.2, 69.9, False), ("Keypoint Transformer", 0.432, 66.3, 64.2, 68.4, False),
         ("Two-layer Transformer", 0.111, 56.4, 54.0, 58.6, False), ("BiLSTM", 0.624, 52.3, 49.9, 54.7, False),
         ("MLP", 0.004, 28.6, 26.7, 30.5, False)]
BUKVA = [("MPE-A1", 0.757, 93.2, 92.3, 94.0, True), ("MPE without graph", 0.193, 94.5, 93.7, 95.2, True),
         ("Larger MPE", 1.327, 93.4, 92.6, 94.3, True), ("ST-GCN", 1.978, 78.7, 77.5, 79.9, False),
         ("SL-GCN", 2.024, 84.9, 83.9, 86.0, False), ("Keypoint Transformer", 0.431, 85.1, 82.8, 87.3, False),
         ("Two-layer Transformer", 0.110, 75.4, 72.9, 78.0, False), ("BiLSTM", 0.621, 84.6, 83.3, 85.8, False),
         ("MLP", 0.001, 70.9, 69.2, 72.7, False)]
# положение подписей в координатах данных: (x, y, выравнивание)
POS = {"slovo": {"MPE-A1": (0.76, 73.6, "center"), "MPE without graph": (0.165, 70.8, "right"),
                 "Larger MPE": (1.5, 74.6, "left"), "ST-GCN": (2.35, 69.6, "left"), "SL-GCN": (2.35, 67.0, "left"),
                 "Keypoint Transformer": (0.36, 66.0, "right"), "Two-layer Transformer": (0.135, 56.0, "left"),
                 "BiLSTM": (0.75, 51.9, "left"), "MLP": (0.005, 28.2, "left")},
       "bukva": {"MPE-A1": (0.757, 91.3, "center"), "MPE without graph": (0.193, 96.3, "center"),
                 "Larger MPE": (1.5, 95.0, "left"), "ST-GCN": (2.35, 78.3, "left"), "SL-GCN": (2.35, 85.4, "left"),
                 "Keypoint Transformer": (0.36, 85.4, "right"), "Two-layer Transformer": (0.135, 75.0, "left"),
                 "BiLSTM": (0.621, 82.6, "center"), "MLP": (0.0013, 70.5, "left")}}


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.6 * CM), constrained_layout=True)
    for ax, data, name, key in ((axes[0], SLOVO, "(a) Slovo", "slovo"), (axes[1], BUKVA, "(b) Bukva", "bukva")):
        for lab, g, acc, lo, hi, mpe in data:
            c = BLUE if mpe else MUTED
            ax.errorbar(g, acc, yerr=[[acc - lo], [hi - acc]], fmt="o" if mpe else "s", ms=4.5, color=c,
                        mfc=c if lab == "MPE-A1" or not mpe else "white", mew=1, elinewidth=0.8, capsize=0, zorder=3)
            x, y, ha = POS[key][lab]
            ax.text(x, y, lab, fontsize=6.5, ha=ha, va="center", color=INK if mpe else MUTED)
        ax.set_xscale("log")
        ax.set_xlim(0.0008, 9)
        ax.set_xlabel("GFLOPs per window (log scale)")
        ax.set_title(name, loc="left")
        grid(ax, "both")
    axes[0].set_ylabel("Top-1 accuracy, %")
    axes[0].set_ylim(25, 77)
    axes[1].set_ylim(68, 97.5)
    fig.savefig(OUT / "fig5_cost.png")
    plt.close(fig)


# ---------- Рис. 6: абляции (m8_tables: M8.5, M8.2–M8.3, основная таблица A0) ----------
ABL = [  # (подпись, Slovo Δ, lo, hi, Bukva Δ, lo, hi)
    ("Without part tokens", -2.7, -3.3, -2.0, -5.7, -6.9, -4.6),
    ("Without graph (MPE-A2)", 0.0, -0.6, 0.7, 1.3, 0.9, 1.8),
    ("Without long-range edges", -0.9, -1.5, -0.3, 0.6, 0.2, 1.0),
    ("Without EMA (MPE-A0)", -1.5, -1.9, -1.0, -0.3, -0.7, 0.1),
    ("Larger model (D = 256, N = 6)", 2.1, 1.6, 2.7, 0.2, -0.1, 0.6),
]


def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 5.0 * CM), sharey=True, constrained_layout=True)
    ys = list(range(len(ABL)))[::-1]
    for ax, off, name in ((axes[0], 1, "(a) Slovo"), (axes[1], 4, "(b) Bukva")):
        for y, row in zip(ys, ABL):
            d, lo, hi = row[off:off + 3]
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=INK, lw=1)
            ax.plot(d, y, "o", ms=5, color=INK, mfc=INK if sig else "white", mew=1, zorder=3)
            ax.annotate(f"{d:+.1f}".replace("-", "\u2212"), (hi, y), xytext=(4, -2.5), textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.axvline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(name, loc="left")
        ax.set_xlabel("Difference to MPE-A1, percentage points")
        grid(ax, "x")
    axes[0].set_yticks(ys, [r[0] for r in ABL])
    axes[0].set_xlim(-4, 3.5)
    axes[1].set_xlim(-7.5, 2.8)
    fig.savefig(OUT / "fig6_ablations.png")
    plt.close(fig)


# ---------- Рис. 7: перенос (mpk_tables: M6; m9_tables: M9.7) ----------
LAD_X = ["1", "3", "10", "all"]
LAD = {"From scratch": ([10.9, 26.3, 63.7, 92.9], [(10.0, 11.9), (24.6, 28.0), (62.0, 65.4), (92.0, 93.7)], MUTED, "s"),
       "Slovo, fine-tuning": ([70.7, 81.8, 86.8, 90.9], [(69.1, 72.2), (80.5, 83.1), (85.6, 88.0), (89.9, 91.9)], BLUE, "o"),
       "Slovo, linear probe": ([70.2, 81.9, 86.1, 89.4], [(68.6, 71.7), (80.5, 83.3), (84.8, 87.3), (88.3, 90.6)], ORANGE, "^")}
PIL_X = ["1 per class", "2 per class", "all"]
PIL = {  # pooled, (участник 002, 003) — M9.7, MPE-A1, полное дообучение
    "From scratch": ([9.5, 11.7, 24.7], [(12.8, 6.8), (9.8, 13.3), (21.0, 27.7)], MUTED, "s"),
    "Slovo": ([31.7, 33.9, 39.4], [(35.2, 28.8), (38.7, 29.9), (42.0, 37.3)], ORANGE, "^"),
    "Bukva": ([35.6, 38.5, 46.3], [(40.6, 31.6), (41.5, 36.0), (50.3, 43.0)], AQUA, "D"),
    "Slovo + Bukva": ([40.3, 41.2, 47.5], [(48.5, 33.7), (46.9, 36.6), (49.9, 45.6)], BLUE, "o"),
}


def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.4 * CM), constrained_layout=True)
    ax = axes[0]
    for lab, (v, ci, c, m) in LAD.items():
        x = range(len(v))
        ax.errorbar(x, v, yerr=[[a - lo for a, (lo, _) in zip(v, ci)], [hi - a for a, (_, hi) in zip(v, ci)]],
                    color=c, marker=m, ms=4.5, lw=1.5, elinewidth=0.8, capsize=0, label=lab)
    ax.set_xticks(range(4), LAD_X)
    ax.set_xlabel("Training recordings per class (Bukva)")
    ax.set_ylabel("Top-1 accuracy, %")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Simulation: Slovo → Bukva (MPE-A0)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    grid(ax)
    ax = axes[1]
    for i, (lab, (v, parts, c, m)) in enumerate(PIL.items()):
        x = [j + (i - 1.5) * 0.08 for j in range(len(v))]
        for xj, (p2, p3) in zip(x, parts):
            ax.plot([xj, xj], [min(p2, p3), max(p2, p3)], color=c, lw=0.8, alpha=0.7)
        ax.plot(x, v, color=c, marker=m, ms=4.5, lw=1.5, label=("Pretrained on " + lab) if lab != "From scratch" else lab)
    ax.set_xticks(range(3), PIL_X)
    ax.set_xlabel("UzSL training data")
    ax.set_ylim(0, 68)
    ax.set_title("(b) UzSL pilot, fine-tuning (MPE-A1)", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, ncol=2)
    ax.text(0.99, 0.02, "vertical bars: participants 002 and 003", transform=ax.transAxes, ha="right",
            fontsize=6, color=MUTED)
    grid(ax)
    fig.savefig(OUT / "fig7_transfer.png")
    plt.close(fig)


# ---------- Рис. 8: валидация на тех же участниках против теста (m7_tables §5) ----------
VT = [  # (условие, val 002, val 003, test 002, test 003)
    ("all · Bukva · FT", 70.5, 100.0, 46.6, 43.6), ("all · Bukva · FT, no augmentation", 73.1, 100.0, 44.3, 40.9),
    ("all · Bukva · FT, synthetic signers", 67.9, 98.4, 45.2, 41.9), ("all · Bukva · probe", 62.8, 88.9, 44.3, 37.5),
    ("all · Slovo+Bukva · FT", 78.2, 100.0, 50.6, 37.3), ("all · Slovo+Bukva · probe", 73.1, 100.0, 48.7, 37.7),
    ("all · scratch", 79.5, 100.0, 13.8, 15.5), ("all · Slovo · FT", 87.2, 98.4, 44.3, 33.9),
    ("all · Slovo · probe", 76.9, 96.8, 44.1, 33.5), ("k1 · Bukva · FT", 74.4, 88.9, 43.1, 38.1),
    ("k1 · Bukva · probe", 69.2, 81.0, 42.7, 33.3), ("k1 · Slovo+Bukva · FT", 76.9, 93.7, 49.7, 35.2),
    ("k1 · Slovo+Bukva · probe", 78.2, 93.7, 48.5, 34.3), ("k1 · scratch", 57.7, 54.0, 9.8, 7.4),
    ("k1 · Slovo · FT", 80.8, 90.5, 38.9, 31.8), ("k1 · Slovo · probe", 75.6, 85.7, 39.4, 30.5),
    ("k2 · Bukva · FT", 74.4, 95.2, 43.8, 36.7), ("k2 · Bukva · probe", 70.5, 88.9, 42.0, 33.1),
    ("k2 · Slovo+Bukva · FT", 85.9, 98.4, 46.6, 37.9), ("k2 · Slovo+Bukva · probe", 82.1, 100.0, 45.7, 35.6),
    ("k2 · scratch", 74.4, 79.4, 6.8, 10.8), ("k2 · Slovo · FT", 87.2, 92.1, 38.7, 33.0),
    ("k2 · Slovo · probe", 82.1, 87.3, 40.3, 31.2),
]


def fig8():
    rows = sorted(((n, (v2 + v3) / 2, (t2 + t3) / 2) for n, v2, v3, t2, t3 in VT), key=lambda r: r[1] - r[2])
    gaps = [v - t for _, v, t in rows]
    assert len(rows) == 23 and abs(sorted(gaps)[11] - 45.2) < 0.06  # медиана из m7_tables §5
    fig, ax = plt.subplots(figsize=(17 * CM, 9.5 * CM), constrained_layout=True)
    for y, (n, v, t) in enumerate(rows):
        ax.plot([t, v], [y, y], color=GRID, lw=2, zorder=1)
        ax.plot(t, y, "o", ms=4.5, color=BLUE, zorder=3, label="Test: unseen participant" if y == 0 else None)
        ax.plot(v, y, "o", ms=4.5, color=ORANGE, mfc="white", mew=1.2, zorder=3,
                label="Validation: training participants" if y == 0 else None)
        ax.annotate(f"+{v - t:.1f}", (v, y), xytext=(5, -2.5), textcoords="offset points", fontsize=6, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.5)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Top-1 accuracy, % (mean of the two folds); label: gap in percentage points")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    grid(ax, "x")
    fig.savefig(OUT / "fig8_valtest.png")
    plt.close(fig)


if __name__ == "__main__":
    fig5(); fig6(); fig7(); fig8()
    print("fig5_cost.png fig6_ablations.png fig7_transfer.png fig8_valtest.png")
