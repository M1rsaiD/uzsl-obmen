"""Рисунок «точность — стоимость» турнира архитектур: одиночки и ансамбли, граница Парето.

Запуск:  python3 pareto_turnir.py turnir_summary.csv [--out ris_turnir] [--x gflops|latency_ms]
Формат таблицы — FORMAT.md. Самопроверка:  python3 pareto_turnir.py --selftest
"""
import argparse
import csv
import math
import sys
from pathlib import Path

REQ = ['dataset', 'model', 'kind', 'members', 'recipe', 'condition', 'top1', 'top1_lo', 'top1_hi', 'ci_level', 'gflops', 'latency_ms', 'params_m']
KINDS = {'single': ('o', 'одиночная'), 'ens_homo': ('s', 'ансамбль однородный'), 'ens_mixed': ('D', 'ансамбль смешанный')}
FAMILY = [('Sohn', '#eb6834'), ('UzSohn', '#eb6834'), ('A1', '#2a78d6'), ('A2', '#2a78d6'), ('MPK', '#2a78d6'),
          ('ST-GCN', '#52514e'), ('SL-GCN', '#52514e')]
DS_RU = {'Slovo': 'Slovo (протокол П)', 'Bukva': 'Bukva (протокол П)', 'UzSL': 'Корпус УзЖЯ (LOSO, предварительно)'}


def color(model):
    for key, c in FAMILY:
        if key in model:
            return c
    return '#9a9893'


def pareto(points):
    """Индексы недоминируемых точек: меньше стоимость и выше точность. points — [(cost, acc)]."""
    order = sorted(range(len(points)), key=lambda k: (points[k][0], -points[k][1]))
    front, best = [], -math.inf
    for k in order:
        if points[k][1] > best:
            front.append(k)
            best = points[k][1]
    return front


def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        sys.exit('таблица пуста')
    miss = [c for c in REQ if c not in rows[0]]
    if miss:
        sys.exit(f'нет столбцов: {miss}')
    for r in rows:
        if r['kind'] not in KINDS:
            sys.exit(f"kind = {r['kind']!r}: допустимы {list(KINDS)}")
        for c in ('top1', 'top1_lo', 'top1_hi', 'gflops', 'latency_ms', 'params_m'):
            r[c] = float(r[c]) if r[c] not in ('', 'нет данных') else math.nan
        r['members'] = int(r['members'])
        if not (r['top1_lo'] <= r['top1'] <= r['top1_hi']):
            sys.exit(f"ДИ не охватывает точку: {r['model']} {r['dataset']}")
    return rows


def plot(rows, out, xcol='gflops', recipe='native', condition='transfer'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    ds = [d for d in ('Slovo', 'Bukva', 'UzSL') if any(r['dataset'] == d for r in rows)]
    fig, axes = plt.subplots(1, len(ds), figsize=(6.2 * len(ds) / 2.54 * 1.45, 7 / 2.54), constrained_layout=True, squeeze=False)
    xlab = {'gflops': 'ГФЛОП на окно (лог. шкала)', 'latency_ms': 'Задержка на ЦП, мс (лог. шкала)'}[xcol]
    for ax, d in zip(axes[0], ds):
        sel = [r for r in rows if r['dataset'] == d and r['recipe'] == recipe and (d != 'UzSL' or r['condition'] == condition)
               and not math.isnan(r[xcol])]
        for r in sel:
            m, _ = KINDS[r['kind']]
            c = '#7b5ea7' if r['kind'] == 'ens_mixed' else color(r['model'])
            ax.errorbar(r[xcol], r['top1'], yerr=[[r['top1'] - r['top1_lo']], [r['top1_hi'] - r['top1']]], fmt=m, ms=4.5,
                        color=c, mfc=c if r['kind'] == 'single' else 'white', mew=1, elinewidth=0.7, capsize=0, zorder=3)
            lab = r['model'] + (f" ×{r['members']}" if r['kind'] == 'ens_homo' else '')
            ax.annotate(lab, (r[xcol], r['top1']), xytext=(4, 3), textcoords='offset points', fontsize=6, color='#3b3a37')
        if sel:
            fr = pareto([(r[xcol], r['top1']) for r in sel])
            pts = sorted(((sel[k][xcol], sel[k]['top1']) for k in fr))
            ax.step([p[0] for p in pts], [p[1] for p in pts], where='post', color='#2a78d6', lw=0.9, ls='--', zorder=2)
        ax.set_xscale('log')
        ax.set_xlabel(xlab)
        ax.set_title(DS_RU.get(d, d), loc='left', fontsize=8)
        ax.grid(color='#d9d8d4', lw=0.5)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel('Точность top-1, %')
    handles = [plt.Line2D([], [], marker=m, ls='', color='#52514e', mfc='#52514e' if k == 'single' else 'white', label=t)
               for k, (m, t) in KINDS.items()] + [plt.Line2D([], [], color='#2a78d6', ls='--', label='граница Парето')]
    axes[0][-1].legend(handles=handles, frameon=False, fontsize=6.5, loc='lower right')
    fig.savefig(f'{out}.png')
    fig.savefig(f'{out}.pdf')
    plt.close(fig)
    return [f'{out}.png', f'{out}.pdf']


def selftest():
    assert pareto([(1, 50), (2, 40), (3, 60), (0.5, 30), (3, 55)]) == [3, 0, 2], 'граница Парето'
    assert pareto([(1, 50), (1, 50)]) == [0]
    import tempfile
    import os
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 's.csv')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(REQ + ['n_seeds', 'source'])
        # синтетические числа — только для проверки кода, не результаты
        for ds in ('Slovo', 'UzSL'):
            for mod, g, a in (('A1', 0.76, 70), ('SL-GCN', 2.19, 74), ('Sohn', 1.2, 72)):
                w.writerow([ds, mod, 'single', 1, 'native', 'transfer', a, a - 2, a + 2, 99, g, 9, 3, 5, 'синтетика'])
                w.writerow([ds, mod, 'ens_homo', 5, 'native', 'transfer', a + 2, a, a + 4, 99, 5 * g, 45, 15, 5, 'синтетика'])
            w.writerow([ds, 'A1+SL-GCN+ST-GCN', 'ens_mixed', 3, 'native', 'transfer', 76, 74, 78, 99, 4.9, 25, 10, 5, 'синтетика'])
    outs = plot(load(p), os.path.join(tmp, 'test'))
    assert all(os.path.getsize(o) > 1000 for o in outs)
    print('самопроверка пройдена:', outs[0])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('table', nargs='?')
    ap.add_argument('--out', default='ris_turnir')
    ap.add_argument('--x', default='gflops', choices=['gflops', 'latency_ms'])
    ap.add_argument('--recipe', default='native', choices=['native', 'shared'])
    ap.add_argument('--condition', default='transfer', choices=['transfer', 'scratch'])
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.table:
        print(plot(load(a.table), a.out, a.x, a.recipe, a.condition))
    else:
        ap.error('укажите таблицу или --selftest')
