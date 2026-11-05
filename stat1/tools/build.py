#!/usr/bin/env python3
"""Сборка статьи 1 v2.0: src/<lang>/*.md + out/numbers.json → tex (cas-dc) → pdf; → docx.

python3 -I tools/build.py en|ru [--no-pdf]
Плейсхолдеры в тексте: {{TABLE:name}}, {{FIG:name}} — подставляются в формате выхода (LaTeX / Markdown).
"""
import json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANG = sys.argv[1] if len(sys.argv) > 1 else 'en'
NOPDF = '--no-pdf' in sys.argv
SRC = ROOT / 'src' / LANG; OUT = ROOT / 'out'; BUILD = ROOT / 'build' / LANG; BUILD.mkdir(parents=True, exist_ok=True)
FIGSRC = Path('/home/claude/dissertatsiya/05-my-articles/2026-uzsl-rsl-transfer/figures')
N = json.load(open(OUT / 'numbers.json', encoding='utf-8'))
RU = LANG == 'ru'

# ---------------------------------------------------------------- тексты подписей (en / ru)
T = {
 'datasets': ('Datasets used in this study. Counts exclude the "no event" videos of Slovo, 400 of which are used as an additional training and validation class.',
              'Наборы данных. Видео «нет события» не учтены; 400 таких видео Slovo используются как дополнительный класс при обучении и валидации.'),
 'corpus': ('Composition of the UzSL corpus. "Evaluation set": classes with at least three recordings from at least two signers (the LOSO test set). Fingerspelling includes numerals.',
            'Состав корпуса УзЖЯ. «Набор оценки» — классы с ≥ 3 записями от ≥ 2 исполнителей (тест LOSO). Дактиль включает цифры.'),
 'rsl_main': ('Keypoint models under SD-5: top-1 and top-5 accuracy, macro-averaged precision, recall and $F_1$ over classes, and the paired difference of top-1 accuracy to MPE-A1 (95% CI over test signers). Rows without a mark: three seeds × five folds; †: released code or recipe, one seed; *: shared recipe without weight averaging. Parameters and GFLOPs are given once, for the Slovo models at the window used at test time (32 frames; 64 for the keypoint Transformer; 150 for the SL-GCN recipe; 300 for ST-GCN with its authors\' recipe); CPU latencies are given in the text.',
              'Модели по ключевым точкам по протоколу SD-5: точность top-1 и top-5, macro-усреднённые по классам точность (precision), полнота и $F_1$, парная разность top-1 с МПК-A1 (95 % ДИ по тестовым исполнителям). Строки без пометки — три затравки × пять фолдов; † — оригинальный код или рецепт, одна затравка; * — общий рецепт без усреднения весов. Параметры и GFLOPs даны один раз, для моделей Slovo на окне проверки (32 кадра; 64 — трансформер по точкам; 150 — рецепт SL-GCN; 300 — ST-GCN с рецептом авторов); задержки на ЦП — в тексте.'),
 'official': ('Official Slovo split (5,100 test videos, 174 signers, 92 of whom appear in training). Three seeds for the keypoint models; MViTv2-S is the single released model. P0: released inference code; R1: short videos padded with the last frame.',
              'Официальное разбиение Slovo (5 100 тестовых видео, 174 исполнителя, 92 из них есть в обучении). Три затравки для моделей по точкам; MViTv2-S — единственная опубликованная модель. P0 — код вывода авторов; R1 — дополнение коротких видео последним кадром.'),
 'loso': ('UzSL corpus, leave-one-signer-out over ten signers (three seeds). Top-1 and top-5: macro-average over signers [95% CI, bootstrap over signers]; top-1 (rec.): micro-average over recordings; precision, recall and $F_1$: macro over classes; words / fingerspelling: top-1 macro over the signers with at least ten recordings of the group. Head: number of classes in the training set.',
          'Корпус УзЖЯ, «исключение одного исполнителя» по десяти исполнителям (три затравки). Top-1 и top-5 — среднее по исполнителям [95 % ДИ, бутстреп по исполнителям]; top-1 (зап.) — среднее по записям; точность (precision), полнота и $F_1$ — macro по классам; слова / дактиль — top-1, среднее по исполнителям с ≥ 10 записями группы. Голова — число классов обучения.'),
 'external': ('External sources, final models trained on the whole corpus (one seed), no adaptation. Only videos whose sign belongs to the model\'s head are scored. Intervals: bootstrap over recordings (N01, one presenter) or over words (dictionary).',
              'Внешние источники, итоговые модели на всём корпусе (одна затравка), без подстройки. Оцениваются только видео, чей знак входит в голову модели. Интервалы — бутстреп по записям (N01, один ведущий) или по словам (словарь).'),
 'pilot': ('UzSL pilot with training budgets fixed in advance (MPE-A1 unless stated; three seeds × two folds): top-1 pooled over the two test participants, with participants 002 / 003 in parentheses. Every pretrained condition is better than training from scratch at the same step in all six seed–fold pairs.',
           'Пилот УзЖЯ с бюджетами, заданными заранее (МПК-A1, если не указано; три затравки × два фолда): top-1 по записям обоих тестовых участников, в скобках — участники 002 / 003. Каждое условие с предобучением лучше обучения с нуля на той же ступени во всех шести парах.'),
 'robustness': ('Top-1 accuracy of MPE-A1 and MPE-A2 under keypoint corruption (SD-5, three seeds × five folds). $p$: share of frames without the dominant hand; $\\sigma$: Gaussian noise as a fraction of the frame width; $\\Delta$: paired difference A2 − A1 [95% CI].',
                'Точность top-1 МПК-A1 и МПК-A2 при искажении точек (SD-5, три затравки × пять фолдов). $p$ — доля кадров без ведущей кисти; $\\sigma$ — гауссов шум в долях ширины кадра; $\\Delta$ — парная разность A2 − A1 [95 % ДИ].'),
}
def cap(key): return T[key][1] if RU else T[key][0]

def fmt(x, d=1):
    s = f'{x:.{d}f}'
    return s.replace('.', ',') if RU else s
def ci(v, d=1): return f'[{fmt(v[0], d)}; {fmt(v[1], d)}]'
def delta(dd, d=1):
    if not dd: return '---'
    return f'{"+" if dd["delta"] >= 0 else "−"}{fmt(abs(dd["delta"]), d)} [{"+" if dd["ci"][0] >= 0 else "−"}{fmt(abs(dd["ci"][0]), d)}; {"+" if dd["ci"][1] >= 0 else "−"}{fmt(abs(dd["ci"][1]), d)}]'
def esc(s): return s.replace('%', r'\%').replace('&', r'\&').replace('_', r'\_') if not s.startswith('$') else s.replace('%', r'\%').replace('&', r'\&')
def capesc(s): return s.replace('%', r'\%').replace('&', r'\&')
def tex_table(label, caption, header, rows, align, wide=True, size=r'\small', notes=None, colsep=None, tophead=None):
    caption = capesc(caption)
    env = 'table*' if wide else 'table'
    if colsep: size += f'\\setlength{{\\tabcolsep}}{{{colsep}}}'
    n = len(header)
    def line(r):  # str → group row spanning all columns
        if isinstance(r, str): return f'\\multicolumn{{{n}}}{{l}}{{\\textit{{{esc(r)}}}}}'
        return ' & '.join(esc(str(c)) if not str(c).startswith('$') else str(c) for c in r)
    body = ' \\\\\n'.join(line(r) for r in rows)
    head = ' & '.join(f'\\textbf{{{esc(h)}}}' for h in header)
    if tophead: head = tophead + '\n' + head
    return (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\\label{{tab:{label}}}\n{size}\n\\centering\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{head} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")
def md_table(label, caption, header, rows):
    out = [f'**{("Таблица" if RU else "Table")} {TABNO[label]}.** {caption}', '', '| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c).replace('|', '/') for c in (r if not isinstance(r, str) else [f'*{r}*'] + [''] * (len(header) - 1))) + ' |' for r in rows]
    return '\n'.join(out) + '\n'
TABNO = {'datasets': 1, 'corpus': 2, 'rsl_main': 3, 'official': 4, 'loso': 5, 'external': 6, 'pilot': 7, 'robustness': 8}
FIGNO = {'fig1_signs': 1, 'fig2_protocol': 2, 'fig3_architecture': 3, 'fig5_cost': 4, 'fig6_ablations': 5, 'fig_loso_signers': 6, 'fig7_transfer': 7, 'fig8_valtest': 8, 'fig_confusions': 9}

# ---------------------------------------------------------------- таблицы
def t_datasets():
    h = ['Dataset', 'Language', 'Content', 'Videos', 'Classes', 'Signers', 'Role'] if not RU else ['Набор', 'Язык', 'Содержание', 'Видео', 'Классы', 'Исп.', 'Роль']
    r = [['Slovo', 'RSL' if not RU else 'РЖЯ', 'isolated words' if not RU else 'изолированные слова', '20,000', '1,000', '194', 'comparison; pretraining' if not RU else 'сравнение; предобучение'],
         ['Bukva', 'RSL' if not RU else 'РЖЯ', 'fingerspelling' if not RU else 'дактиль', '3,757', '33', '155', 'comparison; pretraining; simulation' if not RU else 'сравнение; предобуч.; моделирование'],
         ['UzSL pilot (ours)' if not RU else 'Пилот УзЖЯ (наш)', 'UzSL' if not RU else 'УзЖЯ', 'fingerspelling, numerals' if not RU else 'дактиль, цифры', '322', '40', '3', 'transfer target' if not RU else 'цель переноса'],
         ['UzSL corpus (ours)' if not RU else 'Корпус УзЖЯ (наш)', 'UzSL' if not RU else 'УзЖЯ', 'isolated signs (lessons + pilot)' if not RU else 'изолированные жесты (уроки + пилот)', '5,993', '2,381', '12', 'LOSO; transfer target' if not RU else 'LOSO; цель переноса'],
         ['N01 (external)' if not RU else 'N01 (внешний)', 'UzSL' if not RU else 'УзЖЯ', 'video dictionary' if not RU else 'видеословарь', '191', '191', '1', 'test only' if not RU else 'только тест'],
         ['imo-ishora.uz (external)' if not RU else 'imo-ishora.uz (внешний)', 'UzSL' if not RU else 'УзЖЯ', 'video dictionary' if not RU else 'видеословарь', '1,556', '1,556', 'several' if not RU else 'несколько', 'test only' if not RU else 'только тест']]
    if RU: r = [[c.replace(',', '\u2009') if isinstance(c, str) and re.fullmatch(r'[\d,]+', c) else c for c in row] for row in r]
    return h, r, 'lllrrrl'
def t_corpus():
    h = (['Signer', 'Source', 'Recordings', 'Classes', 'Fingersp. / words', 'Recordings', 'Classes', 'Fingersp. / words'] if not RU else
         ['Исп.', 'Источник', 'Записей', 'Классов', 'Дактиль / слова', 'Записей', 'Классов', 'Дактиль / слова'])
    rows = [('S01', 'C01', 824, 701, '0 / 824', 382, 302, '0 / 382'), ('S02', 'C02', 1831, 835, '3 / 1828', 1095, 446, '3 / 1092'), ('S03', 'C03', 1005, 598, '52 / 953', 590, 318, '20 / 570'),
            ('S04', 'C04', 199, 156, '0 / 199', 125, 96, '0 / 125'), ('S05', 'C06', 362, 344, '1 / 361', 183, 169, '1 / 182'), ('S06', 'C07', 18, 14, '0 / 18', 11, 8, '0 / 11'),
            ('S07', 'C05, C08', 1146, 979, '13 / 1133', 648, 512, '9 / 639'), ('S08', 'C09', 173, 104, '0 / 173', 71, 36, '0 / 71'), ('S09', 'C10', 113, 86, '0 / 113', 74, 55, '0 / 74'),
            ('S10', 'pilot' if not RU else 'пилот', 3, 1, '3 / 0', 3, 1, '3 / 0'), ('S11', 'pilot' if not RU else 'пилот', 143, 29, '143 / 0', 118, 24, '118 / 0'), ('S12', 'pilot' if not RU else 'пилот', 176, 35, '176 / 0', 118, 24, '118 / 0'),
            ('Total' if not RU else 'Всего', '10 + pilot' if not RU else '10 + пилот', 5993, 2381, '391 / 5602', 3418, 682, '272 / 3146')]
    f = (lambda n: f'{n:,}'.replace(',', '\u2009')) if RU else (lambda n: f'{n:,}')
    r = [[a, b, f(c), f(d), e, f(g), f(hh), i] for a, b, c, d, e, g, hh, i in rows]
    return h, r, 'llrrcrrc'
TOPHEAD = {'corpus': ('& & \\multicolumn{3}{c}{\\textbf{All recordings}} & \\multicolumn{3}{c}{\\textbf{Evaluation set}} \\\\ \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}',
                      '& & \\multicolumn{3}{c}{\\textbf{Все записи}} & \\multicolumn{3}{c}{\\textbf{Набор оценки}} \\\\ \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}')}
RSL_LABELS = {'A1': ('MPE-A1', 'МПК-A1'), 'ST-GCN released': ('ST-GCN, released †', 'ST-GCN, оригинал †'), 'SL-GCN released': ('SL-GCN, released †', 'SL-GCN, оригинал †'),
              'ST-GCN released authors': ("ST-GCN, released, authors' recipe †", 'ST-GCN, оригинал, рецепт авторов †'), 'SL-GCN released authors': ("SL-GCN, released, authors' recipe †", 'SL-GCN, оригинал, рецепт авторов †'),
              'A1 SL-GCN recipe': ('MPE-A1, SL-GCN recipe †', 'МПК-A1, рецепт SL-GCN †'), 'ST-GCN reimpl': ('ST-GCN, re-implemented', 'ST-GCN, наша реализация'), 'SL-GCN reimpl': ('SL-GCN, re-implemented', 'SL-GCN, наша реализация'),
              'KP Transformer': ('Keypoint Transformer', 'Трансформер по точкам'), '2-layer Transformer': ('Two-layer Transformer *', 'Двухслойный трансформер *'), 'BiLSTM': ('BiLSTM with attention *', 'BiLSTM с вниманием *'), 'MLP': ('MLP *', 'MLP *')}
COST = {  # params M, GFLOPs, CPU ms — из cost/*.json и статьи 1 (окно проверки)
    'A1': (3.38, 0.76, 9.3), 'ST-GCN released': (3.34, 1.98, 5.7), 'SL-GCN released': (5.07, 2.19, 23.0), 'ST-GCN released authors': (3.34, 18.54, 23.2), 'SL-GCN released authors': (5.07, 9.70, 26.5),
    'A1 SL-GCN recipe': (3.40, 3.59, 21.2), 'ST-GCN reimpl': (3.41, 1.98, 5.1), 'SL-GCN reimpl': (3.93, 2.03, 19.2), 'KP Transformer': (3.53, 0.43, 1.8), '2-layer Transformer': (1.98, 0.11, 1.7), 'BiLSTM': (11.00, 0.62, 5.2), 'MLP': (1.55, 0.004, 1.0)}
RSL_ORDER = ['A1', 'ST-GCN released', 'SL-GCN released', 'ST-GCN released authors', 'SL-GCN released authors', 'A1 SL-GCN recipe', 'ST-GCN reimpl', 'SL-GCN reimpl', 'KP Transformer', '2-layer Transformer', 'BiLSTM', 'MLP']
def t_rsl_main():
    h = (['Model', 'Params (M)', 'GFLOPs', 'Top-1 [95% CI]', 'Top-5', 'Precision', 'Recall', '$F_1$', '$\\Delta$ top-1 vs MPE-A1 [95% CI]'] if not RU else
         ['Модель', 'Парам., млн', 'GFLOPs', 'Top-1 [95 % ДИ]', 'Top-5', 'Точность', 'Полнота', '$F_1$', '$\\Delta$ top-1 с МПК-A1 [95 % ДИ]'])
    rows = []
    for ds, title in (('slovo', 'Slovo (1,000 classes, 188 test signers)' if not RU else 'Slovo (1 000 классов, 188 тестовых исполнителей)'), ('bukva', 'Bukva (33 classes, 155 test signers)' if not RU else 'Bukva (33 класса, 155 тестовых исполнителей)')):
        rows.append(title)
        for key in RSL_ORDER:
            m = N['rsl'][ds].get(key)
            if m is None: continue
            p, g, c = COST[key]
            cost = [fmt(p, 2), fmt(g, 2 if g >= 0.01 else 3)] if ds == 'slovo' else ['', '']
            rows.append([RSL_LABELS[key][1 if RU else 0]] + cost + [f'{fmt(m["top1"])} {ci(m["ci"])}', fmt(m['top5']), fmt(m['precision']), fmt(m['recall']), fmt(m['f1']), delta(m.get('delta_vs_A1')) if key != 'A1' else '---'])
    return h, rows, 'lrrlrrrrl'
def t_official():
    h = ['Model', 'Input', 'Top-1 [95% CI]', 'Top-5', '$\\Delta$ to MPE-A1'] if not RU else ['Модель', 'Вход', 'Top-1 [95 % ДИ]', 'Top-5', '$\\Delta$ с МПК-A1']
    lab = {'A1': ('MPE-A1', 'МПК-A1'), 'A0': ('MPE-A0', 'МПК-A0'), 'A2': ('MPE-A2 (no graph)', 'МПК-A2 (без графа)'), 'A1 large': ('MPE, D = 256, N = 6', 'МПК, D = 256, N = 6'), 'KP Transformer (own recipe)': ('Keypoint Transformer', 'Трансформер по точкам'),
           'MViTv2-S 32/2 P0': ('MViTv2-S, released code (P0)', 'MViTv2-S, код авторов (P0)'), 'MViTv2-S 32/2 R1': ('MViTv2-S, last-frame padding (R1)', 'MViTv2-S, дополнение последним кадром (R1)'), 'MViTv2-S 48/2 R1': ('MViTv2-S 48/2 (R1)', 'MViTv2-S 48/2 (R1)'), 'ResNet-i3d R1': ('ResNet-i3d (R1)', 'ResNet-i3d (R1)')}
    rows = []
    for k, (en, ru) in lab.items():
        m = N['official_slovo'][k]; inp = ('keypoints' if not RU else 'точки') if 'MViT' not in k and 'ResNet' not in k else ('RGB video' if not RU else 'RGB-видео')
        rows.append([ru if RU else en, inp, f'{fmt(m["top1"])} {ci(m["ci"])}', fmt(m['top5']), delta(m.get('delta_vs_A1')) if k != 'A1' else '---'])
    rows.insert(5, ['MViTv2-S, reported by the authors of Slovo' if not RU else 'MViTv2-S, по данным авторов Slovo', 'RGB video' if not RU else 'RGB-видео', fmt(64.09, 2), '---', '---'])
    return h, rows, 'lllrl'
def t_loso():
    h = (['Model', 'Pretraining', 'Head', 'Top-1 [95% CI]', 'Top-1 (rec.)', 'Top-5', 'Precision', 'Recall', '$F_1$', 'Words', 'Fingersp.'] if not RU else
         ['Модель', 'Предобучение', 'Голова', 'Top-1 [95 % ДИ]', 'Top-1 (зап.)', 'Top-5', 'Точность', 'Полнота', '$F_1$', 'Слова', 'Дактиль'])
    SB = 'Slovo + Bukva'; NO = 'none' if not RU else 'нет'
    lab = {'step1': ('MPE-A1', SB), 'B1': ('MPE-A1', NO), 'A': ('MPE-A1', SB), 'B': ('MPE-A1', NO), 'SLGCN': ('SL-GCN (released)' if not RU else 'SL-GCN (оригинал)', SB), 'modelV': ('MPE-A1', SB + ' + ASL')}
    rows = []
    for k, (mod, pre) in lab.items():
        m = N['loso'][k]; head = '682' if k in ('step1', 'B1', 'SLGCN', 'modelV') else ('2,033–2,378' if not RU else '2 033–2 378')
        rows.append([mod.replace('MPE', 'МПК') if RU else mod, pre, head, f'{fmt(m["top1_macro"])} {ci(m["ci"])}', fmt(m['top1_micro']), fmt(m['top5_macro']), fmt(m['precision']), fmt(m['recall']), fmt(m['f1']), fmt(m['words']['macro']), fmt(m['fingerspelling']['macro'])])
    return h, rows, 'lllcrrrrrrr'
def t_external():
    h = (['Source', 'Model', 'Head', 'Videos scored', 'Words', 'Top-1 [95% CI]', 'Top-5'] if not RU else ['Источник', 'Модель', 'Голова', 'Видео', 'Слов', 'Top-1 [95 % ДИ]', 'Top-5'])
    lab = {'step1': ('pretrained (Slovo + Bukva)', 'предобучение (Slovo + Bukva)'), 'modelV': ('pretrained (Slovo + Bukva + ASL)', 'предобучение (Slovo + Bukva + ASL)'), 'A': ('pretrained, all recordings', 'предобучение, все записи'), 'B': ('from scratch, all recordings', 'с нуля, все записи')}
    rows = []
    for src, name in (('N01', 'N01'), ('IMO', 'imo-ishora.uz')):
        for k, (en, ru) in lab.items():
            m = N['external'][f'{src}/{k}']
            rows.append([name, ru if RU else en, f'{m["head_size"]:,}' if not RU else f'{m["head_size"]:,}'.replace(',', '\u2009'), str(m['n_in_head']), str(m['n_words']), f'{fmt(m["top1"])} {ci(m["ci"])}', fmt(m['top5'])])
    return h, rows, 'llrrrlr'
def t_pilot():
    h = (['Pretraining', 'Scheme', 'One example per class', 'Two examples per class', 'All recordings'] if not RU else ['Предобучение', 'Схема', 'Один пример на класс', 'Два примера на класс', 'Все записи'])
    P = N['pilot']
    def cell(c):
        v = P[c]; pp = v['per_participant']; return f'{fmt(v["pooled"])} ({fmt(pp["2"])} / {fmt(pp["3"])})'
    spec = [('None' if not RU else 'Нет', 'from scratch' if not RU else 'с нуля', 'pilot_{k}_scratch_a1'), ('Slovo', 'linear probe' if not RU else 'линейный зонд', 'pilot_{k}_slovo_probe_a1'), ('Slovo', 'fine-tuning' if not RU else 'дообучение', 'pilot_{k}_slovo_ft_a1'),
            ('Bukva', 'linear probe' if not RU else 'линейный зонд', 'pilot_{k}_bukva_probe_a1'), ('Bukva', 'fine-tuning' if not RU else 'дообучение', 'pilot_{k}_bukva_ft_a1'),
            ('Slovo + Bukva', 'linear probe' if not RU else 'линейный зонд', 'pilot_{k}_sb_probe_a1'), ('Slovo + Bukva', 'fine-tuning' if not RU else 'дообучение', 'pilot_{k}_sb_ft_a1'),
            ('Slovo + Bukva, MPE-A2' if not RU else 'Slovo + Bukva, МПК-A2', 'linear probe' if not RU else 'линейный зонд', 'pilot_{k}_sb_probe_a2'), ('Slovo + Bukva, MPE-A2' if not RU else 'Slovo + Bukva, МПК-A2', 'fine-tuning' if not RU else 'дообучение', 'pilot_{k}_sb_ft_a2'),
            ('Slovo + Bukva, SL-GCN (released)' if not RU else 'Slovo + Bukva, SL-GCN (оригинал)', 'linear probe' if not RU else 'линейный зонд', 'pilot_{k}_sb_probe_slgcn_orig'), ('Slovo + Bukva, SL-GCN (released)' if not RU else 'Slovo + Bukva, SL-GCN (оригинал)', 'fine-tuning' if not RU else 'дообучение', 'pilot_{k}_sb_ft_slgcn_orig'),
            ('None, SL-GCN (released)' if not RU else 'Нет, SL-GCN (оригинал)', 'from scratch' if not RU else 'с нуля', 'pilot_{k}_scratch_slgcn_orig')]
    rows = [[a, b, cell(c.format(k='k1')), cell(c.format(k='k2')), cell(c.format(k='all'))] for a, b, c in spec]
    return h, rows, 'lllll'
def t_robustness():
    h = (['Corruption', 'Slovo A1', 'Slovo A2', '$\\Delta$ Slovo', 'Bukva A1', 'Bukva A2', '$\\Delta$ Bukva'] if not RU else ['Искажение', 'Slovo A1', 'Slovo A2', '$\\Delta$ Slovo', 'Bukva A1', 'Bukva A2', '$\\Delta$ Bukva'])
    R = N['robustness']; rows = []
    for p, s, lab in (('0', '0', 'None' if not RU else 'Нет'), ('0.25', '0', '$p = 0.25$'), ('0.5', '0', '$p = 0.5$'), ('0', '0.01', '$\\sigma = 0.01$'), ('0', '0.02', '$\\sigma = 0.02$'), ('0.5', '0.02', '$p = 0.5$, $\\sigma = 0.02$')):
        a = R[f'slovo p={p} σ={s}']; b = R[f'bukva p={p} σ={s}']
        rows.append([lab if not RU else lab.replace('0.25', '0,25').replace('0.5', '0,5').replace('0.01', '0,01').replace('0.02', '0,02'), fmt(a['A1']), fmt(a['A2']), delta(a['delta_A2_A1']), fmt(b['A1']), fmt(b['A2']), delta(b['delta_A2_A1'])])
    return h, rows, 'lrrlrrl'
TABLES = {'datasets': t_datasets, 'corpus': t_corpus, 'rsl_main': t_rsl_main, 'official': t_official, 'loso': t_loso, 'external': t_external, 'pilot': t_pilot, 'robustness': t_robustness}
WIDE = {'datasets': True, 'corpus': True, 'rsl_main': True, 'official': True, 'loso': True, 'external': True, 'pilot': True, 'robustness': True}
SIZE = {'rsl_main': r'\scriptsize', 'pilot': r'\footnotesize', 'corpus': r'\footnotesize', 'loso': r'\scriptsize', 'external': r'\footnotesize', 'official': r'\footnotesize', 'datasets': r'\footnotesize'}
COLSEP = {'rsl_main': '3pt', 'loso': '3pt', 'corpus': '4pt', 'datasets': '4pt'}
if RU: SIZE['datasets'] = r'\scriptsize'; COLSEP['rsl_main'] = '2pt'; COLSEP['datasets'] = '3pt'

# ---------------------------------------------------------------- рисунки
FIGS = {  # name: (file, caption en, caption ru, wide)
 'fig1_signs': ('fig1_signs.png', 'Example fingerspelling signs from the UzSL pilot: participants 002 and 003 (rows) and the letters L, M, O and SH (columns). Each frame is shown next to the same frame with the 85 keypoints and the edges of the 27-node graph; long-range edges are dashed.',
                'Примеры знаков дактиля из пилота УзЖЯ: участники 002 и 003 (строки) и буквы L, M, O и SH (столбцы). Рядом с каждым кадром — тот же кадр с 85 точками и рёбрами графа из 27 узлов; дальние рёбра — пунктиром.', True),
 'fig2_protocol': ('fig2_protocol.png', 'Signer-disjoint five-fold protocol (SD-5). (a) Fold assignment for Slovo: six signers with at least 500 videos are always in the training data; the other 188 signers form five groups, each used once as the test set, and 10% of the signers of the other groups form the validation set. (b) Decision rule: a model is called better or worse only if the 95% bootstrap confidence interval of the paired difference over test signers excludes zero (illustrative distributions).',
                   'Протокол из пяти фолдов с непересекающимися исполнителями (SD-5). (а) Распределение по фолдам для Slovo: шесть исполнителей с ≥ 500 видео всегда в обучении; остальные 188 образуют пять групп, каждая один раз служит тестом, 10 % исполнителей остальных групп — валидация. (б) Правило решения: модель лучше или хуже, только если 95 %-й бутстреп-интервал парной разности по тестовым исполнителям не содержит нуля (распределения условные).', True),
 'fig3_architecture': ('fig3_architecture.png', 'Architecture of the Multilingual Pose Encoder (MPE). (a) Overview: normalisation, spatial encoder on the 27-node graph with joint and bone streams, body-part tokens, temporal encoder and attention pooling; solid arrows show the inference path and dashed arrows the training-only heads. (b) Graph convolution block with three neighbourhood subsets. (c) Conformer layer. (d) Classification head per dataset and auxiliary heads used during training.',
                       'Архитектура многоязычного позового кодировщика (МПК). (а) Общая схема: нормировка, пространственный кодировщик на графе из 27 узлов с потоками суставов и костей, токены частей тела, временной кодировщик и пулинг вниманием; сплошные стрелки — путь распознавания, пунктир — головы только для обучения. (б) Блок графовой свёртки с тремя подмножествами окрестности. (в) Слой Conformer. (г) Классификационная голова для каждого набора и вспомогательные головы обучения.', True),
 'fig5_cost': ('fig5_cost.pdf', 'Top-1 accuracy under SD-5 versus computation per window at test time (logarithmic axis) on (a) Slovo and (b) Bukva. Filled circle: MPE-A1; open circles: MPE-A2 and the larger MPE; squares: baselines with the shared recipe (ST-GCN and SL-GCN from released code, Slovo only); diamonds: released code with the authors\' recipes. Error bars: 95% CIs over test signers.',
               'Точность top-1 по SD-5 и объём вычислений на окно при проверке (логарифмическая шкала): (а) Slovo, (б) Bukva. Закрашенный круг — МПК-A1; незакрашенные — МПК-A2 и большая МПК; квадраты — базовые с общим рецептом (ST-GCN и SL-GCN — оригинальный код, только Slovo); ромбы — оригинальный код с рецептами авторов. Отрезки — 95 % ДИ по исполнителям.', True),
 'fig6_ablations': ('fig6_ablations.pdf', 'Ablations of MPE-A1 under SD-5 (three seeds × five folds): paired differences to MPE-A1 with 95% CIs over test signers. Filled markers: the interval excludes zero.',
                    'Абляции МПК-A1 по SD-5 (три затравки × пять фолдов): парные разности с МПК-A1 и 95 % ДИ по тестовым исполнителям. Закрашенные маркеры — интервал не содержит нуля.', True),
 'fig_loso_signers': ('fig_loso_signers.pdf', 'UzSL corpus: top-1 accuracy on each held-out signer (LOSO, mean of three seeds) for MPE-A1 pretrained on Slovo + Bukva, MPE-A1 trained from scratch and the released SL-GCN pretrained on Slovo + Bukva (682-class evaluation set).',
                      'Корпус УзЖЯ: точность top-1 на каждом отложенном исполнителе (LOSO, среднее трёх затравок) для МПК-A1 с предобучением на Slovo + Bukva, МПК-A1 с нуля и оригинального SL-GCN с предобучением на Slovo + Bukva (набор оценки, 682 класса).', True),
 'fig7_transfer': ('fig7_transfer.pdf', 'Transfer from RSL. (a) Simulation on Bukva under SD-5 (MPE-A0): top-1 accuracy with 1, 3 and 10 training recordings per class and with all recordings, from scratch and after pretraining on Slovo; error bars: 95% CIs over test signers. (b) UzSL pilot (MPE-A1, full fine-tuning): top-1 accuracy pooled over the two test participants with one and two examples per class and with all recordings; vertical bars span the two participants.',
                   'Перенос с РЖЯ. (а) Моделирование на Bukva по SD-5 (МПК-A0): точность top-1 при 1, 3 и 10 обучающих записях на класс и при всех записях, с нуля и после предобучения на Slovo; отрезки — 95 % ДИ по исполнителям. (б) Пилот УзЖЯ (МПК-A1, полное дообучение): top-1 по записям обоих тестовых участников при одном и двух примерах на класс и при всех записях; вертикальные отрезки охватывают двух участников.', True),
 'fig8_valtest': ('fig8_valtest.pdf', 'Validation accuracy on held-out recordings of the training participants (at the selected checkpoint) against test accuracy on the unseen participant for the 23 conditions of the early pilot experiments (MPE-A0), averaged over the two folds and sorted by the gap. k1, k2: one and two examples per class; all: all recordings; FT: full fine-tuning; probe: linear probe.',
                  'Валидационная точность на отложенных записях участников обучения (в выбранной контрольной точке) и тестовая точность на новом участнике для 23 условий первых опытов на пилоте (МПК-A0), среднее по двум фолдам, по убыванию разрыва. k1, k2 — один и два примера на класс; all — все записи; FT — полное дообучение; probe — линейный зонд.', True),
 'fig_confusions': ('fig_confusions.pdf', 'Confusion matrices (rows normalised; off-diagonal cells of at least 5% are labelled). (a) Bukva, MPE-A1 under SD-5, 33 letters of the RSL manual alphabet. (b) UzSL pilot, MPE-A1 pretrained on Slovo + Bukva and fine-tuned on all recordings, 29 letters and 11 numerals; the numerals and the letters H, I, Q, V, Y occur in the recordings of one participant only and are therefore absent from the training data of their test fold.',
                    'Матрицы ошибок (строки нормированы; подписаны внедиагональные ячейки ≥ 5 %). (а) Bukva, МПК-A1 по SD-5, 33 буквы дактильной азбуки РЖЯ. (б) Пилот УзЖЯ, МПК-A1 с предобучением на Slovo + Bukva и дообучением на всех записях, 29 букв и 11 цифр; цифры и буквы H, I, Q, V, Y есть только у одного участника и потому отсутствуют в обучении своего тестового фолда.', True),
}
def make_confusions_figure(path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), gridspec_kw={'width_ratios': [33, 40]})
    for ax, (data, title) in zip(axes, ((N['rsl']['bukva']['confusion_matrix'], '(a) Bukva, MPE-A1, SD-5'), (N['pilot']['confusion_all_sb_ft_a1'], '(b) UzSL pilot, MPE-A1 pretrained on Slovo + Bukva'))):
        cls = data['classes']; M = np.array(data['row_normalized'])
        order = sorted(range(len(cls)), key=lambda i: (0, int(cls[i])) if cls[i].isdigit() else (1, cls[i]))
        cls = [cls[i] for i in order]; M = M[np.ix_(order, order)]
        n = len(cls); im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(cls, rotation=90, fontsize=4.5); ax.set_yticklabels(cls, fontsize=4.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title, fontsize=6.5)
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] >= 0.05: ax.text(j, i, f'{M[i, j]*100:.0f}', ha='center', va='center', fontsize=3.3)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label='Row share')
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)
def fig_block(name):
    f, cen, cru, wide = FIGS[name]; cap = cru if RU else cen; capt = capesc(cap)
    if name == 'fig_confusions':
        make_confusions_figure(BUILD / 'figures' / 'fig_confusions.pdf')
    elif (OUT / 'figures' / f).exists(): shutil.copy(OUT / 'figures' / f, BUILD / 'figures' / f)
    else: shutil.copy(FIGSRC / f, BUILD / 'figures' / f)
    env = 'figure*' if wide else 'figure'
    tex = f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={'\\textwidth' if wide else '\\columnwidth'}]{{figures/{f}}}\n\\caption{{{capt}}}\\label{{fig:{name}}}\n\\end{{{env}}}\n"
    png = f if f.endswith('.png') else f[:-4] + '.png'
    if not (BUILD / 'figures' / png).exists(): subprocess.run(['pdftoppm', '-png', '-r', '200', '-singlefile', str(BUILD / 'figures' / f), str(BUILD / 'figures' / png[:-4])], check=True)
    md = f"![**{('Рисунок' if RU else 'Fig.')} {FIGNO[name]}.** {cap}](figures/{png})\n"
    return tex, md

# ---------------------------------------------------------------- сборка текста
def assemble():
    (BUILD / 'figures').mkdir(exist_ok=True)
    md = '\n\n'.join(open(p, encoding='utf-8').read() for p in sorted(SRC.glob('*.md')))
    meta = re.match(r'---\n(.*?)\n---\n', md, re.S); front = meta.group(1); md = md[meta.end():]
    tex_md, docx_md = md, md
    for key, fn in TABLES.items():
        h, rows, align = fn(); c = cap(key)
        tex_md = tex_md.replace(f'{{{{TABLE:{key}}}}}', '```{=latex}\n' + tex_table(key, c, h, rows, align, WIDE[key], SIZE.get(key, r'\small'), colsep=COLSEP.get(key), tophead=(TOPHEAD[key][1 if RU else 0] if key in TOPHEAD else None)) + '```\n')
        docx_md = docx_md.replace(f'{{{{TABLE:{key}}}}}', md_table(key, c, h, rows))
    for name in FIGS:
        tex, m = fig_block(name)
        tex_md = tex_md.replace(f'{{{{FIG:{name}}}}}', '```{=latex}\n' + tex + '```\n'); docx_md = docx_md.replace(f'{{{{FIG:{name}}}}}', m)
    assert '{{' not in tex_md, re.findall(r'\{\{[^}]*\}\}', tex_md)
    return front, tex_md, docx_md

def block(md, name):
    m = re.search(rf'::: {name}\n(.*?)\n:::', md, re.S); return m.group(1).strip() if m else ''

def tex_fix(s):
    """Символы вне T1 → LaTeX; маркеры ⟦…⟧ → красный текст."""
    rep = {'⟦': r'{\color{red}[', '⟧': ']}', '≥': r'$\geq$', '≤': r'$\leq$', '×': r'$\times$', '±': r'$\pm$', '→': r'$\rightarrow$', '−': r'$-$', 'ʻ': '‘', '≈': r'$\approx$', '∞': r'$\infty$', '·': r'$\cdot$', '\u2009': r'\,', '\u00a0': '~'}
    for a, b in rep.items(): s = s.replace(a, b)
    if RU: s = re.sub(r'(\d) \\%', r'\1~\\%', s)   # «70,8 %» — неразрывный пробел перед знаком процента
    return s

def build_tex(front, tex_md):
    title = re.search(r'^title: "(.*)"$', front, re.M).group(1); short = re.search(r'^short_title: "(.*)"$', front, re.M).group(1)
    abstract = block(tex_md, 'abstract'); keywords = block(tex_md, 'keywords'); highlights = block(tex_md, 'highlights')
    body_md = re.sub(r'::: (abstract|keywords|highlights)\n.*?\n:::\n', '', tex_md, flags=re.S)
    (BUILD / 'body.md').write_text(body_md, encoding='utf-8')
    subprocess.run(['pandoc', str(BUILD / 'body.md'), '-f', 'markdown', '-t', 'latex', '--natbib', '--wrap=none', '--top-level-division=section', '-o', str(BUILD / 'body.tex')], check=True)
    body = (BUILD / 'body.tex').read_text(encoding='utf-8')
    body = tex_fix(body.replace(r'\citep{', r'\cite{').replace(r'\citet{', r'\cite{'))
    body = re.sub(r'\\section\{(CRediT|Declaration of competing|Funding|Ethics statement|Data availability|Declaration of generative|Acknowledgements|Вклад авторов|Декларация|Финансирование|Этическое|Доступность|Благодарности)', r'\\section*{\1', body)
    # пандок ставит \[ \] для $$; оставить. Таблицы/рисунки — raw LaTeX, прошли насквозь.
    hl = '\n'.join(f'\\item {l[2:]}' for l in highlights.splitlines() if l.startswith('- '))
    kw = ' \\sep '.join(k.strip() for k in keywords.split(';'))
    bios = BIOS_RU if RU else BIOS_EN
    tex = (TEMPLATE_RU if RU else TEMPLATE).replace('@TITLE@', tex_fix(capesc(title))).replace('@SHORT@', tex_fix(capesc(short))).replace('@ABSTRACT@', tex_fix(capesc(abstract))).replace('@KEYWORDS@', kw).replace('@HIGHLIGHTS@', tex_fix(capesc(hl))).replace('@BODY@', body).replace('@BIOS@', bios)
    (BUILD / 'main.tex').write_text(tex, encoding='utf-8')
    # библиография
    subprocess.run(['pandoc', str(ROOT / 'src' / 'refs_v2.json'), '-f', 'csljson', '-t', 'bibtex', '-o', str(BUILD / 'refs.bib')], check=True)
    bib = (BUILD / 'refs.bib').read_text(encoding='utf-8')
    for a, b in {'Ł': '{\\L}', 'ł': '{\\l}', 'é': "{\\'e}", 'É': "{\\'E}", 'ć': "{\\'c}", 'ü': '{\\"u}', 'ö': '{\\"o}', 'ä': '{\\"a}', 'ñ': '{\\~n}', 'í': "{\\'\\i}", 'á': "{\\'a}", 'ó': "{\\'o}", 'è': '{\\`e}', 'ç': '{\\c{c}}', '’': "'", '‘': '`', '–': '--', '—': '---', 'ʻ': "'", '×': '$\\times$'}.items():
        bib = bib.replace(a, b)
    assert all(ord(ch) < 128 for ch in bib), [ch for ch in set(bib) if ord(ch) >= 128]
    (BUILD / 'refs.bib').write_text(bib, encoding='utf-8')
    if not NOPDF:
        r = subprocess.run(['latexmk', '-pdf', '-g', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=BUILD, capture_output=True, text=True, errors='replace')
        print(r.stdout[-3000:] if r.returncode else 'PDF ok');
        if r.returncode: print(r.stderr[-2000:]); sys.exit(1)

TEMPLATE = r"""\documentclass[a4paper,fleqn]{cas-dc}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb,booktabs,graphicx,url,xcolor}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\hypersetup{hidelinks}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}
\shorttitle{@SHORT@}
\shortauthors{M.A. Yusupov and M.K. Onarkulov}
\title[mode=title]{@TITLE@}
\author[1]{Mirsaid A. Yusupov}[orcid=0009-0001-1989-0088]
\cormark[1]
\ead{mirsaidbeky@gmail.com}
\affiliation[1]{organization={Department of Applied Mathematics and Informatics, Fergana State University}, city={Fergana}, postcode={150100}, country={Uzbekistan}}
\author[2]{Maksadjon K. Onarkulov}[orcid=0000-0002-3650-4847]
\ead{maxmaqsad@gmail.com}
\affiliation[2]{organization={Faculty of Physics and Mathematics, Fergana State University}, city={Fergana}, postcode={150100}, country={Uzbekistan}}
\cortext[1]{Corresponding author.}
\begin{abstract}
@ABSTRACT@
\end{abstract}
\begin{highlights}
@HIGHLIGHTS@
\end{highlights}
\begin{keywords}
@KEYWORDS@
\end{keywords}
\maketitle
@BODY@
\bibliographystyle{elsarticle-num}
\bibliography{refs}
@BIOS@
\end{document}
"""
TEMPLATE_RU = r"""\documentclass[a4paper,fleqn]{cas-dc}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{amsmath,amssymb,booktabs,graphicx,url,xcolor}
\usepackage[T2A,T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english,russian]{babel}
\shorthandoff{"}
\renewcommand{\rmdefault}{cmr}
\hypersetup{hidelinks}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\newcommand{\authA}{Мирсаид А. Юсупов}
\newcommand{\authB}{Максаджон К. Онаркулов}
\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}
\shorttitle{@SHORT@}
\shortauthors{М.А. Юсупов, М.К. Онаркулов}
\title[mode=title]{@TITLE@}
\author[1]{\authA}[orcid=0009-0001-1989-0088]
\cormark[1]
\ead{mirsaidbeky@gmail.com}
\affiliation[1]{organization={Кафедра прикладной математики и информатики, Ферганский государственный университет}, city={Фергана}, postcode={150100}, country={Узбекистан}}
\author[2]{\authB}[orcid=0000-0002-3650-4847]
\ead{maxmaqsad@gmail.com}
\affiliation[2]{organization={Физико-математический факультет, Ферганский государственный университет}, city={Фергана}, postcode={150100}, country={Узбекистан}}
\cortext[1]{Автор для корреспонденции.}
\begin{abstract}
@ABSTRACT@
\end{abstract}
\begin{highlights}[Основные положения]
@HIGHLIGHTS@
\end{highlights}
\begin{keywords}[Ключевые слова]
@KEYWORDS@
\end{keywords}
\maketitle
@BODY@
\bibliographystyle{elsarticle-num}
\bibliography{refs}
@BIOS@
\end{document}
"""
BIOS_EN = r"""\bio{}
\textbf{Mirsaid A. Yusupov} received the bachelor's and master's degrees from Fergana State University, Fergana, Uzbekistan, in 2021 and 2023. He is a lecturer with the Department of Applied Mathematics and Informatics, Fergana State University, and is working toward the Ph.D. degree. His research interests include computer vision, sign language recognition and deep learning.
\endbio
\bio{}
\textbf{Maksadjon K. Onarkulov} received the Ph.D. degree in physical and mathematical sciences. He is an Associate Professor and the Dean of the Faculty of Physics and Mathematics, Fergana State University, Fergana, Uzbekistan.
\endbio
"""
BIOS_RU = r"""\bio{}
\textbf{Мирсаид А. Юсупов} окончил бакалавриат и магистратуру Ферганского государственного университета (Фергана, Узбекистан) в 2021 и 2023 гг. Преподаватель кафедры прикладной математики и информатики Ферганского государственного университета, работает над диссертацией PhD. Научные интересы: компьютерное зрение, распознавание жестового языка, глубокое обучение.
\endbio
\bio{}
\textbf{Максаджон К. Онаркулов} — доктор философии (PhD) по физико-математическим наукам, доцент, декан физико-математического факультета Ферганского государственного университета (Фергана, Узбекистан).
\endbio
"""

def build_docx(front, docx_md):
    title = re.search(r'^title: "(.*)"$', front, re.M).group(1)
    abstract = block(docx_md, 'abstract'); keywords = block(docx_md, 'keywords'); highlights = block(docx_md, 'highlights')
    body = re.sub(r'::: (abstract|keywords|highlights)\n.*?\n:::\n', '', docx_md, flags=re.S)
    head = (f'# {title}\n\nМирсаид А. Юсупов ^1,\\*^, Максаджон К. Онаркулов ^2^\n\n^1^ Кафедра прикладной математики и информатики, Ферганский государственный университет, Фергана 150100, Узбекистан; mirsaidbeky@gmail.com\n\n^2^ Физико-математический факультет, Ферганский государственный университет, Фергана 150100, Узбекистан; maxmaqsad@gmail.com\n\n\\* Автор для корреспонденции.\n\n**Аннотация.** {abstract}\n\n**Ключевые слова:** {keywords}\n\n**Основные положения:**\n\n{highlights}\n\n' if RU else
            f'# {title}\n\nMirsaid A. Yusupov ^1,\\*^ and Maksadjon K. Onarkulov ^2^\n\n^1^ Department of Applied Mathematics and Informatics, Fergana State University, Fergana 150100, Uzbekistan; mirsaidbeky@gmail.com\n\n^2^ Faculty of Physics and Mathematics, Fergana State University, Fergana 150100, Uzbekistan; maxmaqsad@gmail.com\n\n\\* Corresponding author.\n\n**Abstract.** {abstract}\n\n**Keywords:** {keywords}\n\n**Highlights:**\n\n{highlights}\n\n')
    text = head + body + ('\n\n# Список литературы\n' if RU else '\n\n# References\n')
    if RU: text = re.sub(r'(\d) %', '\\1\u00a0%', text)
    (BUILD / 'article.md').write_text(text, encoding='utf-8')
    csl = Path('/home/claude/dissertatsiya/05-my-articles/2026-uzsl-rsl-transfer/csl/ieee.csl')
    ref = ROOT / 'tools' / 'reference.docx'
    cmd = ['pandoc', str(BUILD / 'article.md'), '-f', 'markdown', '-t', 'docx', '--citeproc', '--bibliography', str(ROOT / 'src' / 'refs_v2.json'), '--csl', str(csl), '-o', str(BUILD / 'article.docx'), '--resource-path', str(BUILD)]
    if ref.exists(): cmd += ['--reference-doc', str(ref)]
    subprocess.run(cmd, check=True); print('docx ok')

if __name__ == '__main__':
    front, tex_md, docx_md = assemble()
    build_docx(front, docx_md)
    build_tex(front, tex_md)
