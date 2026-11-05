#!/usr/bin/env python3
"""Все числа статьи 1 из выгрузки к2 (k2_k3/0001_article1_exports): одна точка истины.

python3 -I analyze.py <exports_dir> <out_dir>
Выход: numbers.json (все числа с источником), tables/*.md, figures/*.{pdf,png}.
"""
import csv, gzip, json, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

EXP = Path(sys.argv[1]); OUT = Path(sys.argv[2]); (OUT / 'tables').mkdir(parents=True, exist_ok=True); (OUT / 'figures').mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
NB = 2000          # бутстреп-повторов (как в статье 1)
NB_LOSO = 10000    # как в отчётах корпуса
NUM = {}           # numbers.json

# ---------------------------------------------------------------- загрузка
_records, _classes = {}, {}
def space_of(dataset):
    return {'slovo': 'slovo', 'bukva': 'bukva', 'pilot': 'pilot', 'uzsl': 'uzsl', 'IMO': 'uzsl', 'N01': 'uzsl'}[dataset]
def records(space):
    if space not in _records:
        df = pd.read_csv(EXP / 'records' / f'{space}.csv.gz')
        _records[space] = df.set_index('rec')
        _classes[space] = json.load(open(EXP / 'classes' / f'{space}.json'))
    return _records[space]
def classes(space):
    records(space); return _classes[space]

MANIFEST = pd.read_csv(EXP / 'manifest.csv', dtype=str).fillna('')
def runs(series, condition=None, dataset=None):
    m = MANIFEST[(MANIFEST.series == series)]
    if condition is not None: m = m[m.condition == condition]
    if dataset is not None: m = m[m.dataset == dataset]
    m = m[m.path != '']
    return m

def load_run(row):
    p = EXP / row.path / 'predictions.csv.gz'
    if not p.exists(): return None
    df = pd.read_csv(p)
    if len(df) == 0: return None
    ds = row.dataset
    if not ds:  # в manifest у части серий dataset пуст — выводим из имени условия/пути
        s = (row.condition + ' ' + row.path).lower()
        ds = 'bukva' if 'bukva' in s else 'pilot' if 'pilot' in s else 'uzsl' if 'uzsl' in s else 'slovo'
    sp = space_of(ds)
    rec = records(sp)
    df['signer'] = rec.loc[df.rec, 'signer'].values
    df['record_id'] = rec.loc[df.rec, 'record_id'].values
    df['top5'] = (df[['pred1', 'pred2', 'pred3', 'pred4', 'pred5']].values == df.true.values[:, None]).any(1).astype(int)
    df['correct'] = (df.pred1 == df.true).astype(int)
    df['fold'] = row.fold; df['seed'] = int(row.seed)
    return df

def load_condition(series, condition, dataset=None):
    """Все фолды и затравки одного условия в одном DataFrame."""
    parts = [load_run(r) for r in runs(series, condition, dataset).itertuples()]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else None

# ---------------------------------------------------------------- метрики
def per_class_prf(df):
    """macro precision / recall / F1 по классам, присутствующим среди истинных меток (среднее по затравкам)."""
    out = []
    for s, d in df.groupby('seed'):
        labels = np.unique(d.true)
        tp = d[d.correct == 1].groupby('true').size().reindex(labels, fill_value=0)
        fn = d.groupby('true').size().reindex(labels, fill_value=0) - tp
        fp = d[d.correct == 0].groupby('pred1').size().reindex(labels, fill_value=0)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        rec_ = tp / np.maximum(tp + fn, 1)
        f1 = np.where(prec + rec_ > 0, 2 * prec * rec_ / np.maximum(prec + rec_, 1e-12), 0.0)
        out.append((prec.mean(), rec_.mean(), f1.mean()))
    return tuple(100 * np.mean(out, axis=0))

def per_class_table(df, names):
    """Per-class support (per seed), precision, recall and F1 in %, averaged over seeds; for the supplementary tables."""
    labels = np.unique(df.true); acc = []
    for s, d in df.groupby('seed'):
        tp = d[d.correct == 1].groupby('true').size().reindex(labels, fill_value=0)
        sup = d.groupby('true').size().reindex(labels, fill_value=0)
        fp = d[d.correct == 0].groupby('pred1').size().reindex(labels, fill_value=0)
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0); rec_ = tp / np.maximum(sup, 1)
        f1 = np.where(prec + rec_ > 0, 2 * prec * rec_ / np.maximum(prec + rec_, 1e-12), 0.0)
        acc.append(np.stack([sup.values, prec, rec_, f1], 1))
    m = np.mean(acc, 0)
    return [dict(cls=str(names[l]), support=float(m[i, 0]), precision=100 * float(m[i, 1]), recall=100 * float(m[i, 2]), f1=100 * float(m[i, 3])) for i, l in enumerate(labels)]

def signer_table(df, col='correct'):
    """Таблица [затравка × исполнитель] сумм и счётчиков -> для бутстрепа по исполнителям."""
    g = df.groupby(['seed', 'signer'])[col].agg(['sum', 'count']).unstack('signer').fillna(0)
    return g['sum'].values, g['count'].values, list(g['sum'].columns)

def boot_pooled(sums, counts, nb=NB, rng=RNG):
    """Объединённая по записям точность, усреднённая по затравкам; ДИ бутстрепом по исполнителям."""
    S = sums.shape[1]
    point = (sums.sum(1) / counts.sum(1)).mean()
    idx = rng.integers(0, S, size=(nb, S))
    bs = (sums[:, idx].sum(2) / np.maximum(counts[:, idx].sum(2), 1)).mean(0)
    return 100 * point, 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)

def metrics(df, min_clips=10, nb=NB):
    sums, counts, signers = signer_table(df)
    t1, lo, hi = boot_pooled(sums, counts, nb)
    s5, c5, _ = signer_table(df, 'top5')
    t5 = (s5.sum(1) / c5.sum(1)).mean() * 100
    per_signer = (sums / np.maximum(counts, 1)).mean(0) * 100   # средняя по затравкам точность исполнителя
    macro = float(per_signer.mean())
    elig = counts.mean(0) >= min_clips
    worst = float(per_signer[elig].min()) if elig.any() else float('nan')
    p, r, f = per_class_prf(df)
    return dict(top1=t1, ci=[lo, hi], top5=t5, macro_signer=macro, worst_signer=worst,
                precision=p, recall=r, f1=f, n_test=int(counts.sum(1).mean()), n_signers=len(signers), n_seeds=int(sums.shape[0]))

def paired_diff(dfA, dfB, nb=NB, level=95, rng=RNG):
    """Парная разность A − B на общих записях (среднее по затравкам на запись), бутстреп по исполнителям."""
    a = dfA.groupby('record_id').agg(c=('correct', 'mean'), signer=('signer', 'first'))
    b = dfB.groupby('record_id').agg(c=('correct', 'mean'))
    j = a.join(b, how='inner', lsuffix='A', rsuffix='B')
    j['d'] = j.cA - j.cB
    g = j.groupby('signer').d.agg(['sum', 'count'])
    sums, counts = g['sum'].values, g['count'].values
    point = sums.sum() / counts.sum()
    idx = rng.integers(0, len(sums), size=(nb, len(sums)))
    bs = sums[idx].sum(1) / np.maximum(counts[idx].sum(1), 1)
    a_ = (100 - level) / 2
    return dict(delta=100 * point, ci=[100 * np.percentile(bs, a_), 100 * np.percentile(bs, 100 - a_)], level=level,
                n_records=int(len(j)), signers_positive=int((g['sum'] > 0).sum()), signers=int(len(g)))

# ---------------------------------------------------------------- 1. РЖЯ: главная таблица (протокол П)
def a1(dataset):
    if dataset == 'slovo':   # затравка 0 — M3, затравки 1–2 — M7 (как в выгрузке)
        return pd.concat([load_condition('M3', 'slovo_a_ema'), load_condition('M7', 'slovo_a_ema')], ignore_index=True)
    return load_condition('M7', 'bukva_a_ema')

RSL_ROWS = [  # (ключ, подпись, Slovo (series, condition), Bukva (series, condition))
    ('A1', 'MPE-A1', ('A1',), ('A1',)),
    ('ST-GCN released', 'ST-GCN, released code', ('M11G3', 'g3_slovo_stgcn_orig_mpkrec'), None),
    ('SL-GCN released', 'SL-GCN, released code', ('M11G2', 'g2_slovo_slgcn_orig_mpkrec'), None),
    ('ST-GCN released authors', "ST-GCN, released code and authors' recipe", ('M11', 'stgcn'), ('M11', 'stgcn_bukva')),
    ('SL-GCN released authors', "SL-GCN, released code and authors' recipe", ('M11', 'samslr'), ('M11', 'samslr_bukva')),
    ('A1 SL-GCN recipe', 'MPE-A1, SL-GCN recipe', ('M11G2', 'g2_slovo_a1_samslr'), None),
    ('ST-GCN reimpl', 'ST-GCN, our reimplementation', ('M8', 'm81_slovo_stgcn_mpkrec'), ('M8', 'm81_bukva_stgcn_mpkrec')),
    ('SL-GCN reimpl', 'SL-GCN, our reimplementation', ('M8', 'm81_slovo_slgcn_mpkrec'), ('M8', 'm81_bukva_slgcn_mpkrec')),
    ('KP Transformer', 'Keypoint Transformer', ('M8', 'm81_slovo_e1_transformer_e1rec'), ('M8', 'm81_bukva_e1_transformer_e1rec')),
    ('KP Transformer shared', 'Keypoint Transformer, shared recipe', ('M8', 'm81_slovo_e1_transformer_mpkrec'), None),
    ('2-layer Transformer', 'Two-layer Transformer', ('E1F', 'slovo_e16_transformer'), ('E1F', 'bukva_e16_transformer')),
    ('BiLSTM', 'BiLSTM with attention', ('E1F', 'slovo_e16_bilstm'), ('E1F', 'bukva_e16_bilstm')),
    ('MLP', 'MLP', ('E1F', 'slovo_e16_mlp'), ('E1F', 'bukva_e16_mlp')),
    ('A2', 'MPE-A2 (no graph)', ('M8', 'm85_slovo_a1_bez_grafa'), ('M8', 'm85_bukva_a1_bez_grafa')),
    ('A1 large', 'MPE large (D=256, N=6)', ('M8', 'm83_slovo_a1_big'), ('M8', 'm83_bukva_a1_big')),
    ('A0', 'MPE-A0 (no EMA)', ('M1', 'slovo_a0'), ('M1', 'bukva_a0')),
    ('A1 no tokens', 'MPE-A1 without part tokens', ('M8', 'm85_slovo_a1_bez_tokenov'), ('M8', 'm85_bukva_a1_bez_tokenov')),
    ('A1 no long edges', 'MPE-A1 without long-range edges', ('M8', 'm85_slovo_a1_dalnie'), ('M8', 'm85_bukva_a1_dalnie')),
    ('A1 64-frame window', 'MPE-A1, 64-frame window', ('M8', 'm82_slovo_a1_okno_e1'), ('M8', 'm82_bukva_a1_okno_e1')),
]
def rsl_tables():
    res = {}
    for ds in ('slovo', 'bukva'):
        base = a1(ds); res[ds] = {}
        for key, label, sl, bk in RSL_ROWS:
            spec = sl if ds == 'slovo' else bk
            if spec is None: continue
            df = base if spec == ('A1',) else load_condition(*spec)
            if df is None: continue
            m = metrics(df); m['label'] = label
            if spec != ('A1',):
                m['delta_vs_A1'] = paired_diff(df, base); m['delta_vs_A1_99'] = paired_diff(df, base, level=99)
            res[ds][key] = m
            print(f'{ds:6s} {key:28s} top1 {m["top1"]:.1f} [{m["ci"][0]:.1f}; {m["ci"][1]:.1f}] top5 {m["top5"]:.1f} F1 {m["f1"]:.1f} macroS {m["macro_signer"]:.1f} worst {m["worst_signer"]:.1f} n={m["n_test"]} seeds={m["n_seeds"]}')
    # разности между базовыми: оригинал − своя реализация; рецепт авторов − общий
    pairs = {'slovo': [('ST-GCN released', 'ST-GCN reimpl'), ('SL-GCN released', 'SL-GCN reimpl'), ('SL-GCN released authors', 'SL-GCN released'),
                       ('A1 SL-GCN recipe', 'A1'), ('A1 SL-GCN recipe', 'SL-GCN released authors'), ('KP Transformer', 'KP Transformer shared'),
                       ('ST-GCN released authors', 'ST-GCN reimpl'), ('ST-GCN released', 'ST-GCN released authors')],
             'bukva': [('SL-GCN released authors', 'SL-GCN reimpl'), ('ST-GCN released authors', 'ST-GCN reimpl')]}
    loaded = {}
    def get(ds, key):
        if (ds, key) not in loaded:
            spec = next(r for r in RSL_ROWS if r[0] == key); spec = spec[2] if ds == 'slovo' else spec[3]
            loaded[(ds, key)] = a1(ds) if spec == ('A1',) else load_condition(*spec)
        return loaded[(ds, key)]
    for ds, ps in pairs.items():
        for x, y in ps:
            d = paired_diff(get(ds, x), get(ds, y)); d99 = paired_diff(get(ds, x), get(ds, y), level=99)
            res[ds].setdefault('pairs', {})[f'{x} − {y}'] = {'95': d, '99': d99}
            print(f'  Δ {ds} {x} − {y}: {d["delta"]:+.2f} [{d["ci"][0]:+.2f}; {d["ci"][1]:+.2f}] (99%: [{d99["ci"][0]:+.2f}; {d99["ci"][1]:+.2f}])')
    NUM['rsl'] = res
    # путаницы Bukva (матрица) и Slovo (топ-15 пар) для A1
    for ds in ('slovo', 'bukva'):
        df = a1(ds); cls = classes(ds)
        wrong = df[df.correct == 0].groupby(['true', 'pred1']).size().sort_values(ascending=False)
        NUM['rsl'][ds]['top_confusions'] = [(cls[t], cls[p], int(n / df.seed.nunique() * 1000) / 1000) for (t, p), n in wrong.head(20).items()]
        if ds == 'bukva':
            n = len(cls); M = np.zeros((n, n))
            for (t, p), c in df.groupby(['true', 'pred1']).size().items(): M[t, p] += c
            M = M / M.sum(1, keepdims=True)
            NUM['rsl']['bukva']['confusion_matrix'] = {'classes': cls, 'row_normalized': M.round(4).tolist()}
            NUM['rsl']['bukva']['per_class'] = per_class_table(df, cls)
            fig_confusion(M, cls, 'fig_confusion_bukva', 'Bukva (RSL fingerspelling), MPE-A1 under SD-5')

# ---------------------------------------------------------------- 2. Абляции (серии M3/M4/M5/M7) — Slovo, A-варианты без EMA → сравнение с A0
ABL = {  # condition: подпись
    'M3/slovo_a_bez_augm': 'no augmentation', 'M3/slovo_a_bez_conv': 'no temporal convolution', 'M3/slovo_a_bez_grafa': 'no graph (A0)',
    'M3/slovo_a_bez_tokenov': 'no part tokens (A0)', 'M3/slovo_a_bez_vspom': 'no auxiliary heads', 'M3/slovo_a_dalnie': 'no long-range edges (A0)',
    'M3/slovo_a_dvizh': 'motion features', 'M3/slovo_a_kosti': 'no bone stream', 'M3/slovo_a_maskbn': 'masked batch norm', 'M3/slovo_a_obshchie_kisti': 'shared hand weights',
    'M3/slovo_a_pozdnee': 'late fusion', 'M3/slovo_a_pul': 'mean pooling', 'M3/slovo_a_transf': 'Transformer instead of Conformer', 'M3/slovo_a_uzly': 'node variant',
    'M4/slovo_a_bez_nrm': 'no normalization', 'M4/slovo_a_dominant': 'dominant-hand mirroring', 'M4/slovo_a_kist': 'hands only', 'M4/slovo_a_lico': 'no face points',
    'M4/slovo_a_maska': 'no mask channel', 'M4/slovo_a_nrm_kadr': 'per-frame normalization', 'M4/slovo_a_telo': 'no body points', 'M4/slovo_a_ugly': 'joint angles', 'M4/slovo_a_z': 'with depth z',
    'M5/slovo_d128': 'D = 128', 'M5/slovo_d256': 'D = 256', 'M5/slovo_n2': 'N = 2', 'M5/slovo_n4': 'N = 4', 'M5/slovo_t48': 'T = 48',
    'M7/slovo_d256_n6': 'D = 256, N = 6 (no EMA)', 'M7/slovo_d384_n8': 'D = 384, N = 8 (no EMA)',
}
def ablations():
    base = load_condition('M1', 'slovo_a0'); res = {}
    for cond, label in ABL.items():
        s, c = cond.split('/'); df = load_condition(s, c)
        if df is None: continue
        m = metrics(df); m['label'] = label; m['delta_vs_A0'] = paired_diff(df, base); res[cond] = m
        print(f'abl {label:36s} {m["top1"]:.1f}  Δ {m["delta_vs_A0"]["delta"]:+.2f} [{m["delta_vs_A0"]["ci"][0]:+.2f}; {m["delta_vs_A0"]["ci"][1]:+.2f}] seeds={m["n_seeds"]}')
    NUM['ablations_A0'] = res

# ---------------------------------------------------------------- 3. Официальный Slovo
def official():
    specs = {'A1': ('M9', 'slovo_official_a1'), 'A0': ('M1', 'slovo_official_a0'), 'A2': ('M11', 'slovo_official_a2'), 'A1 large': ('M11', 'slovo_official_a1_big'),
             'KP Transformer (M1)': ('M1', 'slovo_official_e1_transformer'), 'KP Transformer (own recipe)': ('M7', 'slovo_official_e1tr_e1rec'),
             'MViTv2-S 32/2 P0': ('M11', 'video_mvit32-2_P0'), 'MViTv2-S 32/2 R1': ('M11', 'video_mvit32-2_R1'), 'MViTv2-S 32/2 R2': ('M11', 'video_mvit32-2_R2'),
             'MViTv2-S 48/2 P0': ('M11', 'video_mvit48-2_P0'), 'MViTv2-S 48/2 R1': ('M11', 'video_mvit48-2_R1'), 'ResNet-i3d P0': ('M11', 'video_resnet32-2_P0'), 'ResNet-i3d R1': ('M11', 'video_resnet32-2_R1')}
    res = {}; base = load_condition(*specs['A1'])
    for k, sp in specs.items():
        df = load_condition(*sp)
        if df is None: continue
        m = metrics(df)
        if k != 'A1': m['delta_vs_A1'] = paired_diff(df, base)
        res[k] = m
        print(f'official {k:28s} {m["top1"]:.2f} [{m["ci"][0]:.1f}; {m["ci"][1]:.1f}] top5 {m["top5"]:.1f} n={m["n_test"]}')
    NUM['official_slovo'] = res

# ---------------------------------------------------------------- 4. Пилот УзЖЯ (бюджеты заранее, M9.7) + SL-GCN (M11P)
def pilot():
    res = {}
    conds = {}
    for r in runs('M9_7').itertuples(): conds.setdefault(r.condition, []).append(r)
    for r in runs('M11P').itertuples():
        if r.path: conds.setdefault(r.condition, []).append(r)
    dfs = {}
    for cond, rows in conds.items():
        parts = [load_run(r) for r in rows]; parts = [p for p in parts if p is not None]
        if not parts: continue
        df = pd.concat(parts, ignore_index=True); dfs[cond] = df
        pooled = df.groupby('seed').correct.mean().mean() * 100
        per_part = df.groupby(['signer', 'seed']).correct.mean().groupby('signer').mean() * 100
        res[cond] = dict(pooled=pooled, per_participant=per_part.round(2).to_dict(), top5=df.groupby('seed').top5.mean().mean() * 100,
                         f1=per_class_prf(df)[2], n=int(len(df) / df.seed.nunique()))
    # знак разности во всех шести парах (фолд × затравка) против обучения с нуля той же ступени
    def sign_rule(a, b):
        A = dfs[a].groupby(['fold', 'seed']).correct.mean(); B = dfs[b].groupby(['fold', 'seed']).correct.mean()
        d = (A - B).dropna(); return dict(mean_of_pairs=float(d.mean() * 100), positive=int((d > 0).sum()), negative=int((d < 0).sum()), ties=int((d == 0).sum()), pairs=int(len(d)), all_same_sign=bool((d > 0).all() or (d < 0).all()))
    for step in ('k1', 'k2', 'all'):
        for cond in list(dfs):
            if cond.startswith(f'pilot_{step}_') and 'scratch' not in cond and cond.endswith('_a1'):
                res[cond]['vs_scratch'] = sign_rule(cond, f'pilot_{step}_scratch_a1')
    for step in ('k1', 'k2', 'all'):
        for scheme in ('ft', 'probe', 'scratch'):
            a = f'pilot_{step}_sb_{scheme}_a1' if scheme != 'scratch' else f'pilot_{step}_scratch_a1'
            b = f'pilot_{step}_sb_{scheme}_slgcn_orig' if scheme != 'scratch' else f'pilot_{step}_scratch_slgcn_orig'
            if a in dfs and b in dfs: res.setdefault('A1_vs_SLGCN', {})[f'{step} {scheme}'] = dict(A1=res[a]['pooled'], SLGCN=res[b]['pooled'], **sign_rule(a, b))
            a2 = f'pilot_{step}_sb_{scheme}_a2'
            if a2 in dfs and a in dfs: res.setdefault('A2_vs_A1', {})[f'{step} {scheme}'] = dict(A2=res[a2]['pooled'], A1=res[a]['pooled'], **sign_rule(a2, a))
    # путаницы по буквам пилота: all, sb ft a1
    df = dfs['pilot_all_sb_ft_a1']; cls = classes('pilot'); n = len(cls); M = np.zeros((n, n))
    for (t, p), c in df.groupby(['true', 'pred1']).size().items(): M[t, p] += c
    M = M / np.maximum(M.sum(1, keepdims=True), 1)
    # верхняя граница точности: доля тестовых записей, чей класс есть в обучении фолда (metrics.json → seen_classes)
    bound = {}
    for r in runs('M9_7', 'pilot_all_sb_ft_a1').itertuples():
        mj = EXP / r.path / 'metrics.json'
        if mj.exists():
            t = json.load(open(mj))['test']['pilot']; bound[r.fold] = 100 * t['seen_classes']['n_clips'] / t['n_clips']
    res['upper_bound_seen_classes'] = bound
    res['confusion_all_sb_ft_a1'] = {'classes': cls, 'row_normalized': M.round(4).tolist()}
    res['per_class_all_sb_ft_a1'] = per_class_table(df, cls)
    fig_confusion(M, cls, 'fig_confusion_pilot', 'UzSL pilot (fingerspelling and numerals), MPE-A1 pretrained on Slovo+Bukva, fine-tuned on all recordings')
    res['early_val_vs_test'] = 'источник — m7_tables.md §5 (A0, валидация на отложенных записях обучающих участников), данные в figures/fig5_8.py; в выгрузке M9 — M9.2 с артефактом EMA, не используется'
    for k, v in res.items():
        if isinstance(v, dict) and 'pooled' in v: print(f'pilot {k:34s} {v["pooled"]:.1f}  ({v["per_participant"]})  ' + (str(v.get("vs_scratch")) if v.get('vs_scratch') else ''))
    NUM['pilot'] = res

# ---------------------------------------------------------------- 5. Корпус УзЖЯ: LOSO
def is_letter(name): return name.startswith('harf:')
def is_numeral(name): return name.startswith('raqam') or re.fullmatch(r'\d+', name) is not None
def loso():
    cls = classes('uzsl'); res = {}
    conds = {'step1': ('UZ_step1', 'loso'), 'A': ('UZ_step1b', 'A'), 'B': ('UZ_step1b', 'B'), 'B1': ('UZ_step1b', 'B1'), 'SLGCN': ('UZ_M12_slgcn', 'M12_slgcn_loso'), 'modelV': ('UZ_modelV', 'loso')}
    dfs = {k: load_condition(*v) for k, v in conds.items()}
    names = np.array(cls)
    def macro10(df, col='correct'):
        d = df[df.fold != 'other']
        ps = d.groupby(['signer', 'seed'])[col].mean().groupby('signer').mean()   # по исполнителям, среднее затравок
        vals = ps.values; idx = RNG.integers(0, len(vals), size=(NB_LOSO, len(vals)))
        bs = vals[idx].mean(1)
        return dict(macro=100 * vals.mean(), ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)], per_signer=(100 * ps).round(2).to_dict(),
                    micro=100 * d.groupby('seed')[col].mean().mean())
    def paired10(dfA, dfB, level=95):
        a = dfA[dfA.fold != 'other'].groupby(['signer', 'seed']).correct.mean().groupby('signer').mean()
        b = dfB[dfB.fold != 'other'].groupby(['signer', 'seed']).correct.mean().groupby('signer').mean()
        d = (a - b).dropna().values; idx = RNG.integers(0, len(d), size=(NB_LOSO, len(d))); bs = d[idx].mean(1); q = (100 - level) / 2
        return dict(delta=100 * d.mean(), ci=[100 * np.percentile(bs, q), 100 * np.percentile(bs, 100 - q)], positive=int((d > 0).sum()), signers=int(len(d)), level=level)
    for k, df in dfs.items():
        if df is None: continue
        m1 = macro10(df); m5 = macro10(df, 'top5'); p, r, f = per_class_prf(df[df.fold != 'other'])
        # слова / дактиль
        letter = df.true.map(lambda i: is_letter(names[i])); num = df.true.map(lambda i: is_numeral(names[i]))
        words = df[~letter & ~num]; fs = df[letter | num]
        def group_macro(d, min_n=10):
            d = d[d.fold != 'other']; cnt = d.groupby('signer').size() / d.seed.nunique(); keep = cnt[cnt >= min_n].index
            ps = d[d.signer.isin(keep)].groupby(['signer', 'seed']).correct.mean().groupby('signer').mean()
            return dict(macro=100 * ps.mean(), n_signers=int(len(ps)), per_signer=(100 * ps).round(1).to_dict())
        res[k] = dict(top1_macro=m1['macro'], ci=m1['ci'], top1_micro=m1['micro'], top5_macro=m5['macro'], top5_ci=m5['ci'], top5_micro=m5['micro'],
                      per_signer=m1['per_signer'], precision=p, recall=r, f1=f, words=group_macro(words), fingerspelling=group_macro(fs),
                      n_test=int(len(df[df.fold != 'other']) / df.seed.nunique()), n_classes_head=int(len(json.load(open(EXP / conds[k][0] / conds[k][1] / 'labels.json'))['uzsl'])) if (EXP / conds[k][0] / conds[k][1] / 'labels.json').exists() else None)
        print(f'LOSO {k:7s} top1 {m1["macro"]:.1f} [{m1["ci"][0]:.1f}; {m1["ci"][1]:.1f}] micro {m1["micro"]:.1f} top5 {m5["macro"]:.1f} P {p:.1f} R {r:.1f} F1 {f:.1f} words {res[k]["words"]["macro"]:.1f} ({res[k]["words"]["n_signers"]}) fs {res[k]["fingerspelling"]["macro"]:.1f} ({res[k]["fingerspelling"]["n_signers"]})')
    pairs = [('step1', 'B1'), ('A', 'B'), ('A', 'step1'), ('step1', 'SLGCN'), ('modelV', 'step1'), ('SLGCN', 'B1')]
    for x, y in pairs:
        if dfs.get(x) is None or dfs.get(y) is None: continue
        d = paired10(dfs[x], dfs[y]); d99 = paired10(dfs[x], dfs[y], 99); res.setdefault('pairs', {})[f'{x} − {y}'] = {'95': d, '99': d99}
        print(f'  Δ LOSO {x} − {y}: {d["delta"]:+.2f} [{d["ci"][0]:+.2f}; {d["ci"][1]:+.2f}] 99%: [{d99["ci"][0]:+.2f}; {d99["ci"][1]:+.2f}] +{d["positive"]}/{d["signers"]}')
    # перенос по группам: слова / дактиль (шаг 1 − Б1)
    for grp, mask in (('words', lambda d: ~d.true.map(lambda i: is_letter(names[i]) or is_numeral(names[i]))), ('fingerspelling', lambda d: d.true.map(lambda i: is_letter(names[i]) or is_numeral(names[i])))):
        a = dfs['step1']; b = dfs['B1']; a = a[mask(a)]; b = b[mask(b)]
        cnt = a[a.fold != 'other'].groupby('signer').size() / a.seed.nunique(); keep = cnt[cnt >= 10].index   # исполнители с >= 10 записями группы
        d = paired10(a[a.signer.isin(keep)], b[b.signer.isin(keep)]); res['pairs'][f'step1 − B1 ({grp})'] = {'95': d}
        print(f'  Δ LOSO step1 − B1 {grp}: {d["delta"]:+.2f} [{d["ci"][0]:+.2f}; {d["ci"][1]:+.2f}] +{d["positive"]}/{d["signers"]}')
    # путаницы: слова топ-15, буквы матрица (латиница)
    df = dfs['step1']; wrong = df[df.correct == 0].groupby(['true', 'pred1']).size().sort_values(ascending=False)
    res['top_confusions_step1'] = [(names[t], names[p], round(n / df.seed.nunique(), 2)) for (t, p), n in wrong.head(20).items()]
    res['per_class_step1'] = per_class_table(df[df.fold != 'other'], names)
    lat = [i for i, c in enumerate(names) if is_letter(c) and re.fullmatch(r"harf:[a-z']+", c)]
    sub = df[df.true.isin(lat)]
    if len(sub):
        M = np.zeros((len(lat), len(lat))); pos = {c: j for j, c in enumerate(lat)}; other = np.zeros(len(lat))
        for (t, p), c in sub.groupby(['true', 'pred1']).size().items():
            if p in pos: M[pos[t], pos[p]] += c
            else: other[pos[t]] += c
        tot = M.sum(1) + other; M = M / np.maximum(tot[:, None], 1)
        labels = [names[i][5:].upper() for i in lat]
        res['confusion_letters_step1'] = {'classes': labels, 'row_normalized': M.round(4).tolist(), 'share_predicted_non_letter': (other / np.maximum(tot, 1)).round(4).tolist()}
        fig_confusion(M, labels, 'fig_confusion_uzsl_letters', 'UzSL corpus, Latin letters (LOSO, MPE-A1 pretrained on Slovo+Bukva)')
    # рисунок: точность по исполнителям
    fig_loso_signers(res)
    NUM['loso'] = res

# ---------------------------------------------------------------- 6. Внешние источники
def external():
    res = {}
    for src in ('N01', 'IMO'):
        for model in ('step1', 'modelV', 'A', 'B'):
            df = load_condition(f'EXT_{src}', model)
            if df is None: continue
            head = set(json.load(open(EXP / f'EXT_{src}' / model / 'labels.json'))['uzsl'])
            names = classes('uzsl')
            inhead = df[df.true.map(lambda i: names[i] in head)]
            def boot_records(d, col):
                v = d[col].values; idx = RNG.integers(0, len(v), size=(NB_LOSO, len(v))); bs = v[idx].mean(1)
                return [100 * v.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]
            # для imo-ishora — бутстреп по словам (классам)
            if src == 'IMO':
                byw = inhead.groupby('true').correct.mean().values; idx = RNG.integers(0, len(byw), size=(NB_LOSO, len(byw))); bs = byw[idx].mean(1)
                t1 = [100 * inhead.correct.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]
            else: t1 = boot_records(inhead, 'correct')
            res[f'{src}/{model}'] = dict(top1=t1[0], ci=t1[1:], top5=100 * inhead.top5.mean(), n_in_head=int(len(inhead)), n_total=int(len(df)), n_words=int(inhead.true.nunique()), head_size=len(head), f1=per_class_prf(inhead)[2])
            print(f'EXT {src} {model:6s} head {len(head):5d} in-head {len(inhead):4d}/{len(df)} words {inhead.true.nunique()} top1 {t1[0]:.1f} [{t1[1]:.1f}; {t1[2]:.1f}] top5 {res[f"{src}/{model}"]["top5"]:.1f}')
        # парная разность А − Б на полной голове (по записям / по словам)
        A = load_condition(f'EXT_{src}', 'A'); B = load_condition(f'EXT_{src}', 'B')
        if A is not None and B is not None:
            headAB = set(json.load(open(EXP / f'EXT_{src}' / 'A' / 'labels.json'))['uzsl']); names = classes('uzsl')
            j = A.set_index('rec').correct.to_frame('a').join(B.set_index('rec').correct.to_frame('b'), how='inner'); j['true'] = A.set_index('rec').true
            j = j[j.true.map(lambda i: names[i] in headAB)]
            if src == 'IMO':
                d = (j.groupby('true').a.mean() - j.groupby('true').b.mean()).values
            else: d = (j.a - j.b).values
            idx = RNG.integers(0, len(d), size=(NB_LOSO, len(d))); bs = d[idx].mean(1)
            res[f'{src}/A − B'] = dict(delta=100 * d.mean(), ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)], unit='words' if src == 'IMO' else 'records')
            print(f'  EXT {src} A − B: {100*d.mean():+.1f} [{100*np.percentile(bs,2.5):+.1f}; {100*np.percentile(bs,97.5):+.1f}]')
        V = load_condition(f'EXT_{src}', 'modelV'); S = load_condition(f'EXT_{src}', 'step1')
        if V is not None and S is not None:
            head = set(json.load(open(EXP / f'EXT_{src}' / 'step1' / 'labels.json'))['uzsl']); names = classes('uzsl')
            j = V.set_index('rec').correct.to_frame('v').join(S.set_index('rec').correct.to_frame('s'), how='inner'); j['true'] = V.set_index('rec').true
            j = j[j.true.map(lambda i: names[i] in head)]
            d = (j.groupby('true').v.mean() - j.groupby('true').s.mean()).values if src == 'IMO' else (j.v - j.s).values
            idx = RNG.integers(0, len(d), size=(NB_LOSO, len(d))); bs = d[idx].mean(1)
            res[f'{src}/modelV − step1'] = dict(delta=100 * d.mean(), ci=[100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)])
            print(f'  EXT {src} V − step1: {100*d.mean():+.1f} [{100*np.percentile(bs,2.5):+.1f}; {100*np.percentile(bs,97.5):+.1f}]')
    NUM['external'] = res

# ---------------------------------------------------------------- 7. Устойчивость (M9.4) и немногие примеры (M6)
def robustness():
    res = {}
    for ds in ('slovo', 'bukva'):
        for p in ('0', '0.25', '0.5'):
            for s in ('0', '0.01', '0.02'):
                a1_ = load_condition('M9_4', f'A1_{ds}_p{p}_s{s}'); a2_ = load_condition('M9_4', f'A2_{ds}_p{p}_s{s}')
                if a1_ is None or a2_ is None: continue
                m1 = metrics(a1_); m2 = metrics(a2_); d = paired_diff(a2_, a1_)
                res[f'{ds} p={p} σ={s}'] = dict(A1=m1['top1'], A2=m2['top1'], delta_A2_A1=d)
                print(f'robust {ds} p={p} σ={s}: A1 {m1["top1"]:.1f} A2 {m2["top1"]:.1f} Δ {d["delta"]:+.1f} [{d["ci"][0]:+.1f}; {d["ci"][1]:+.1f}]')
    NUM['robustness'] = res
def fewshot():
    res = {}
    scratch_all = load_condition('M1', 'bukva_a0')
    for k in ('k1', 'k3', 'k10', 'all'):
        for scheme in ('scratch', 'slovo_probe', 'slovo_ft'):
            if k == 'all' and scheme == 'scratch': df = scratch_all
            else:
                df = load_condition('M6', f'bukva_{k}_{scheme}')
            if df is None: continue
            m = metrics(df); res[f'{k} {scheme}'] = dict(top1=m['top1'], ci=m['ci'])
        if f'{k} scratch' in res:
            for scheme in ('slovo_probe', 'slovo_ft'):
                a = load_condition('M6', f'bukva_{k}_{scheme}'); b = scratch_all if k == 'all' else load_condition('M6', f'bukva_{k}_scratch')
                if a is not None and b is not None: res[f'{k} {scheme} − scratch'] = paired_diff(a, b)
    for k, v in res.items(): print('fewshot', k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()} if 'delta' not in v else f'{v["delta"]:+.1f} [{v["ci"][0]:+.1f}; {v["ci"][1]:+.1f}]')
    NUM['fewshot_bukva'] = res

# ---------------------------------------------------------------- 8. Проверка меток, стоимость, QC
def label_review():
    df = pd.read_csv(EXP / 'docs' / 'label_review_pairs.csv')
    ref = df[df.has_reference_v1 == 1]
    res = dict(n_total=int(len(df)), n_with_reference=int(len(ref)), same=int((ref.verdict == 'same').sum()), different=int((ref.verdict == 'different').sum()), unsure=int((ref.verdict == 'unsure').sum()))
    v = (ref.verdict == 'same').values; sig = ref.signer.values; uniq = np.unique(sig)
    idx = RNG.integers(0, len(uniq), size=(NB_LOSO, len(uniq)))
    bs = [np.mean(np.concatenate([v[sig == uniq[i]] for i in row])) for row in idx]
    res['accuracy'] = 100 * v.mean(); res['ci'] = [100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]
    nr = df[df.has_reference_v1 == 0]; res['no_reference'] = dict(n=int(len(nr)), same=int((nr.verdict == 'same').sum()), different=int((nr.verdict == 'different').sum()), unsure=int((nr.verdict == 'unsure').sum()))
    res['pilot_pairs'] = dict(n=int(ref.signer.isin(['S10', 'S11', 'S12']).sum()), same=int(((ref.verdict == 'same') & ref.signer.isin(['S10', 'S11', 'S12'])).sum()))
    res['lesson_pairs'] = dict(n=int((~ref.signer.isin(['S10', 'S11', 'S12'])).sum()), same=int(((ref.verdict == 'same') & ~ref.signer.isin(['S10', 'S11', 'S12'])).sum()))
    res['by_signer'] = df.groupby('signer').verdict.value_counts().unstack(fill_value=0).to_dict('index')
    res['different_signers'] = df[df.verdict == 'different'].signer.tolist(); res['unsure_signers'] = df[df.verdict == 'unsure'].signer.tolist()
    print('labels', res['accuracy'], res['ci'], res['same'], res['different'], res['no_reference'], res['pilot_pairs'], res['lesson_pairs'])
    NUM['label_review'] = res
def cost():
    res = {}
    m8 = json.load(open(EXP / 'cost' / 'm8_cost.json'))['models']; res['m8'] = {k: dict(params_inference_m=v.get('params_inference_m'), gflops=v['cost']['gflops_per_window'], cpu_ms=v['cost']['cpu_latency_ms_median'], T=v['cost']['T']) for k, v in m8.items()}
    for f in ('m11_cost', 'm11_g2_cost', 'm11_g3_cost'):
        res[f] = json.load(open(EXP / 'cost' / f'{f}.json'))
    res['tflite'] = pd.read_csv(EXP / 'cost' / 'm11b_tflite.csv').to_dict('records')
    res['onnx_sizes'] = pd.read_csv(EXP / 'cost' / 'onnx_sizes.csv').to_dict('records')
    if (EXP / 'cost' / 'onnx_runtime.csv').exists(): res['onnx_runtime'] = pd.read_csv(EXP / 'cost' / 'onnx_runtime.csv').to_dict('records')
    NUM['cost'] = res
def qc():
    res = {}
    for f in ('slovo_qc_summary', 'bukva_qc_summary', 'uzsign_pilot_qc_summary', 'imo_ishora_qc_summary'):
        res[f] = json.load(open(EXP / 'qc' / f'{f}.json'))
    cp = pd.read_csv(EXP / 'qc' / 'uzsl_corpus_present.csv.gz')
    res['uzsl_corpus_present_columns'] = list(cp.columns); res['uzsl_corpus_present_summary'] = cp.describe().round(4).to_dict()
    NUM['qc'] = res

# ---------------------------------------------------------------- рисунки
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
def fig_confusion(M, labels, name, title):
    n = len(labels); fig, ax = plt.subplots(figsize=(min(7.0, 0.19 * n + 1.5), min(7.0, 0.19 * n + 1.5)))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title, fontsize=7)
    if n <= 45:
        for i in range(n):
            for j in range(n):
                if M[i, j] >= 0.05 and i != j: ax.text(j, i, f'{M[i, j]*100:.0f}', ha='center', va='center', fontsize=4.5, color='black')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label='Row share')
    fig.tight_layout(); fig.savefig(OUT / 'figures' / f'{name}.pdf'); fig.savefig(OUT / 'figures' / f'{name}.png', dpi=300); plt.close(fig)
def fig_loso_signers(res):
    keys = [k for k in ('step1', 'B1', 'SLGCN') if k in res]
    signers = sorted(res['step1']['per_signer'], key=lambda s: -res['step1']['per_signer'][s])
    fig, ax = plt.subplots(figsize=(7.0, 2.4)); w = 0.8 / len(keys)
    lab = {'step1': 'MPE-A1, pretrained on Slovo+Bukva', 'B1': 'MPE-A1, from scratch', 'SLGCN': 'SL-GCN (released), pretrained on Slovo+Bukva'}
    col = {'step1': '#1f77b4', 'B1': '#9ecae1', 'SLGCN': '#7f7f7f'}
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(signers)) + (i - (len(keys) - 1) / 2) * w, [res[k]['per_signer'][s] for s in signers], w, label=lab[k], color=col[k])
    ax.set_xticks(range(len(signers))); ax.set_xticklabels(signers); ax.set_ylabel('Top-1 accuracy, %'); ax.set_xlabel('Held-out signer (LOSO fold)')
    ax.legend(frameon=False, fontsize=6, loc='upper right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / 'figures' / 'fig_loso_signers.pdf'); fig.savefig(OUT / 'figures' / 'fig_loso_signers.png', dpi=300); plt.close(fig)

def main():
    rsl_tables(); ablations(); official(); pilot(); loso(); external(); robustness(); fewshot(); label_review(); cost(); qc()
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    json.dump(conv(NUM), open(OUT / 'numbers.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('written', OUT / 'numbers.json')

if __name__ == '__main__':
    main()
