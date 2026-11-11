#!/usr/bin/env python3
"""Supplementary material (EN) for article 1 — generated from out/numbers.json.
python3 -I tools/supp.py  →  build/supp/supp.pdf (+ per-class CSV files)."""
import json, subprocess, sys, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = json.load(open(ROOT / 'out' / 'numbers.json', encoding='utf-8'))
BUILD = ROOT / 'build' / 'supp'; BUILD.mkdir(parents=True, exist_ok=True)

def esc(s):
    t = str(s).replace('\\', '/').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#').replace("'", r"\textquotesingle{}")
    return f'\\ruT{{{t}}}' if any('\u0400' <= ch <= '\u04ff' for ch in t) else t
def f(x, d=1): return f'{x:.{d}f}'
def sg(x, d=1): return ('+' if x >= 0 else '$-$') + f(abs(x), d)
def ci(v, d=1): return f'[{f(v[0], d)}; {f(v[1], d)}]'
def dci(dd, d=1): return f'{sg(dd["delta"], d)} [{sg(dd["ci"][0], d)}; {sg(dd["ci"][1], d)}]'
def rows_tex(rows): return ' \\\\\n'.join(' & '.join(r) for r in rows)
def table(label, caption, head, rows, align, size=r'\footnotesize', long=False):
    h = ' & '.join(f'\\textbf{{{x}}}' for x in head)
    if long:
        return (f"{size}\n\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}\\label{{tab:{label}}}\\\\\n\\toprule\n{h} \\\\\n\\midrule\n\\endfirsthead\n"
                f"\\caption[]{{(continued)}}\\\\\n\\toprule\n{h} \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{rows_tex(rows)} \\\\\n\\end{{longtable}}\n\\normalsize\n")
    return (f"\\begin{{table}}[H]\n\\caption{{{caption}}}\\label{{tab:{label}}}\n\\centering\n{size}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n{h} \\\\\n\\midrule\n"
            f"{rows_tex(rows)} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")

out = []
# ---------------------------------------------------------------- S1 ablation sweep
A = N['ablations_A0']; a0 = N['rsl']['slovo']['A0']
groups = [('Input and normalisation', ['M4/slovo_a_bez_nrm', 'M4/slovo_a_nrm_kadr', 'M4/slovo_a_kist', 'M4/slovo_a_telo', 'M4/slovo_a_lico', 'M4/slovo_a_maska', 'M4/slovo_a_z', 'M4/slovo_a_ugly', 'M3/slovo_a_dvizh', 'M4/slovo_a_dominant']),
          ('Spatial encoder', ['M3/slovo_a_bez_grafa', 'M3/slovo_a_dalnie', 'M3/slovo_a_bez_tokenov', 'M3/slovo_a_kosti', 'M3/slovo_a_uzly', 'M3/slovo_a_obshchie_kisti', 'M3/slovo_a_maskbn', 'M3/slovo_a_pozdnee']),
          ('Temporal encoder and pooling', ['M3/slovo_a_bez_conv', 'M3/slovo_a_transf', 'M3/slovo_a_pul']),
          ('Training', ['M3/slovo_a_bez_augm', 'M3/slovo_a_bez_vspom']),
          ('Capacity and window', ['M5/slovo_d128', 'M5/slovo_d256', 'M5/slovo_n2', 'M5/slovo_n4', 'M5/slovo_t48', 'M7/slovo_d256_n6', 'M7/slovo_d384_n8'])]
LAB = {'no normalization': 'No normalisation (raw coordinates)', 'per-frame normalization': 'Per-frame instead of per-clip normalisation', 'hands only': 'Hands only (no body, no face points)', 'no body points': 'No body points', 'no face points': 'No face points',
       'no mask channel': 'No mask channel', 'with depth z': 'Depth $z$ added', 'joint angles': 'Joint angles added', 'motion features': 'Motion (frame-difference) features added', 'dominant-hand mirroring': 'Dominant-hand mirroring',
       'no graph (A0)': 'No graph: per-part MLP (as in MPE-A2)', 'no long-range edges (A0)': 'No long-range edges', 'no part tokens (A0)': 'No part tokens: one token per frame', 'no bone stream': 'No bone stream', 'node variant': 'Alternative set of hand nodes',
       'shared hand weights': 'Shared weights for the two hands', 'masked batch norm': 'Mask-aware batch normalisation', 'late fusion': 'Late fusion of joint and bone streams', 'no temporal convolution': 'No depthwise temporal convolution',
       'Transformer instead of Conformer': 'Transformer layers in place of Conformer layers', 'mean pooling': 'Mean pooling in place of attention pooling', 'no augmentation': 'No augmentation', 'no auxiliary heads': 'No auxiliary heads',
       'D = 128': '$D = 128$', 'D = 256': '$D = 256$', 'N = 2': '$N = 2$', 'N = 4': '$N = 4$', 'T = 48': '$T = 48$', 'D = 256, N = 6 (no EMA)': '$D = 256$, $N = 6$', 'D = 384, N = 8 (no EMA)': '$D = 384$, $N = 8$'}
rows = [[r'\multicolumn{4}{l}{\textit{Reference: MPE-A0 (' + f(a0['top1']) + ' ' + ci(a0['ci']) + ', three seeds)}}']]
for g, keys in groups:
    rows.append([f'\\multicolumn{{4}}{{l}}{{\\textit{{{g}}}}}'])
    for k in keys:
        v = A[k]; d = v['delta_vs_A0']
        rows.append([LAB.get(v['label'], v['label']), f'{f(v["top1"])} {ci(v["ci"])}', dci(d), f'{d["signers_positive"]} / {d["signers"]}'])
out.append(r'\section{Full ablation sweep on Slovo (SD-5)}' + '\n' +
           'Table~\\ref{tab:s_abl} lists every input, architecture, training and capacity variant trained on Slovo under SD-5 with one seed and compared with MPE-A0 (no weight averaging) by the paired difference over test signers; the five main ablations of MPE-A1 with three seeds are in Fig.~5 of the article. The last column gives the number of test signers (of 188) for which the variant is more accurate than MPE-A0.\n' +
           table('s_abl', 'Variants of MPE-A0 on Slovo under SD-5 (one seed, five folds): top-1 accuracy [95\\% CI over test signers], paired difference $\\Delta$ to MPE-A0 [95\\% CI] and the number of test signers with a positive difference.',
                 ['Variant', 'Top-1 [95\\% CI]', '$\\Delta$ vs MPE-A0 [95\\% CI]', 'Signers $+$'], rows, 'llll'))

# ---------------------------------------------------------------- S2 LOSO per signer
L = N['loso']; EVAL_N = {'S01': 382, 'S02': 1095, 'S03': 590, 'S04': 125, 'S05': 183, 'S07': 648, 'S08': 71, 'S09': 74, 'S11': 118, 'S12': 118}
signers = list(L['step1']['per_signer'].keys())
conds = [('step1', 'MPE-A1, pretrained (Slovo + Bukva), 682-class head'), ('B1', 'MPE-A1, from scratch, 682-class head'), ('SLGCN', 'SL-GCN (released), pretrained, 682-class head'),
         ('modelV', 'MPE-A1, pretrained (Slovo + Bukva + ASL), 682-class head'), ('A', 'MPE-A1, pretrained, all recordings (2,033–2,378-class head)'), ('B', 'MPE-A1, from scratch, all recordings')]
rows = [['Recordings (evaluation set)'] + [f'{EVAL_N[s]:,}' for s in signers] + ['3,404']]
for k, lab in conds:
    ps = L[k]['per_signer']; rows.append([lab] + [f(ps[s]) for s in signers] + [f(L[k]['top1_macro'])])
rows.append([r'\multicolumn{12}{l}{\textit{Top-1 on words / fingerspelling, MPE-A1 pretrained (signers with at least ten recordings of the group)}}'])
for grp, lab in (('words', 'Words'), ('fingerspelling', 'Fingerspelling')):
    ps = L['step1'][grp]['per_signer']; rows.append([lab] + [f(ps[s]) if s in ps else '---' for s in signers] + [f(L['step1'][grp]['macro'])])
out.append(r'\section{Accuracy per held-out signer on the UzSL corpus}' + '\n' +
           'Table~\\ref{tab:s_loso} gives the top-1 accuracy of every condition of Table~5 of the article on each held-out signer (mean of three seeds), the number of evaluation-set recordings per signer, and the accuracy on words and on fingerspelling for the main model. S06 and S10 (fewer than 20 recordings) form the "other" fold and are excluded from the macro-average.\n' +
           table('s_loso', 'LOSO on the UzSL corpus: top-1 accuracy (\\%) per held-out signer, mean of three seeds; last column: macro-average over the ten signers.',
                 ['Condition'] + signers + ['Macro'], rows, 'l' + 'r' * 11, size=r'\scriptsize'))

# ---------------------------------------------------------------- S3 per-class metrics
def pc_rows(items, letters=True):
    return [[esc(r['cls']), f(r['support'], 0), f(r['precision']), f(r['recall']), f(r['f1'])] for r in items]
bk = N['rsl']['bukva']['per_class']; pl = N['pilot']['per_class_all_sb_ft_a1']
# side-by-side: Bukva (33) | pilot (40)
n = max(len(bk), len(pl)); rows = []
for i in range(n):
    a = pc_rows([bk[i]])[0] if i < len(bk) else [''] * 5; b = pc_rows([pl[i]])[0] if i < len(pl) else [''] * 5
    rows.append(a + b)
out.append(r'\section{Per-class precision, recall and $F_1$}' + '\n' +
           'Tables~\\ref{tab:s_pc1} and~\\ref{tab:s_pc2} give, for every class, the number of test recordings per seed (support), precision, recall and $F_1$ averaged over seeds: Bukva under SD-5 (MPE-A1, pooled over the five folds), the UzSL pilot (MPE-A1 pretrained on Slovo + Bukva and fine-tuned on all recordings, pooled over the two test participants) and the 682-class evaluation set of the UzSL corpus under LOSO (MPE-A1 pretrained on Slovo + Bukva, pooled over the ten folds). Pilot classes with zero support in the training data of their fold have precision and recall equal to zero. The same tables are provided as CSV files with the code.\n' +
           table('s_pc1', 'Per-class metrics (\\%) on Bukva (left; 33 letters of the RSL manual alphabet, SD-5) and on the UzSL pilot (right; 29 letters and 11 numerals). $n$: test recordings per seed.',
                 ['Class', '$n$', 'P', 'R', '$F_1$', 'Class', '$n$', 'P', 'R', '$F_1$'], rows, 'lrrrr|lrrrr', size=r'\scriptsize'))
cp = sorted(N['loso']['per_class_step1'], key=lambda r: r['cls'])
k = (len(cp) + 2) // 3; rows = []
for i in range(k):
    r = []
    for j in range(3):
        idx = i + j * k; r += pc_rows([cp[idx]])[0] if idx < len(cp) else [''] * 5
    rows.append(r)
out.append(table('s_pc2', f'Per-class metrics (\\%) on the 682-class evaluation set of the UzSL corpus (LOSO, MPE-A1 pretrained on Slovo + Bukva, {len(cp)} classes in alphabetical order; "harf:" marks fingerspelled letters, "raqam:" numerals). $n$: test recordings per seed.',
                 ['Class', '$n$', 'P', 'R', '$F_1$'] * 3, rows, 'lrrrr|lrrrr|lrrrr', size=r'\tiny', long=True))
for name, items in (('per_class_bukva_sd5_mpe_a1.csv', bk), ('per_class_pilot_sb_ft_mpe_a1.csv', pl), ('per_class_uzsl_loso_mpe_a1.csv', cp)):
    with open(BUILD / name, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['class', 'support_per_seed', 'precision', 'recall', 'f1'])
        for r in items: w.writerow([r['cls'], f(r['support'], 2), f(r['precision'], 2), f(r['recall'], 2), f(r['f1'], 2)])

# ---------------------------------------------------------------- S4 confusions
def conf_rows(lst, unit):
    return [[esc(t), esc(p), f(c, 1) if unit == 'n' else f(c)] for t, p, c in lst]
sl = N['rsl']['slovo']['top_confusions']; bu = N['rsl']['bukva']['top_confusions']; uz = L['top_confusions_step1']
M = N['pilot']['confusion_all_sb_ft_a1']; cls = M['classes']; R = M['row_normalized']
pil = sorted(((cls[i], cls[j], 100 * R[i][j]) for i in range(len(cls)) for j in range(len(cls)) if i != j and R[i][j] > 0), key=lambda x: -x[2])[:15]
def two(a, b, ha, hb, label, caption):
    n = max(len(a), len(b)); rows = []
    for i in range(n): rows.append((a[i] if i < len(a) else [''] * 3) + (b[i] if i < len(b) else [''] * 3))
    return table(label, caption, ha + hb, rows, 'llr|llr', size=r'\scriptsize')
out.append(r'\section{Most frequent confusions}' + '\n' +
           'Tables~\\ref{tab:s_conf1} and~\\ref{tab:s_conf2} list the most frequent (true class $\\rightarrow$ predicted class) pairs: for Slovo and Bukva under SD-5 (MPE-A1; number of errors per seed, pooled over the five folds), for the UzSL corpus under LOSO (MPE-A1 pretrained on Slovo + Bukva, 682-class head; errors per seed, pooled over the ten folds) and for the UzSL pilot (share of the test recordings of the true class, pooled over the two test participants and three seeds).\n' +
           two(conf_rows(sl, 'n'), conf_rows(bu, 'n'), ['Slovo: true', 'predicted', 'errors'], ['Bukva: true', 'predicted', 'errors'], 's_conf1', 'Most frequent confusions of MPE-A1 under SD-5 on Slovo (left, 20 pairs) and Bukva (right, 20 pairs); errors per seed.') +
           two(conf_rows(uz, 'n'), conf_rows(pil, '%'), ['Corpus: true', 'predicted', 'errors'], ['Pilot: true', 'predicted', '\\% of class'], 's_conf2', 'Most frequent confusions on the UzSL corpus (left, LOSO, 20 pairs; errors per seed) and on the UzSL pilot (right, 15 pairs; per cent of the test recordings of the true class).'))

# ---------------------------------------------------------------- S5 label review
LR = N['label_review']; rows = []
for s, v in LR['by_signer'].items(): rows.append([s, str(v['same'] + v['different'] + v['unsure']), str(v['same']), str(v['different']), str(v['unsure'])])
rows.append([r'\textbf{Total}', str(LR['n_total']), str(sum(v['same'] for v in LR['by_signer'].values())), str(sum(v['different'] for v in LR['by_signer'].values())), str(sum(v['unsure'] for v in LR['by_signer'].values()))])
out.append(r'\section{Label verification}' + '\n' +
           f"The label check of Section~3.4 of the article covered {LR['n_total']} recordings (30 per LOSO signer; see Table~\\ref{{tab:s_lab}}). For {LR['n_with_reference']} of them a reference recording of the same sign existed in the video dictionary (pilot pairs: {LR['pilot_pairs']['n']}, all judged the same sign; lesson pairs: {LR['lesson_pairs']['n']}, {LR['lesson_pairs']['same']} the same); the reviewer judged {LR['same']} of the {LR['n_with_reference']} pairs as the same sign and {LR['different']} as different, an accuracy of {f(LR['accuracy'])}\\% {ci(LR['ci'])} (percentile bootstrap over signers). For the {LR['no_reference']['n']} recordings without a dictionary reference the reviewer compared the recording with the written label alone: {LR['no_reference']['same']} judged correct, {LR['no_reference']['different']} wrong and {LR['no_reference']['unsure']} uncertain; these judgements are not used as an accuracy estimate.\n" +
           table('s_lab', 'Label check by signer: number of reviewed recordings and the reviewer\'s judgement (same sign / different sign / uncertain), with and without a dictionary reference combined.',
                 ['Signer', 'Reviewed', 'Same', 'Different', 'Uncertain'], rows, 'lrrrr'))

# ---------------------------------------------------------------- S6 pilot paired comparisons
P = N['pilot']; rows = []
for k, lab in (('k1 ft', 'One example per class, fine-tuning'), ('k1 probe', 'One example per class, linear probe'), ('k1 scratch', 'One example per class, from scratch'),
               ('k2 ft', 'Two examples per class, fine-tuning'), ('k2 probe', 'Two examples per class, linear probe'), ('k2 scratch', 'Two examples per class, from scratch'),
               ('all ft', 'All recordings, fine-tuning'), ('all probe', 'All recordings, linear probe'), ('all scratch', 'All recordings, from scratch')):
    v = P['A1_vs_SLGCN'][k]; w = P['A2_vs_A1'].get(k)
    rows.append([lab, f(v['A1']), f(v['SLGCN']), f"{v['positive']} / {v['negative']} / {v['ties']}", f(w['A2']) if w else '---', f"{w['positive']} / {w['negative']} / {w['ties']}" if w else '---'])
ub = P['upper_bound_seen_classes']
out.append(r'\section{Pilot recordings: paired comparisons}' + '\n' +
           f"Table~\\ref{{tab:s_pilot}} gives, for each training condition of the pilot, the pooled top-1 accuracy of MPE-A1, of the released SL-GCN and of the graph-free MPE-A2 (all pretrained on Slovo + Bukva where applicable), and the outcome of the pre-registered sign rule: the number of seed--fold pairs (of six) in which the first model is more accurate, less accurate, or equal. A difference is established only when all six pairs have the same sign. Because some test classes do not occur in the training data of a fold, the accuracy reachable on the test participant is bounded by {f(ub['test002'])}\\% (participant 002) and {f(ub['test003'])}\\% (participant 003).\n" +
           table('s_pilot', 'UzSL pilot: pooled top-1 accuracy (\\%) and sign-rule outcomes over the six seed--fold pairs (more / less / equal).',
                 ['Condition', 'MPE-A1', 'SL-GCN', 'A1 vs SL-GCN', 'MPE-A2', 'A2 vs A1'], rows, 'lrrcrc'))

# ---------------------------------------------------------------- S7 few-shot Bukva
F = N['fewshot_bukva']; rows = []
for k, lab in (('k1', '1'), ('k3', '3'), ('k10', '10'), ('all', 'all (about 68)')):
    rows.append([lab, f'{f(F[f"{k} scratch"]["top1"])} {ci(F[f"{k} scratch"]["ci"])}', f'{f(F[f"{k} slovo_probe"]["top1"])} {ci(F[f"{k} slovo_probe"]["ci"])}', f'{f(F[f"{k} slovo_ft"]["top1"])} {ci(F[f"{k} slovo_ft"]["ci"])}',
                 dci(F[f'{k} slovo_probe − scratch']), dci(F[f'{k} slovo_ft − scratch'])])
out.append(r'\section{Simulation of transfer on Bukva}' + '\n' +
           'Table~\\ref{tab:s_few} gives the numbers behind Fig.~7a of the article: MPE-A0 under SD-5 trained from scratch, as a linear probe and with full fine-tuning after pretraining on Slovo, with 1, 3 and 10 training recordings per class and with all training recordings (three seeds; CIs and paired differences over test signers).\n' +
           table('s_few', 'Bukva under SD-5 with reduced training data (MPE-A0, three seeds): top-1 accuracy [95\\% CI] and paired differences to training from scratch.',
                 ['Recordings per class', 'From scratch', 'Linear probe (Slovo)', 'Fine-tuning (Slovo)', '$\\Delta$ probe', '$\\Delta$ fine-tuning'], rows, 'llllll', size=r'\scriptsize\setlength{\tabcolsep}{3pt}'))

# ---------------------------------------------------------------- S8 cost and exports
C = N['cost']; m8 = C['m8']; g2 = C['m11_g2_cost']; g3 = C['m11_g3_cost']; m11 = C['m11_cost']
def m8row(key, lab):
    v = m8[key]; return [lab, f(v['params_inference_m'], 2), f(v['gflops'], 3 if v['gflops'] < 0.01 else 2), f(v['cpu_ms']), str(v['T'])]
rows = [m8row('slovo/**МПК A1** (A0 + EMA)', 'MPE-A1'), m8row('slovo/МПК A1-большая (D 256, N 6)', 'MPE, $D = 256$, $N = 6$'),
        ['MPE-A2 (no graph)', f(m11['ours']['A2']['params_inference_m'], 2), f(m11['ours']['A2']['cpu']['gflops_per_window'], 3), f(m11['ours']['A2']['cpu']['cpu_latency_ms_median']), '32'],
        m8row('slovo/ST-GCN (27 узлов), рецепт mpkrec, EMA', 'ST-GCN, re-implemented'), m8row('slovo/SL-GCN (27 узлов), рецепт mpkrec, EMA', 'SL-GCN, re-implemented'),
        ['ST-GCN, released, 32 frames', f(g3['stgcn_orig']['T32']['params_m'], 2), f(g3['stgcn_orig']['T32']['gflops_per_window'], 2), f(g3['stgcn_orig']['T32']['cpu_latency_ms_median']), '32'],
        ['ST-GCN, released, authors\' recipe (300 frames)', f(g3['stgcn_orig']['T300']['params_m'], 2), f(g3['stgcn_orig']['T300']['gflops_per_window'], 2), f(g3['stgcn_orig']['T300']['cpu_latency_ms_median']), '300'],
        ['SL-GCN, released, 32 frames', f(g2['slgcn_orig']['T32']['params_m'], 2), f(g2['slgcn_orig']['T32']['gflops_per_window'], 2), f(g2['slgcn_orig']['T32']['cpu_latency_ms_median']), '32'],
        ['SL-GCN, released, authors\' recipe (150 frames)', f(g2['slgcn_orig']['T150']['params_m'], 2), f(g2['slgcn_orig']['T150']['gflops_per_window'], 2), f(g2['slgcn_orig']['T150']['cpu_latency_ms_median']), '150'],
        ['MPE-A1, SL-GCN recipe (150 frames)', f(g2['A1']['T150_g2_1']['params_inference_m'], 2), f(g2['A1']['T150_g2_1']['gflops_per_window'], 2), f(g2['A1']['T150_g2_1']['cpu_latency_ms_median']), '150'],
        m8row('slovo/трансформер E1 (вход E1), рецепт e1rec, EMA', 'Keypoint Transformer'), m8row('slovo/E1.6 трансформер-2L (без EMA, общий рецепт)', 'Two-layer Transformer'),
        m8row('slovo/E1.6 Bi-LSTM + внимание (без EMA, общий рецепт)', 'BiLSTM with attention'), m8row('slovo/E1.6 MLP (без EMA, общий рецепт)', 'MLP'),
        ['MViTv2-S (RGB video, released)', f(m11['video']['mvit32-2']['params_m'], 1), '---', f(m11['video']['mvit32-2']['cpu_latency_ms_median']), '32']]
tf = {r['model']: r for r in C['tflite']}
rows2 = [['MPE-A1 (Slovo + Bukva heads)', f(tf['b1_a1']['size_mb']), f(tf['b1_a1']['latency_ms_default_median']), f(tf['b1_a1']['latency_ms_1thread_median']), str(tf['b1_a1']['n_check']), f(100 * tf['b1_a1']['argmax_agree'], 1)],
         ['MPE-A2 (no graph)', f(tf['b3_a2']['size_mb']), f(tf['b3_a2']['latency_ms_default_median']), f(tf['b3_a2']['latency_ms_1thread_median']), str(tf['b3_a2']['n_check']), f(100 * tf['b3_a2']['argmax_agree'], 1)],
         ['MPE-A1 fine-tuned on the pilot', f(tf['b4_sb_ft']['size_mb']), f(tf['b4_sb_ft']['latency_ms_default_median']), f(tf['b4_sb_ft']['latency_ms_1thread_median']), str(tf['b4_sb_ft']['n_check']), f(100 * tf['b4_sb_ft']['argmax_agree'], 1)]]
ORT = {(r['model'], r['threads']): r for r in C.get('onnx_runtime', [])}
rows3 = []
for key, lab in (('A1', 'MPE-A1'), ('A2', 'MPE-A2 (no graph)'), ('ST-GCN_orig', 'ST-GCN, released'), ('SL-GCN_orig', 'SL-GCN, released')):
    if (key, 1) in ORT:
        a = ORT[(key, 1)]; b = ORT[(key, 4)]
        rows3.append([lab, f(a['onnx_mb']), f'{f(a["latency_ms_median"])} ({f(a["latency_ms_p90"])})', f'{f(b["latency_ms_median"])} ({f(b["latency_ms_p90"])})', f'{a["max_abs_dlogit"]:.1e}'])
out.append(r'\section{Computational cost and exported models}' + '\n' +
           'Table~\\ref{tab:s_cost} gives the measured parameters at inference, floating-point operations per window (\\texttt{torch.utils.flop\\_counter}, a multiply--add counted as two operations) and the median CPU latency of 100 runs after 10 warm-up runs (Intel Core i9-14900KF, float32, batch 1, eager PyTorch 2.14.1, default thread count), for the Slovo models at the window used at test time. Latencies of the same model differed by 1--2 ms between sessions. Table~\\ref{tab:s_tflite} gives the TensorFlow Lite exports (float32, tflite-runtime ' + tf['b1_a1']['tflite_runtime'] + ') on the same CPU and the agreement of their argmax with the PyTorch model on the checked recordings.\n' +
           table('s_cost', 'Measured cost of the keypoint models (Slovo) and of the released video model.', ['Model', 'Params (M)', 'GFLOPs / window', 'CPU (ms)', 'Frames'], rows, 'lrrrr') +
           table('s_tflite', 'TensorFlow Lite exports: size, median latency per 32-frame window (default threads / one thread) and argmax agreement with PyTorch.', ['Export', 'Size (MiB)', 'Latency (ms)', '1 thread (ms)', 'Checked', 'Agree (\\%)'], rows2, 'lrrrrr') +
           ('Table~\\ref{tab:s_ort} gives the latency of the ONNX exports in ONNX Runtime ' + ORT[('A1', 1)]['onnxruntime'] + ' (CPUExecutionProvider, opset 17, batch 1, 32-frame window, trained Slovo weights of fold 1, 10 warm-up and 100 timed runs) with one and four intra-op threads on the same CPU. The measurement was taken while the machine was partly loaded by training jobs (1-minute load average 2.5--4.7 on 32 logical cores), so the values are upper bounds; the maximum absolute difference between ONNX and PyTorch logits is given in the last column.\n' +
            table('s_ort', 'ONNX Runtime latency per 32-frame window: median (P90) in ms with one and four threads.', ['Model', 'ONNX (MB)', '1 thread', '4 threads', 'max $|\\Delta$ logit$|$'], rows3, 'lrrrr') if rows3 else ''))

# ---------------------------------------------------------------- S9 keypoint availability
Q = N['qc']['uzsl_corpus_present_summary']; rows = []
for k, lab in (('n_frames', 'Frames per recording'), ('rate_pose', 'Share of frames with pose'), ('rate_face', 'Share of frames with face'), ('rate_rh', 'Share of frames with right hand'), ('rate_lh', 'Share of frames with left hand'), ('rate_any_hand', 'Share of frames with at least one hand'), ('rate_both_hands', 'Share of frames with both hands')):
    v = Q[k]; d = 0 if k == 'n_frames' else 3
    rows.append([lab, f(v['mean'], d), f(v['std'], d), f(v['min'], d), f(v['25%'], d), f(v['50%'], d), f(v['75%'], d), f(v['max'], d)])
out.append(r'\section{Keypoint availability in the UzSL corpus}' + '\n' +
           f"Table~\\ref{{tab:s_qc}} summarises, over the {int(Q['n_frames']['count']):,} recordings of the corpus, the length in frames and the share of frames in which MediaPipe Holistic returned each body part (the detector's presence flags at the default thresholds).\n" +
           table('s_qc', 'Length and keypoint availability of the corpus recordings.', ['Quantity', 'Mean', 'SD', 'Min', 'Q1', 'Median', 'Q3', 'Max'], rows, 'lrrrrrrr'))

TEX = r"""\documentclass[a4paper,10pt]{article}
\usepackage[T2A,T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian,english]{babel}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,float,textcomp}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\setlength{\tabcolsep}{4pt}\sloppy
\newcommand{\ruT}[1]{{\fontencoding{T2A}\selectfont #1}}
\title{Supplementary Material for\\ \emph{Signer-disjoint evaluation and Russian-to-Uzbek transfer for isolated sign recognition: a multi-signer Uzbek Sign Language corpus and a compact pose encoder}}
\author{Maksadjon K. Onarkulov \and Mirsaid A. Yusupov\\ \small Fergana State University, Fergana, Uzbekistan}
\date{}
\begin{document}
\maketitle
\noindent All numbers were computed from the per-recording predictions of the runs by the analysis script released with the code (\texttt{tools/analyze.py}); confidence intervals are percentile bootstrap intervals with 2,000 resamples over test signers (SD-5) or 10,000 resamples over signers (LOSO).
""" + '\n'.join(out) + '\n\\end{document}\n'
(BUILD / 'supp.tex').write_text(TEX, encoding='utf-8')
r = subprocess.run(['latexmk', '-pdf', '-g', '-interaction=nonstopmode', '-halt-on-error', 'supp.tex'], cwd=BUILD, capture_output=True, text=True, errors='replace')
if r.returncode:
    print(r.stdout[-3000:]); sys.exit(1)
print('supp PDF ok', BUILD / 'supp.pdf')
