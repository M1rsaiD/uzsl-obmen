#!/usr/bin/env python3
"""Проверка: каждый интервал [a; b] и каждая разность в тексте статьи существует в out/numbers.json (с учётом знака и округления)."""
import json, re, sys, glob
lang = sys.argv[1] if len(sys.argv) > 1 else 'en'
N = json.load(open('out/numbers.json'))
S = set()
def f(x): return f'{x:.1f}'
def add(lo, hi):
    for a, b in ((lo, hi), (-hi, -lo)):   # и обратная разность
        S.add(f'[{f(a)}; {f(b)}]'); S.add(f'[{"+" if a >= 0 else "−"}{f(abs(a))}; {"+" if b >= 0 else "−"}{f(abs(b))}]')
def walk(o):
    if isinstance(o, dict):
        if 'ci' in o and isinstance(o['ci'], list) and len(o['ci']) == 2: add(*o['ci'])
        if 'top5_ci' in o: add(*o['top5_ci'])
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(N)
txt = ''.join(open(p, encoding='utf-8').read() for p in sorted(glob.glob(f'src/{lang}/*.md')))
if lang == 'ru': txt = re.sub(r'(\d),(\d)', r'\1.\2', txt)
ivs = re.findall(r'\[[+−]?\d+\.\d; [+−]?\d+\.\d\]', txt)
bad = [iv for iv in ivs if iv not in S]
print(lang, len(ivs), 'intervals;', len(bad), 'not found:', bad)
sys.exit(1 if bad else 0)
