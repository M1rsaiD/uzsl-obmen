"""Ред. 3, 22.11 № 2: прирост параметров PartTokens против stem Sohn — 16 сочетаний (U01a/U04/U01b × Slovo 1001, Bukva 33). ЦП, только чтение кода ветки."""
import itertools, sys
sys.path.insert(0, sys.argv[1])          # <wt_e2>/src (ветка turnir-e2-levers)
import torch
from mpk import sohn
torch.manual_seed(0)
cnt = lambda m: sum(p.numel() for p in m.parameters())
rows = []
for n_cls, name in ((1001, "slovo"), (33, "bukva")):
    for u01a, u04, u01b in itertools.product((0, 1), repeat=3):
        tok = tuple(t for t, f in (("u01a", u01a), ("u04", u04), ("u01b", u01b)) if f)
        ch = sohn.input_channels(tok)
        base = cnt(sohn.SohnNet(n_cls, channels=ch))
        n2 = sohn.SohnNet(n_cls, channels=ch, part_tokens=True, tokens=tok)
        d = cnt(n2) - base
        rows.append((name, "+".join(tok) or "база", ch, n2.tok.n_tok, base, d, 100 * d / base))
        cols = sohn.part_cols(tok)
        assert sorted(sum(cols, [])) == list(range(ch)) and len(cols) == n2.tok.n_tok
for r in rows:
    print("%-6s %-14s ch=%4d n_tok=%d base=%9d d=%7d %+.2f%%" % r)
ds = {r[3]: set() for r in rows}
for r in rows: ds[r[3]].add(r[5])
print("прирост по n_tok:", {k: sorted(v) for k, v in ds.items()})
print("диапазон %%: %.2f … %.2f" % (min(r[6] for r in rows), max(r[6] for r in rows)))
print("формула n_tok·37440 + 37248:", {k: k * 37440 + 37248 for k in (3, 4)})
