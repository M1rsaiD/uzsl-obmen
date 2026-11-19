"""к2 0176 (к3 0055 разд. 4): src/mpk/train.py и model.py — копия к3 (mpe-slovo-bukva 7563b5e, блобы 92247ac, 14cd33b) против OM
(master = ветки рычагов и порта: abee856, 8fe58c6). Тексты разные (в выпуске комментарии и строки переведены на английский),
поэтому сравниваю код: ast.unparse без строк документации, строковые константы → "S"; печать — дифф кода.
Запуск: python -I ast_same_k2.py <старый.py> <новый.py>"""
import ast, difflib, sys

class N(ast.NodeTransformer):
    def generic_visit(self, n):
        super().generic_visit(n)
        b = getattr(n, "body", None)
        if isinstance(b, list) and b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            n.body = b[1:] or [ast.Pass()]
        return n
    def visit_Constant(self, n):
        return ast.Constant("S") if isinstance(n.value, str) else n
    def visit_JoinedStr(self, n):
        return ast.Constant("S")

def code(p):
    return ast.unparse(N().visit(ast.parse(open(p, encoding="utf-8").read()))).splitlines()

a, b = code(sys.argv[1]), code(sys.argv[2])
d = list(difflib.unified_diff(a, b, "7563b5e", "OM", n=1, lineterm=""))
print(f"строк кода {len(a)} / {len(b)}; отличий {sum(1 for l in d if l[:1] in '+-' and l[:3] not in ('+++', '---'))}")
print("\n".join(d))
