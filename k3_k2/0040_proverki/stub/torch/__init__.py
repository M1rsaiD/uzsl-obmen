# заглушка torch: проверяются только numpy-функции sohn.py (предобработка)
class Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else Dummy()
    def __getattr__(self, n): return Dummy()
    def __mro_entries__(self, bases): return (Dummy,)
def __getattr__(n): return Dummy
