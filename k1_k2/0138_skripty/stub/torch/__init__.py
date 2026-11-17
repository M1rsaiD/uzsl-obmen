# к1 0138: заглушка torch только для импорта sohn.py (numpy-часть); слои не исполняются
class _D:
    def __init__(self,*a,**k): pass
    def __call__(self,*a,**k): return _D()
    def __getattr__(self,n): return _D()
    def __mro_entries__(self,bases): return (object,)
def __getattr__(n): return _D()
nn=_D(); 
