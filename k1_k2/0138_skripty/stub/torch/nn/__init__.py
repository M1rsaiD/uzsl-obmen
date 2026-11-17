class Module:
    def __init__(self,*a,**k): pass
def __getattr__(n):
    return type(n,(Module,),{})
