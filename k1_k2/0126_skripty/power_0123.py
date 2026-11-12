import numpy as np
from scipy import stats
rng=np.random.default_rng(1)
sizes=np.array([143,114,22,11,3,1]); N=sizes.sum()
def sim(delta,q,nsim=1000,nb=2000,mode='hier',het=0.0):
    opens=0
    for _ in range(nsim):
        # per-record paired diff d in {+1,-1,0}; per-cluster effect
        eff=delta+het*rng.standard_normal(len(sizes)) if het>0 else np.full(len(sizes),delta)
        cnt=[]
        for n,e in zip(sizes,eff):
            pp=np.clip(q/2+e/2,0,1); pm=np.clip(q/2-e/2,0,1)
            c=rng.multinomial(n,[pp,pm,1-pp-pm]); cnt.append(c)
        cnt=np.array(cnt)
        est=(cnt[:,0]-cnt[:,1]).sum()/N*100
        if est<1.008: continue
        # bootstrap
        if mode=='hier':
            idx=rng.integers(0,len(sizes),(nb,len(sizes)))
            tot=np.zeros(nb); den=np.zeros(nb)
            for j in range(len(sizes)):
                cj=cnt[idx[:,j]]; nj=sizes[idx[:,j]]
                p=cj/nj[:,None]
                draws=np.array([rng.multinomial(n,pr) for n,pr in zip(nj,p)])
                tot+=draws[:,0]-draws[:,1]; den+=nj
            b=tot/den*100
        else:
            tot=np.zeros(nb)
            for j,n in enumerate(sizes):
                draws=rng.multinomial(n,cnt[j]/n,size=nb)
                tot+=draws[:,0]-draws[:,1]
            b=tot/N*100
        lo=np.quantile(b,0.005)
        if lo>0: opens+=1
    return opens/nsim
for mode in ('hier','strat'):
    for q in (0.1,0.2,0.3):
        print(mode,q,[round(sim(d,q,nsim=400,nb=1000,mode=mode),3) for d in (0.0,0.05,0.10)])
print('strat het5 zero-mean', [round(sim(0.0,q,nsim=400,nb=1000,mode='strat',het=0.05),3) for q in (0.1,0.2,0.3)])
# Q21 coverage of percentile-like normal interval with plug-in SE, n=10
n=10; k=stats.norm.ppf(0.995)*np.sqrt((n-1)/n)
print('k',round(k,3),'coverage under t9', round(1-2*stats.t.sf(k,df=n-1),4), 't9 99% quantile', round(stats.t.ppf(0.995,9),3))
