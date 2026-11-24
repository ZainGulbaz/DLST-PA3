"""Exhaustive multiset checks and ring-road numerical experiments, seed 42."""
import json
from fractions import Fraction
import numpy as np
import os
os.environ['MPLCONFIGDIR']=os.path.join(os.getcwd(),'.cache','matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
random.seed(42)
np.random.seed(42)
triples=np.array([(a,b,c) for a in range(9) for b in range(9) for c in range(9) if 1<=a+b+c<=8])
codes=triples@np.array([1,9,81]); decoded=np.stack([codes%9,(codes//9)%9,codes//81],1)
assert np.array_equal(decoded,triples) and len(set(codes))==len(codes)
print('Triples checked:',len(triples))
rows=[]
for task in range(3):
    answers=[int(t[1]) if task==0 else bool(2*t[0]>sum(t)) if task==1 else bool(t[0]>0) for t in triples]
    for method in ['sum','mean','max']:
        groups={}; witness=None
        for t,y in zip(triples,answers):
            key=tuple(map(int,t)) if method=='sum' else tuple(Fraction(int(v),int(sum(t))) for v in t) if method=='mean' else tuple(map(int,t>0))
            if key in groups and groups[key][1]!=y: witness={'counts1':groups[key][0],'answer1':groups[key][1],'counts2':t.tolist(),'answer2':y,'aggregate':[str(v) for v in key]}
            if witness is None: groups[key]=(t.tolist(),y)
        if witness is None:
            for t,y in zip(triples,answers):
                ag=t if method=='sum' else t/sum(t) if method=='mean' else (t>0).astype(int)
                pred=int(ag[1]) if task==0 else bool(2*ag[0]>sum(ag)) if task==1 else bool(ag[0]>0)
                assert pred==y
        rows.append({'task':task+1,'method':method,'possible':witness is None,'witness':witness})
assert [r['possible'] for r in rows]==[True,False,False,True,True,False,True,True,True]
x=np.eye(1,40,20).ravel(); target=np.roll(x,-1)
M=np.stack([x,np.roll(x,1)+np.roll(x,-1)],1)
a,b=np.linalg.lstsq(M,target,rcond=None)[0]; assert np.allclose([a,b],[0,.5])
initial=np.exp(-(np.arange(40)-20)**2/8); fitted=[initial]; true=[initial]
for _ in range(20): fitted.append(a*fitted[-1]+b*(np.roll(fitted[-1],1)+np.roll(fitted[-1],-1))); true.append(np.roll(true[-1],-1))
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,arr,title in zip(axs,[fitted,true],['Symmetric fitted layer','True backward shift']):
    im=ax.imshow(arr,vmin=0,vmax=1,aspect='auto',origin='upper',extent=[-.5,39.5,20.5,-.5]); ax.set(xlabel='Sensor',ylabel='Time step',title=title)
fig.colorbar(im,ax=axs,label='Slowdown'); fig.savefig('figures/q2_rollout.png',dpi=160); plt.close(fig)
rng=np.random.default_rng(42); current=rng.normal(size=40)
directional=np.roll(current,-1)
back_prev=np.roll(current,1); forward_prev=np.roll(current,-1)
hist_back=np.roll(current,1)+np.roll(current,-1)-back_prev
hist_forward=np.roll(current,1)+np.roll(current,-1)-forward_prev
assert np.allclose(hist_back,np.roll(current,-1))
assert np.allclose(hist_forward,np.roll(current,1))
result={'triples_checked':len(triples),'rows':rows,'least_squares':[float(a),float(b)],'isolated_predictions':(M@np.array([a,b]))[[19,20,21]].tolist(),'isolated_targets':target[[19,20,21]].tolist(),'directional_error':float(np.max(np.abs(directional-np.roll(current,-1)))),'history_backward_error':float(np.max(np.abs(hist_back-np.roll(current,-1)))),'history_forward_error':float(np.max(np.abs(hist_forward-np.roll(current,1))))}
json.dump(result,open('results/q1_q2.json','w'),indent=2)
print(json.dumps(result,indent=2))
