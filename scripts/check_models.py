import torch,numpy as np,time,json
from src.models import Forecaster
from src.common import seed,Windows
seed(); road=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
results={}
for kind in ['temporal','fixed','adaptive','hybrid']:
    torch.manual_seed(42); m=Forecaster(kind,road,k=2); x=torch.randn(2,8,3)
    y=m(x); assert y.shape==(2,4,3) and torch.isfinite(y).all(); y.square().mean().backward()
    if kind!='temporal':
        a=m.adjacency().to_dense(); assert torch.allclose(a.sum(1),torch.ones(3),atol=1e-6); assert (a>=0).all()
    if kind in ['adaptive','hybrid']: assert m.E1.grad is not None and m.E1.grad.abs().sum()>0
    if kind=='hybrid': assert m.alpha_logit.grad is not None
    results[kind]='shape, finite outputs, gradients and normalization passed'
a=np.arange(60,dtype='float32').reshape(20,3); w=Windows(a,0,1); assert len(w)==9 and np.array_equal(w[8][1].numpy(),a[16:20])
json.dump(results,open('results/model_checks.json','w'),indent=2); print(results)
