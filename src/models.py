"""Shared temporal encoder and head; graph models differ only in adjacency."""
import torch
from torch import nn

class Forecaster(nn.Module):
    def __init__(self,kind,road,hidden=16,k=20):
        super().__init__(); self.kind=kind; self.k=k; n=road.shape[0]
        self.encoder=nn.Sequential(nn.Conv1d(1,hidden,3,padding=1),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU())
        self.head=nn.Linear(hidden,4)
        # A common last-reading residual permits local persistence in every variant.
        if kind!='temporal': self.gconv=nn.Linear(hidden,hidden,bias=False)
        a=road.float()+torch.eye(n); a=a/a.sum(1,keepdim=True)
        self.register_buffer('fixed',a.to_sparse().coalesce())
        if kind in ('adaptive','hybrid'):
            self.E1=nn.Parameter(torch.randn(n,10)*.1); self.E2=nn.Parameter(torch.randn(n,10)*.1)
        if kind=='hybrid': self.alpha_logit=nn.Parameter(torch.zeros(()))
    def adaptive(self):
        scores=torch.relu(self.E1@self.E2.T)
        # Top-k BEFORE softmax. Gradients through selected logits/values, not indices.
        values,indices=scores.topk(self.k,dim=1)
        weights=values.softmax(1)
        rows=torch.arange(scores.shape[0])[:,None].expand_as(indices)
        return torch.sparse_coo_tensor(torch.stack([rows.reshape(-1),indices.reshape(-1)]),weights.reshape(-1),scores.shape).coalesce()
    def adjacency(self):
        if self.kind=='fixed': return self.fixed
        adp=self.adaptive()
        if self.kind=='adaptive': return adp
        alpha=self.alpha_logit.sigmoid()
        return (self.fixed*alpha+adp*(1-alpha)).coalesce()
    def forward(self,x):
        b,t,n=x.shape
        z=self.encoder(x.permute(0,2,1).reshape(b*n,1,t))[:,:,-1].reshape(b,n,-1)
        if self.kind!='temporal':
            flat=z.permute(1,0,2).reshape(n,-1)
            z=torch.relu(self.gconv(torch.sparse.mm(self.adjacency(),flat).reshape(n,b,-1).permute(1,0,2)))
        return self.head(z).permute(0,2,1)+x[:,-1:,:]
