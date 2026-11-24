import os,random
import numpy as np
import torch

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS',os.cpu_count()//2)))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

class Windows(torch.utils.data.Dataset):
    def __init__(self,data,mean,std):
        self.data=torch.from_numpy(((data-mean)/std).astype('float32'))
    def __len__(self): return len(self.data)-11
    def __getitem__(self,i): return self.data[i:i+8],self.data[i+8:i+12]

def loaders(z,batch=64):
    return {s:torch.utils.data.DataLoader(Windows(z[s],z['mean'],z['std']),batch_size=batch,shuffle=s=='train',num_workers=0) for s in ('train','val','test')}

@torch.no_grad()
def evaluate(model,loader,std):
    model.eval(); absolute=np.zeros(4); squared=np.zeros(4); count=0
    for x,y in loader:
        e=((model(x)-y)*std).double(); absolute+=e.abs().sum((0,2)).numpy(); squared+=(e**2).sum((0,2)).numpy(); count+=e.shape[0]*e.shape[2]
    return {'mae':float(absolute.sum()/(4*count)),'rmse':float(np.sqrt(squared.sum()/(4*count))),'horizon_mae':(absolute/count).tolist(),'horizon_rmse':np.sqrt(squared/count).tolist(),'count_per_horizon':count}
