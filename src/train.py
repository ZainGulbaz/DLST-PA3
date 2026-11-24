"""Run as python -m src.train; timing trials followed by budgeted full runs."""
import os,json,time,argparse,platform,copy
import numpy as np
import pandas as pd
import torch
from .models import Forecaster
from .common import seed,loaders,evaluate

def run(kind,z,epochs,trial=False):
    torch.manual_seed(42); np.random.seed(42)
    ls=loaders(z); model=Forecaster(kind,torch.from_numpy(z['adj']))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    best=float('inf'); bad=0; history=[]; started=time.perf_counter(); best_epoch=0
    for epoch in range(1,epochs+1):
        t=time.perf_counter(); model.train(); loss_sum=0.; count=0
        for x,y in ls['train']:
            opt.zero_grad(set_to_none=True); loss=(model(x)-y).abs().mean(); loss.backward(); opt.step()
            loss_sum+=loss.item()*len(x); count+=len(x)
        train_seconds=time.perf_counter()-t
        v=evaluate(model,ls['val'],1.)
        row={'epoch':epoch,'train_mae_normalized':loss_sum/count,'val_mae_normalized':v['mae'],'training_epoch_seconds':train_seconds,'epoch_seconds':time.perf_counter()-t}
        history.append(row); print(kind,'trial' if trial else 'final',json.dumps(row),flush=True)
        if v['mae']<best:
            best=v['mae']; best_epoch=epoch; bad=0; best_state=copy.deepcopy(model.state_dict())
        else: bad+=1
        if bad>=5: break
    seconds=time.perf_counter()-started
    model.load_state_dict(best_state)
    result={'model':kind,'epochs':len(history),'best_epoch':best_epoch,'max_epochs':epochs,'training_seconds':seconds,'history':history,'parameters':sum(p.numel() for p in model.parameters())}
    if not trial:
        torch.save({'state_dict':best_state,'kind':kind,'hidden':16,'k':20,'seed':42},f'checkpoints/{kind}.pt')
        result['val']=evaluate(model,ls['val'],float(z['std'])); result['test']=evaluate(model,ls['test'],float(z['std']))
        if kind=='hybrid': result['alpha']=model.alpha_logit.sigmoid().item()
        json.dump(result,open(f'results/{kind}.json','w'),indent=2)
    return result

def main():
    seed(); z=np.load('data/processed/sd.npz'); kinds=['temporal','fixed','adaptive','hybrid']
    trials={}
    for kind in kinds:
        trials[kind]=run(kind,z,1,True)
        json.dump(trials,open('results/timing_trials.json','w'),indent=2)
    # Equal maximum epochs; reserve timing-test cost inside the two-hour budget.
    used=sum(r['training_seconds'] for r in trials.values()); per_epoch=used
    epochs=max(1,min(30,int((7200-used)*.88/per_epoch)))
    config={'seed':42,'threads':torch.get_num_threads(),'torch':torch.__version__,'python':platform.python_version(),'hidden':16,'embedding_dim':10,'top_k':20,'batch_size':64,'learning_rate':.001,'patience':5,'max_epochs':epochs,'timing_trial_seconds':used,'budget_seconds':7200,'normalization':'global train-only z-score','residual':'last reading added equally in all models','fixed_normalization':'row-normalized A+I (directed adjacency)'}
    json.dump(config,open('results/config.json','w'),indent=2); print('CONFIG',json.dumps(config),flush=True)
    results={}
    for kind in kinds:
        results[kind]=run(kind,z,epochs)
        json.dump(results,open('results/results.json','w'),indent=2)
    pd.DataFrame([{'Model':k,'Adjacency':'None' if k=='temporal' else k,'MAE':v['test']['mae'],'RMSE':v['test']['rmse'],'Training time (s)':v['training_seconds'],'Epochs':v['epochs'],'Best epoch':v['best_epoch']} for k,v in results.items()]).to_csv('results/results_table.csv',index=False)
    print('TRAINING COMPLETE',flush=True)
if __name__=='__main__': main()
