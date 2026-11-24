"""Evaluate saved predictions and quantify learned graph geometry."""
import os,json
os.environ['OMP_NUM_THREADS']=str(os.cpu_count()//2)
import numpy as np,pandas as pd,torch
import os
os.environ['MPLCONFIGDIR']=os.path.join(os.getcwd(),'.cache','matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models import Forecaster
from src.common import seed
seed(); z=np.load('data/processed/sd.npz'); meta=pd.read_csv('results/sd_meta.csv'); road=z['adj']; n=len(road)
results=json.load(open('results/results.json')); graphs={}; predictions={}
test=z['test']; ids=z['ids']; sensor_idx=[0,357,715]; length=288
x=torch.from_numpy(np.stack([(test[i:i+8]-z['mean'])/z['std'] for i in range(length)]).astype('float32'))
for kind in results:
    model=Forecaster(kind,torch.from_numpy(road)); ck=torch.load(f'checkpoints/{kind}.pt',weights_only=True); model.load_state_dict(ck['state_dict']); model.eval()
    with torch.no_grad():
        chunks=[model(b).numpy()*z['std']+z['mean'] for b in x.split(64)]
        predictions[kind]=np.concatenate(chunks)
        if kind in ['adaptive','hybrid']:
            graphs[kind]=model.adjacency().to_dense().numpy()
            if kind=='hybrid': graphs['hybrid_adaptive_component']=model.adaptive().to_dense().numpy()
np.savez_compressed('results/predictions_sample.npz',actual=np.stack([test[i+8:i+12] for i in range(length)]),sensor_indices=sensor_idx,timestamps=z['test_times'][8:8+length],**predictions)
np.savez_compressed('results/learned_adjacencies.npz',**graphs)
tt=pd.to_datetime(z['test_times'][8:8+length])
for s in sensor_idx:
    fig,ax=plt.subplots(figsize=(11,4)); ax.plot(tt,test[8:8+length,s],c='black',label='Actual',linewidth=1.3)
    for k,p in predictions.items(): ax.plot(tt,p[:,0,s],label=k,linewidth=.8,alpha=.8)
    ax.set(title=f'Sensor {ids[s]}: first horizon, test period',xlabel='Target timestamp',ylabel='Vehicles / 15 minutes'); ax.legend(ncol=5); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(f'figures/prediction_{ids[s]}.png',dpi=150); plt.close(fig)
for kind,a in graphs.items():
    fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(a,cmap='magma'); fig.colorbar(im,label='Row-normalized weight'); ax.set(xlabel='Source j',ylabel='Receiving node i',title=kind.replace('_',' ')); fig.tight_layout(); fig.savefig(f'figures/{kind}_adjacency.png',dpi=150); plt.close(fig)
assert np.isfinite(meta[['Lat','Lng']].to_numpy()).all()
for a in graphs.values():
    assert np.isfinite(a).all() and (a>=0).all() and np.allclose(a.sum(1),1,atol=1e-6)
lat=np.deg2rad(meta.Lat.to_numpy()); lon=np.deg2rad(meta.Lng.to_numpy()); q=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2; dist=6371*2*np.arcsin(np.sqrt(np.clip(q,0,1)))
mask=road>0; np.fill_diagonal(mask,False); road_dist=dist[mask]; rng=np.random.default_rng(42); ri=rng.integers(n,size=7160); rj=(ri+rng.integers(1,n,size=7160))%n; random_dist=dist[ri,rj]
summary={}; overlap_rows=[]
for kind,a in graphs.items():
    tmp=a.copy(); np.fill_diagonal(tmp,-np.inf); top=np.argsort(-tmp,axis=1)[:,:10]; rows=np.arange(n)[:,None]
    overlaps=mask[rows,top].mean(1); distances=dist[rows,top]
    # Positive links only when quantifying absent-road examples.
    absent=(~mask)&(a>0); np.fill_diagonal(absent,False); pairs=np.argwhere(absent); weights=a[absent]; order=np.argsort(-weights)[:10]
    examples=[{'i':int(pairs[p,0]),'j':int(pairs[p,1]),'sensor_i':str(ids[pairs[p,0]]),'sensor_j':str(ids[pairs[p,1]]),'weight':float(weights[p]),'km':float(dist[tuple(pairs[p])])} for p in order]
    summary[kind]={'mean_top10_road_overlap':float(overlaps.mean()),'median_top10_road_overlap':float(np.median(overlaps)),'top10_mean_km':float(distances.mean()),'top10_median_km':float(np.median(distances)),'road_mean_km':float(road_dist.mean()),'road_median_km':float(np.median(road_dist)),'random_mean_km':float(random_dist.mean()),'random_median_km':float(np.median(random_dist)),'positive_edges':int((a>0).sum()),'examples_absent_road':examples}
    overlap_rows.extend([{'graph':kind,'sensor_id':str(ids[i]),'top10_road_fraction':float(v)} for i,v in enumerate(overlaps)])
    fig,ax=plt.subplots(figsize=(8,4)); ax.boxplot([distances.ravel(),road_dist,random_dist],tick_labels=['Learned top 10','Road edges','Random ordered pairs'],showfliers=False); ax.set(ylabel='Haversine distance (km)',title=kind+' geographic distances'); fig.tight_layout(); fig.savefig(f'figures/{kind}_distances.png',dpi=150); plt.close(fig)
json.dump(summary,open('results/graph_analysis.json','w'),indent=2); pd.DataFrame(overlap_rows).to_csv('results/graph_overlap_per_sensor.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for k,r in results.items():
    h=r['history']; axs[0].plot([v['epoch'] for v in h],[v['train_mae_normalized'] for v in h],label=k); axs[1].plot([v['epoch'] for v in h],[v['val_mae_normalized'] for v in h],label=k)
for ax,title in zip(axs,['Train MAE','Validation MAE']): ax.set(xlabel='Epoch',ylabel='Normalized MAE',title=title); ax.legend()
fig.savefig('figures/learning_curves.png',dpi=150); plt.close(fig)
print(json.dumps(summary,indent=2)); print('ANALYSIS COMPLETE')
