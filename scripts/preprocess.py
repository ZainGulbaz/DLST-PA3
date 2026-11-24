"""LargeST District-11 extraction using ID2 graph indices and ID data columns."""
import os
os.environ['OMP_NUM_THREADS']=str(os.cpu_count()//2)
import json,time
from pathlib import Path
import h5py,numpy as np,pandas as pd
import os
os.environ['MPLCONFIGDIR']=os.path.join(os.getcwd(),'.cache','matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
random.seed(42)
np.random.seed(42)
t0=time.perf_counter()
meta=pd.read_csv('data/raw/ca_meta.csv'); sd=meta.loc[meta.District==11].copy().reset_index(drop=True)
assert len(sd)==716 and sd.ID.is_unique and sd.ID2.is_unique
idx=sd.ID2.to_numpy(dtype=int); ids=sd.ID.astype(str).to_numpy(dtype=str)
full=np.load('data/raw/ca_rn_adj.npy',mmap_mode='r'); adj=np.array(full[np.ix_(idx,idx)],dtype='float32')
assert np.isfinite(adj).all() and (adj>=0).all() and adj.shape==(716,716)
# Verify ID2 refers to exactly the matching metadata ID; graph has metadata ID2 order.
lookup=meta.set_index('ID2').loc[idx,'ID'].astype(str).to_numpy(); assert np.array_equal(lookup,ids)
frames=[]; raw_missing=[]; zero_counts=[]
for year in [2020,2021]:
    path=f'data/raw/ca_his_raw_{year}.h5'
    with h5py.File(path,'r') as f:
        print('HDF layout',year,flush=True); f.visititems(lambda n,o: print(n,getattr(o,'shape',''),flush=True))
        # pandas fixed-format block, slice only selected columns (avoids reading all CA).
        groups=[v for v in f.values() if isinstance(v,h5py.Group)]
        g=groups[0]; cols=np.array([s.decode() if isinstance(s,bytes) else str(s) for s in g['axis0'][:]])
        pos=np.array([int(np.flatnonzero(cols==s)[0]) for s in ids]); order=np.argsort(pos)
        block=g['block0_values']; assert block.shape[1]==len(cols),block.shape
        arr=block[:,pos[order]][:,np.argsort(order)].astype('float32')
        timestamps=pd.to_datetime(g['axis1'][:])
        assert np.array_equal(cols[pos],ids)
    df=pd.DataFrame(arr,index=timestamps,columns=ids).sort_index()
    assert not df.index.has_duplicates and (df.index.year==year).all()
    complete=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='5min',inclusive='left')
    df=df.reindex(complete); df=df.replace([np.inf,-np.inf],np.nan)
    raw_missing.append(df.isna().sum().to_numpy()); zero_counts.append((df==0).sum().to_numpy())
    frames.append(df); print('Read',year,df.shape,'missing',int(df.isna().sum().sum()),flush=True)
raw=pd.concat(frames); del frames
ranges={'train':('2020-01-01','2021-01-01'),'val':('2021-01-01','2021-04-01'),'test':('2021-04-01','2022-01-01')}
arrays={}; times={}; miss_rows=[]; fallback=np.zeros(716,dtype='float32'); stats={}
for name,(start,end) in ranges.items():
    df=raw.loc[(raw.index>=start)&(raw.index<end)]; missing=df.isna().sum()
    for sensor,cnt in zip(ids,missing): miss_rows.append({'split':name,'sensor_id':sensor,'missing_count':int(cnt),'missing_percent':float(cnt/len(df)*100)})
    # Causal within-split forward fill. Initial gaps use zero for train, train-only mean otherwise.
    clean=df.ffill().fillna(pd.Series(fallback,index=ids))
    if name=='train': fallback=clean.mean().to_numpy(dtype='float32')
    resampled=clean.resample('15min').sum(min_count=3)
    assert np.isfinite(resampled.to_numpy()).all()
    arrays[name]=resampled.to_numpy(dtype='float32'); times[name]=resampled.index.astype('int64').to_numpy()
    stats[name]={'timestamps':len(resampled),'windows':len(resampled)-11,'start':str(resampled.index[0]),'end':str(resampled.index[-1]),'raw_missing':int(missing.sum()),'raw_total':int(df.size),'missing_percent':float(missing.sum()/df.size*100)}
mean=float(arrays['train'].mean(dtype='float64')); std=float(arrays['train'].std(dtype='float64')); assert std>0
np.savez_compressed('data/processed/sd.npz',**arrays,**{k+'_times':v for k,v in times.items()},mean=mean,std=std,adj=adj,ids=ids,graph_indices=idx)
sd.to_csv('data/processed/sd_meta.csv',index=False); sd.to_csv('results/sd_meta.csv',index=False)
pd.DataFrame(miss_rows).to_csv('results/missing_per_sensor.csv',index=False)
pd.DataFrame({'sensor_id':ids,'missing_2020':raw_missing[0],'missing_2021':raw_missing[1],'zeros_2020':zero_counts[0],'zeros_2021':zero_counts[1]}).to_csv('results/raw_quality.csv',index=False)
binary=adj!=0; off=binary.copy(); np.fill_diagonal(off,False)
deg=off.sum(1); selected=int(np.argsort(deg)[len(deg)//2]); one=np.flatnonzero(off[selected]); within=set(one.tolist())
for j in one: within.update(np.flatnonzero(off[j]).tolist())
within.discard(selected); two=np.array(sorted(within),dtype=int)
summary={'selection':'District == 11; graph ID2; data ID string','nodes':716,'counties':sd.County.value_counts().to_dict(),'edges_with_diagonal':int(binary.sum()),'edges_without_diagonal':int(off.sum()),'diagonal_nonzeros':int(np.diag(binary).sum()),'symmetry_max':float(np.abs(adj-adj.T).max()),'mean_degree':float(deg.mean()),'min_degree':int(deg.min()),'max_degree':int(deg.max()),'splits':stats,'train_mean':mean,'train_std':std,'missing_total':int(sum(x.sum() for x in raw_missing)),'missing_percent':float(sum(x.sum() for x in raw_missing)/raw.size*100),'neighborhood':{'index':selected,'id':str(ids[selected]),'one_hop_indices':one.tolist(),'one_hop_ids':ids[one].tolist(),'within_two_hop_indices':two.tolist(),'within_two_hop_ids':ids[two].tolist()},'preprocess_seconds':time.perf_counter()-t0}
json.dump(summary,open('results/data_summary.json','w'),indent=2)
fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(adj,cmap='viridis'); fig.colorbar(im,label='Road adjacency weight'); ax.set(xlabel='Source sensor index j',ylabel='Receiving sensor index i',title='San Diego road adjacency'); fig.tight_layout(); fig.savefig('figures/road_adjacency.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4)); ax.hist(deg,bins=30); ax.set(xlabel='Nonself row neighbours',ylabel='Number of sensors',title='Road graph degree distribution'); fig.tight_layout(); fig.savefig('figures/degrees.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4)); tt=pd.to_datetime(times['train'][:672]);
for s in [0,357,715]: ax.plot(tt,arrays['train'][:672,s],label=str(ids[s]),linewidth=.8)
ax.set(xlabel='Timestamp (source local clock)',ylabel='Vehicles / 15 minutes',title='First week of 2020'); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig('figures/week_flow.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(sd.Lng,sd.Lat,c='lightgray',s=8,label='Other sensors'); ax.scatter(sd.Lng.iloc[two],sd.Lat.iloc[two],s=20,label='Within two hops'); ax.scatter(sd.Lng.iloc[one],sd.Lat.iloc[one],s=30,label='One hop'); ax.scatter(sd.Lng.iloc[selected],sd.Lat.iloc[selected],c='red',s=65,marker='*',label='Selected '+str(ids[selected])); ax.set(xlabel='Longitude',ylabel='Latitude',title='Outgoing row-neighbour reachability'); ax.legend(); fig.tight_layout(); fig.savefig('figures/neighborhood.png',dpi=150); plt.close(fig)
print(json.dumps(summary,indent=2),flush=True); print('PREPROCESS COMPLETE',flush=True)
