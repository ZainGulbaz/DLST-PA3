"""One-time migration of this run's own generated string IDs to a non-pickle dtype.
Future preprocessing already writes Unicode IDs directly. Numeric arrays are verified unchanged.
"""
import numpy as np,hashlib,json
path='data/processed/sd.npz'
with np.load(path,allow_pickle=True) as z: arrays={k:z[k] for k in z.files}
old_ids=arrays['ids'].tolist()
hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items() if k!='ids'}
arrays['ids']=np.asarray(old_ids,dtype=str)
np.savez_compressed(path,**arrays)
with np.load(path,allow_pickle=False) as z:
    assert z['ids'].tolist()==old_ids
    for k,h in hashes.items(): assert hashlib.sha256(z[k].tobytes()).hexdigest()==h
json.dump({'status':'PASS','change':'IDs object dtype to Unicode; all numeric arrays byte-identical','numeric_sha256':hashes},open('results/id_dtype_migration.json','w'),indent=2)
print('ID dtype migration complete; numeric arrays unchanged.')
