#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export XDG_CACHE_HOME="$PWD/.cache"
.kaggle-venv/bin/kaggle datasets files liuxu77/largest --csv > results/kaggle_files.csv
cat results/kaggle_files.csv
.venv/bin/python - <<'PY'
import csv,json
rows=list(csv.DictReader(open('results/kaggle_files.csv')))
key=next(k for k in rows[0] if k.lower()=='name')
names=[r[key] for r in rows]
selected=[n for n in names if (n.endswith('.h5') and ('2020' in n or '2021' in n)) or n.endswith(('.csv','.npy'))]
assert len(selected)==4,selected
json.dump(selected,open('results/download_files.json','w'),indent=2)
open('results/download_files.txt','w').write('\n'.join(selected)+'\n')
PY
while IFS= read -r filename; do
 .kaggle-venv/bin/kaggle datasets download -d liuxu77/largest -f "$filename" -p data/raw
 done < results/download_files.txt
.venv/bin/python - <<'PY'
import pathlib,zipfile
for p in pathlib.Path('data/raw').glob('*.zip'):
 with zipfile.ZipFile(p) as z: z.extractall('data/raw')
print('DOWNLOAD COMPLETE')
PY
