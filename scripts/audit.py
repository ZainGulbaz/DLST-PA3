"""Artifact, leakage-boundary, checkpoint and report consistency checks."""
import os,json,re,hashlib,pathlib
os.environ['OMP_NUM_THREADS']=str(os.cpu_count()//2)
import numpy as np,torch,pandas as pd
from pypdf import PdfReader
from src.models import Forecaster
from src.common import seed
seed(); z=np.load('data/processed/sd.npz'); r=json.load(open('results/results.json')); d=json.load(open('results/data_summary.json'))
assert set(r)=={'temporal','fixed','adaptive','hybrid'}
assert z['adj'].shape==(716,716) and len(set(z['ids']))==716
assert np.isclose(z['mean'],z['train'].mean(dtype='float64'))
assert np.isclose(z['std'],z['train'].std(dtype='float64'))
for split in ['train','val','test']:
    a=z[split]; times=z[split+'_times']; assert np.isfinite(a).all(); assert (np.diff(times)==15*60*10**9).all()
    assert len(a)-11==d['splits'][split]['windows']
assert z['train_times'][-1]<z['val_times'][0] and z['val_times'][-1]<z['test_times'][0]
for kind,result in r.items():
    ck=torch.load(f'checkpoints/{kind}.pt',weights_only=True); m=Forecaster(kind,torch.from_numpy(z['adj'])); m.load_state_dict(ck['state_dict'])
    base=Forecaster('fixed',torch.from_numpy(z['adj'])).fixed.to_dense(); assert torch.equal(m.fixed.to_dense(),base)
    for split in ['val','test']:
        a=result[split]; assert np.isclose(a['mae'],np.mean(a['horizon_mae']))
        assert np.isclose(a['rmse'],np.sqrt(np.mean(np.square(a['horizon_rmse']))))
        assert a['count_per_horizon']==(len(z[split])-11)*716
    h=result['history']; assert result['best_epoch']==min(h,key=lambda x:x['val_mae_normalized'])['epoch']
    if kind!='temporal': assert torch.allclose(m.adjacency().to_dense().sum(1),torch.ones(716),atol=1e-6)
    # Re-evaluate sample predictions from saved weights to verify analysis is using selected states.
    m.eval(); sample=np.load('results/predictions_sample.npz')
    xx=torch.from_numpy(((z['test'][:8]-z['mean'])/z['std']).astype('float32'))[None]
    with torch.no_grad(): predicted=m(xx).numpy()*z['std']+z['mean']
    assert np.allclose(predicted[0],sample[kind][0],atol=.001)
tex=pathlib.Path('report/PA3_Zain_Gulbaz_25280005.tex').read_text(); pdfpath='report/PA3_Zain_Gulbaz_25280005.pdf'
reader=PdfReader(pdfpath); text='\n'.join(p.extract_text() for p in reader.pages)
pathlib.Path('report/extracted_report.txt').write_text(text)
for term in ['Zain Gulbaz','25280005','https://github.com/ZainGulbaz/DLST-PA3','AI Usage Disclosure','Edits made by the student','OpenAI Codex','Claude','Q1','Q2','Q3','Q4','Q5','Q6','Q7']:
    assert term in text,term
assert '\\VerbatimInput[fontsize=\\scriptsize,breaklines=true,breakanywhere=true]{../TASK.md}' in tex
for term in ['TODO','TBD','INSERT HERE','???']: assert term not in text,term
figures=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex)
for fig in figures: assert (pathlib.Path('report')/fig).is_file()
image_count=sum(len(page.images) for page in reader.pages)
assert image_count>=len(figures),(image_count,len(figures))
fig_captions=len(re.findall(r'Figure \d+:',text)); table_captions=len(re.findall(r'Table \d+:',text))
assert fig_captions>=len(figures),(fig_captions,len(figures))
assert table_captions>=tex.count('\\begin{table}')+1
prompt=text.split('Exact supplied prompt')[-1]
normalized=lambda value:re.sub(r'\s+','',value)
assert all(normalized(line) in normalized(prompt) for line in pathlib.Path('TASK.md').read_text().splitlines() if line.strip())
buildlog=pathlib.Path('logs/report_build.log').read_text()
assert 'Missing character:' not in buildlog, 'Missing glyphs in report'
assert 'error:' not in buildlog.lower(), 'LaTeX build errors'
summary={'status':'PASS','pages':len(reader.pages),'figure_references':len(figures),'embedded_images':image_count,'figure_captions_in_pdf':fig_captions,'table_captions_in_pdf':table_captions,'task_sha256':hashlib.sha256(pathlib.Path('TASK.md').read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pathlib.Path(pdfpath).read_bytes()).hexdigest(),'checks':['train-only normalization','split boundaries and timestamp continuity','all 716 node IDs','fixed graph unchanged','row normalization','metric pooling and entry counts','best-validation checkpoint selection','saved prediction consistency','PDF text, images, tables and disclosure','verbatim TASK.md inclusion','every original TASK.md line found in PDF after whitespace normalization','no missing glyphs or LaTeX errors']}
json.dump(summary,open('results/audit.json','w'),indent=2); print(json.dumps(summary,indent=2))
