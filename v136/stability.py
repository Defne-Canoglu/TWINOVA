"""Fixed-model calibration sensitivity, resampling independent person scores."""
import time
import joblib,numpy as np,pandas as pd
from v135.data import load
from v135.replay import run
from v134.models import conformal
from .common import ROOT,REPORT,write,now

def bootstrap_scores(scores,n,replicates=2000,seed=136501):
 s=np.asarray(scores,float)
 if s.ndim!=1 or len(s)==0 or not np.isfinite(s).all()or n<1:raise ValueError('Finite independent person scores required')
 rng=np.random.default_rng(seed);draw=s[rng.integers(0,len(s),size=(replicates,n))];k=int(np.ceil((n+1)*.9));q=np.sort(draw,axis=1)[:,k-1]if k<=n else np.full(replicates,np.inf)
 return q
def run_stability():
 start=time.monotonic();b=joblib.load(ROOT/'models_v135/study/twin.joblib');study=load(ROOT/'data_v135/study');_,be,af=run(study,b,b['splits']['calibration'],b['policy'],b['duration']['mode'],b['duration']['precision'],b['interval_method']);labels=study.labels.set_index(['person','task']);rows=[];summary=[];persons=[];influence=[]
 for stage,outs in [('before',be),('after',af)]:
  valid=[o for o in outs if o['interpretable']and o['stage']==stage];errors=np.array([abs(o['prediction_ms']-labels.loc[o['person'],o['task']].rt)for o in valid]);X=pd.DataFrame([o['model_features']for o in valid]);scale=np.maximum(3,b['intervals'][stage]['scale'].predict(X[b['models'][stage]['cols']]))
  for method,scores in [('constant',errors),('normalized',errors/scale)]:
   maximum=pd.DataFrame(dict(person=[o['person']for o in valid],score=scores)).groupby('person').score.max();baseline=conformal(maximum)
   for person,value in maximum.items():persons.append(dict(stage=stage,method=method,person=person,score=value))
   for n in sorted(set([4,10,15,len(maximum)])):
    if n>len(maximum):continue
    q=bootstrap_scores(maximum.to_numpy(),n);width=2*q*(float(np.mean(scale))if method=='normalized'else 1.)
    for i,(radius,w)in enumerate(zip(q,width)):rows.append(dict(stage=stage,method=method,n_people=n,replicate=i,quantile=radius,mean_width_ms=w,finite=bool(np.isfinite(radius))))
    finite=q[np.isfinite(q)];summary.append(dict(stage=stage,method=method,n_people=n,replicates=len(q),baseline_quantile=baseline,finite_fraction=float(np.isfinite(q).mean()),quantile_ci95=np.quantile(finite,[.025,.975]).tolist()if len(finite)else None,median_width_ms=float(np.median(width))if np.isfinite(width).all()else None,scope='Calibration resampling sensitivity, not a new coverage guarantee'))
   for person in maximum.index:
    q=conformal(maximum.drop(person));influence.append(dict(stage=stage,method=method,removed_person=person,quantile_without=q,baseline=baseline,change=q-baseline))
 pd.DataFrame(rows).to_csv(REPORT/'calibration_bootstrap_replicates.csv',index=False);pd.DataFrame(persons).to_csv(REPORT/'calibration_person_scores.csv',index=False);pd.DataFrame(influence).to_csv(REPORT/'calibration_leave_one_out.csv',index=False)
 cells=[dict(stage=s,cell=k,**v)for s,cal in b['intervals'].items()for k,v in cal.get('conditional',{}).items()]
 write(REPORT/'calibration_stability_summary.json',dict(results=summary,conditional_cells=cells,independent_people=len(b['splits']['calibration']),duplicates_retained=True,model_or_radius_function_changed=False,runtime_seconds=time.monotonic()-start,completed_at=now()));return summary
if __name__=='__main__':run_stability()
