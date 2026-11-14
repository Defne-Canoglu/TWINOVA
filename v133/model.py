import json,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import beta
class ConstantBinary:
 def __init__(self,p):self.p=float(p)
 def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))

from .data import BASE,AFTER,features,load

def ridge(d,cols,y='target_rt'):
 m=make_pipeline(StandardScaler(),Ridge(alpha=10));w=1/d.groupby('person').person.transform('size');m.fit(d[cols],d[y] if isinstance(y,str) else y,ridge__sample_weight=w/w.mean());return m

def posterior(past,model,now_h):
 # Only compatible past independent outcomes; process variance tracks drift.
 mean=0.;var=100.;last=None;used=[]
 for r in past.sort_values('target_available_h').itertuples():
  if not r.target_available_h<=now_h or now_h-r.target_available_h>72 or r.quality!='valid':continue
  if last is not None:var+=max(0,r.target_available_h-last)*.25
  q=pd.DataFrame([r._asdict()]);residual=r.target_rt-float(model.predict(q[AFTER])[0]);gain=var/(var+100.);mean+=gain*(residual-mean);var*=1-gain;last=r.target_available_h;used.append(dict(task=r.task,available_h=r.target_available_h))
 if last is not None:var+=max(0,now_h-last)*.25
 return dict(mean_ms=mean if len(used)>=2 else 0.,variance_ms2=var,accepted=len(used),outcomes=used)

def personal_predictions(d,model):
 general=model.predict(d[AFTER]);adjusted=[]
 for r,g in zip(d.itertuples(),general):
  past=d[(d.person==r.person)&(d.protocol==r.protocol)&(d.target_available_h<=r.issued_h)&(d.task!=r.task)];q=posterior(past,model,r.issued_h);adjusted.append(g+q['mean_ms'])
 return np.array(adjusted)
def boot(values,seed=133):
 a=np.asarray(values);r=np.random.default_rng(seed);draw=a[r.integers(0,len(a),(2000,len(a)))].mean(axis=1);return [float(np.quantile(draw,.025)),float(np.quantile(draw,.975))]
def radius(d,errors,nominal=.9):
 a=pd.DataFrame(dict(person=d.person,error=errors)).groupby('person').error.max().sort_values().to_numpy();k=math.ceil((len(a)+1)*nominal)
 if k>len(a):return math.inf
 return float(a[k-1])
def train(folder,out,report):
 out=Path(out);report=Path(report);out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 if (out/'twin.joblib').exists():raise ValueError('Use a new model output directory; never overwrite locked runs')
 m,d,a=features(folder)
 if len(m['splits']['train'])<4 or len(m['splits']['calibration'])<10:raise ValueError('Need >=4 training people and >=10 separate calibration people for this locked 90% protocol')
 d.to_csv(Path(folder)/'features.csv.gz',index=False);a.to_csv(report/'causal_feature_audit.csv',index=False);parts={k:d[d.person.isin(v)].copy() for k,v in m['splits'].items()};tr=parts['train'];before=ridge(tr,BASE);after=ridge(tr,AFTER);lapse=ridge(tr,AFTER,tr.target_lapses/tr.target_n)
 # Cross-person gain labels avoid in-sample optimism.
 labels=np.zeros(len(tr));y0=np.zeros(len(tr));y1=np.zeros(len(tr));folds=[]
 for fit,held in GroupKFold(4).split(tr,groups=tr.person):
  x=tr.iloc[fit];y=tr.iloc[held];m0=ridge(x,BASE);m1=ridge(x,AFTER);y0[held]=m0.predict(y[BASE]);y1[held]=m1.predict(y[AFTER]);folds.append(dict(fit=sorted(x.person.unique()),label=sorted(y.person.unique())))
 gains=abs(tr.target_rt.to_numpy()-y0)-abs(tr.target_rt.to_numpy()-y1);labels=(gains>10).astype(int)
 risk=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=133)) if len(np.unique(labels))>1 else ConstantBinary((labels.sum()+1)/(len(labels)+2))
 if len(np.unique(labels))>1:risk.fit(tr[BASE],labels)
 prob=parts['probability'];cal_labels=(abs(prob.target_rt-before.predict(prob[BASE]))-abs(prob.target_rt-after.predict(prob[AFTER]))>10).astype(int);raw=np.clip(risk.predict_proba(prob[BASE])[:,1],1e-6,1-1e-6)
 platt=LogisticRegression(C=1,random_state=133).fit(np.log(raw/(1-raw)).reshape(-1,1),cal_labels) if cal_labels.nunique()>1 else ConstantBinary((cal_labels.mean()*prob.person.nunique()+1)/(prob.person.nunique()+2))
 def risks(z):
  p=np.clip(risk.predict_proba(z[BASE])[:,1],1e-6,1-1e-6);return platt.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
 po=parts['policy'];pp=personal_predictions(po,after);diff=pd.DataFrame(dict(person=po.person,difference=abs(po.target_rt-pp)-abs(po.target_rt-after.predict(po[AFTER])))).groupby('person').difference.mean();ci=boot(diff);enabled=ci[1]<0
 curves=[];pr=risks(po);p0=before.predict(po[BASE]);p1=after.predict(po[AFTER]);D=abs(po.target_rt-p0)-abs(po.target_rt-p1)
 for threshold in [0,.05,.1,.2,.3,.4,.6,1.]:
  measure=pr>=threshold;skip=~measure;harm=skip&(D>10);rates=pd.DataFrame(dict(person=po.person,harm=harm.astype(float),skip=skip.astype(float))).groupby('person')[['harm','skip']].sum();eligible_rates=rates[rates.skip>0];upper=max(boot((eligible_rates.harm/eligible_rates.skip))[1],(1. if (eligible_rates.harm>0).all() else float(beta.ppf(.95,(eligible_rates.harm>0).sum()+1,len(eligible_rates)-(eligible_rates.harm>0).sum())))) if len(eligible_rates) else None
  curves.append(dict(threshold=threshold,measure_rate=float(measure.mean()),skipped=int(skip.sum()),harmful_skipped=int(harm.sum()),harm_per_skipped=float(harm.sum()/skip.sum()) if skip.sum() else None,cluster_upper=upper,MAE_ms=float(np.mean(abs(po.target_rt-np.where(measure,p1,p0))))))
 eligible=[c for c in curves if c['cluster_upper'] is None or c['cluster_upper']<=.1];chosen=min(eligible,key=lambda q:(q['measure_rate'],q['MAE_ms']))
 ca=parts['calibration'];q0=radius(ca,abs(ca.target_rt-before.predict(ca[BASE])));q1=radius(ca,abs(ca.target_rt-after.predict(ca[AFTER])));pca=personal_predictions(ca,after);qp=radius(ca,abs(ca.target_rt-pca))
 bundle=dict(version='13.3.0',before=before,after=after,lapse=lapse,risk=risk,platt=platt,personal_enabled=enabled,radii={'before':q0,'after':q1,'personal':qp},threshold=chosen['threshold'],source=m['source'],origin=m['origin'],population=m['population'],manifest_sha256=hashlib.sha256((Path(folder)/'manifest.json').read_bytes()).hexdigest(),files=m['files'],feature_sha256=hashlib.sha256((Path(folder)/'features.csv.gz').read_bytes()).hexdigest(),splits=m['splits'],features=BASE,after_features=AFTER)
 joblib.dump(bundle,out/'twin.joblib');(report/'fit_protocol.json').write_text(json.dumps(dict(version='13.3.0',seed=m.get('seed'),splits=m['splits'],alpha=10,forger_population_prior=True,personal_enabled=enabled,personal_policy_difference=float(diff.mean()),personal_policy_ci95=ci,policy=chosen,delta_ms=10,engineering_thresholds=True,probability_calibration_classes=int(cal_labels.nunique()),probability_single_class_fallback=cal_labels.nunique()==1,policy_bound='max of person-bootstrap mean harm and one-sided exact probability of any harmful skip per person; conservative exploratory bound',gain_label_folds=folds,radii=bundle['radii'],origin=m['origin'],population=m['population'],human_validation=False,physical_validation=False),indent=2));pd.DataFrame(curves).to_csv(report/'policy_candidates.csv',index=False)
 return bundle

def evaluate(folder,models,report):
 report=Path(report);report.mkdir(parents=True,exist_ok=True);b=joblib.load(Path(models)/'twin.joblib');m,h,t=load(folder)
 if b['manifest_sha256']!=hashlib.sha256((Path(folder)/'manifest.json').read_bytes()).hexdigest() or b['feature_sha256']!=hashlib.sha256((Path(folder)/'features.csv.gz').read_bytes()).hexdigest():raise ValueError('model/data provenance mismatch; train a new study version')
 d=pd.read_csv(Path(folder)/'features.csv.gz');te=d[d.person.isin(b['splits']['test'])].copy();preds={'before':b['before'].predict(te[BASE]),'after':b['after'].predict(te[AFTER]),'personal_candidate':personal_predictions(te,b['after']),'last_completed':te.last_rt.to_numpy(),'personal_mean':te.history_mean.to_numpy()};rows=[];personrows=[];outputs=[]
 for name,p in preds.items():
  q=pd.DataFrame(dict(person=te.person,error=abs(te.target_rt-p)));pm=q.groupby('person').error.mean();interval=boot(pm);rows.append(dict(model=name,people=te.person.nunique(),tasks=len(te),MAE_ms=float(pm.mean()),ci95_low=interval[0],ci95_high=interval[1]));personrows.extend(dict(model=name,person=i,MAE_ms=v) for i,v in pm.items());outputs.extend(dict(person=r.person,task=r.task,model=name,prediction_ms=float(v),target_ms=r.target_rt) for r,v in zip(te.itertuples(),p))
 pd.DataFrame(rows).to_csv(report/'model_results.csv',index=False);pd.DataFrame(personrows).to_csv(report/'person_results.csv',index=False);pd.DataFrame(outputs).to_csv(report/'test_predictions.csv',index=False)
 intervals=[]
 for name in ['before','after']:
  e=abs(te.target_rt-preds[name]);q=b['radii'][name];g=pd.DataFrame(dict(person=te.person,covered=e<=q)).groupby('person').covered.all();intervals.append(dict(model=name,width_ms=2*q,task_coverage=float(np.mean(e<=q)),person_all_tasks_coverage=float(g.mean()),nominal=.9,calibration_people=10))
 pd.DataFrame(intervals).to_csv(report/'interval_results.csv',index=False);lp=np.clip(b['lapse'].predict(te[AFTER]),0,1);(report/'lapse_result.json').write_text(json.dumps(dict(MAE_fraction=float(np.mean(abs(te.target_lapses/te.target_n-lp))),scope='independent binomial lapse fraction; separate from median RT')))
 return rows
