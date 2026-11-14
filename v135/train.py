import copy,json,hashlib,math,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from scipy.special import logit
from v133.data import BASE
from v133.model import ConstantBinary,boot
from v134.models import conformal
from .engine import Engine,AFTER,TARGETS,SHORT
from .data import load,get_measurement
from .replay import run
from .metrics import score

class ZeroGain:
 def predict(self,x):return np.zeros(len(x))
class CalibratedEvent:
 def __init__(self,model,cols,calibrator=None):self.model=model;self.cols=cols;self.calibrator=calibrator
 def predict_proba(self,x):
  p=np.clip(self.model.predict_proba(x[self.cols])[:,1],1e-6,1-1e-6)
  return self.calibrator.predict_proba(logit(p).reshape(-1,1)) if self.calibrator is not None else np.column_stack([1-p,p])

def dump(path,value):
 def clean(v):
  if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,np.generic):v=v.item()
  if isinstance(v,float) and not math.isfinite(v):return None
  return v
 Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False))

def matrix(study,people,duration='long',precision=.18):
 d=study.observations[study.observations.person.isin(people)].copy();rows=[]
 for r in d.to_dict('records'):
  m=get_measurement(study,(r['person'],r['task']),duration,precision);rows.append({**r,**{k:v for k,v in m.items() if k!='duration_min'}})
 return pd.DataFrame(rows).merge(study.labels,on=['person','task'],suffixes=('','_label'),validate='one_to_one')

def regression(name,x,y):
 if name=='ridge_10':return make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x,y)
 return MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=7,l2_regularization=25,learning_rate=.07,random_state=135)).fit(x,y)

def classification(name,x,y):
 if len(np.unique(y))<2:return ConstantBinary((np.sum(y)+1)/(len(y)+2))
 if name=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=1000,random_state=135)).fit(x,y)
 return HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=5,l2_regularization=25,random_state=135).fit(x,y)

def fit_bundle(study,people,name='ridge_10',classifier='logistic',names=None):
 d=matrix(study,people);names=names or [k for k in TARGETS if d[k].notna().all()];b=dict(version='13.5.0',models={},population_models={},events={},event_cols={},target_names=names,gain=ZeroGain(),gain_probability=ConstantBinary(.5),alarm_threshold=.5,policy={'kind':'always'},history_mode='history',interval_method='constant',intervals={},personal_config={'intercept_var':9,'slope_var':25,'noise_var':144,'decay_h':48})
 for stage,cols in [('before',BASE),('after',AFTER)]:
  stage_data=d if stage=='before' or not study.trial_frames else pd.concat([matrix(study,people,'short'),matrix(study,people,'adaptive',.18),d],ignore_index=True)
  b['models'][stage]=dict(model=regression(name,stage_data[cols],stage_data[names]),cols=cols)
  pop=[c for c in cols if c not in ['last_rt','history_mean','history_n']];b['population_models'][stage]=dict(model=regression(name,stage_data[pop],stage_data[names]),cols=pop)
  b['events'][stage]=CalibratedEvent(classification(classifier,stage_data[cols],(stage_data.lapse_fraction>=.1).astype(int)),cols);b['event_cols'][stage]=cols
 return b

def calibrate_probabilities(study,b,people):
 # Raw predictions use the same engine and acquisition sequence as subsequent inference.
 z,be,af=run(study,b,people,{'kind':'always'},'long')
 labels=study.labels.set_index(['person','task'])
 for stage,outs in [('before',be),('after',af)]:
  x=np.array([v['event_probability'] for v in outs]);y=np.array([labels.loc[v['person'],v['task']].lapse_fraction>=.1 for v in outs]);ids=len(set(v['person'] for v in outs));cal=LogisticRegression(C=.3,random_state=135).fit(logit(np.clip(x,1e-6,1-1e-6)).reshape(-1,1),y) if len(np.unique(y))>1 else ConstantBinary((y.mean()*ids+1)/(ids+2));b['events'][stage].calibrator=cal
 return z

def interval_fit(study,b,people):
 z,be,af=run(study,b,people,{'kind':'always'},'long');lookup=study.labels.set_index(['person','task']);b['intervals']={}
 for stage,outs in [('before',be),('after',af)]:
  errors=np.array([abs(v['prediction_ms']-lookup.loc[v['person'],v['task']].rt) for v in outs]);ids=[v['person'] for v in outs];X=pd.DataFrame([v['model_features'] for v in outs]);cols=b['models'][stage]['cols'];scale=b['error_scales'][stage];sc=np.maximum(3,scale.predict(X[cols]));constant=conformal(pd.DataFrame({'person':ids,'e':errors}).groupby('person').e.max());normalized=conformal(pd.DataFrame({'person':ids,'e':errors/sc}).groupby('person').e.max());cal=dict(constant=constant,scale=scale,normalized_q=normalized,conditional={})
  for am in ['am','pm']:
   for high in [0,1]:
    # Four cells use Bonferroni alpha=.025 each. Small cells fall back to global.
    mask=np.array([('am' if v['issued_h']%24<12 else 'pm')==am and int(v['event_probability']>=.5)==high for v in outs]);personscore=pd.DataFrame({'person':np.array(ids)[mask],'e':errors[mask]}).groupby('person').e.max();q=conformal(personscore,.975) if len(personscore) else math.inf;eligible=len(personscore)>=19 and np.isfinite(q);cal['conditional'][f'{am}:{high}']=dict(people=len(personscore),eligible=bool(eligible),radius=q if eligible else constant,reason='four-cell Bonferroni; insufficient independent calibration people or infinite finite-sample quantile' if not eligible else None)
  b['intervals'][stage]=cal


def train(folder,out,report,protocol='config_v135/protocol.json'):
 folder=Path(folder);out=Path(out);report=Path(report);out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
 if (out/'twin.joblib').exists():raise ValueError('Refuse overwrite of frozen model')
 config=json.loads(Path(protocol).read_text());dump(report/'preregistered_protocol.json',config);dump(report/'input_lock.json',dict(manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),test_opened=False));print('Input/protocol locked; loading trials and causal environmental state',flush=True)
 study=load(folder);splits=study.manifest['splits'];tr=splits['train'];po=splits['policy'];prob=splits['probability'];ca=splits['calibration'];names=[k for k in TARGETS if study.labels[k].notna().all()]
 if len(tr)<4 or len(ca)<10:raise ValueError('Need >=4 training and >=10 separate calibration people')
 study.audit.to_csv(report/'causal_feature_audit.csv',index=False);cv=[];predictions={};folds=[]
 for name in config['candidates']:
  for classifier in config['classifier_candidates']:
   allz=[]
   for f,v in GroupKFold(3).split(tr,groups=tr):
    trainids=[tr[i] for i in f];validids=[tr[i] for i in v];b=fit_bundle(study,trainids,name,classifier,names);z,be,af=run(study,b,validids,{'kind':'always'},'long');allz.append(z)
    if name==config['candidates'][0] and classifier==config['classifier_candidates'][0]:folds.append(dict(fit=trainids,held=validids))
   z=pd.concat(allz);s=score(z,False);cv.append(dict(regression=name,classifier=classifier,MAE_ms=s['MAE_ms'],Brier=s['Brier'],PR_AUC=s['PR_AUC']));predictions[name,classifier]=z;print('CV',name,classifier,round(s['MAE_ms'],3),round(s['Brier'],4),flush=True)
 pd.DataFrame(cv).to_csv(report/'candidate_cv.csv',index=False);dump(report/'fold_audit.json',folds);winner=min(cv,key=lambda v:v['Brier']+.01*v['MAE_ms']);b=fit_bundle(study,tr,winner['regression'],winner['classifier'],names);b['selected_models']=winner
 calibrate_probabilities(study,b,prob)
 # Train gain from cross-person predictions, never from in-sample residuals.
 oof=predictions[winner['regression'],winner['classifier']];trainmat=matrix(study,tr);merged=trainmat.merge(oof[['person','task','before_prediction_ms','prediction_ms']],on=['person','task']);gain=abs(merged.rt-merged.before_prediction_ms)-abs(merged.rt-merged.prediction_ms);b['gain']=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(merged[BASE],gain)
 pro=matrix(study,prob);pz,_,_=run(study,b,prob,{'kind':'always'},'long');pj=pro.merge(pz[['person','task','before_prediction_ms','prediction_ms']],on=['person','task']);gain_label=(abs(pj.rt-pj.before_prediction_ms)-abs(pj.rt-pj.prediction_ms)>10).astype(int);rawgain=b['gain'].predict(pj[BASE]);b['gain_probability']=LogisticRegression(C=.3).fit(rawgain.reshape(-1,1),gain_label) if gain_label.nunique()>1 else ConstantBinary((gain_label.mean()*len(prob)+1)/(len(prob)+2))
 # OOF-derived error scales; final quantiles use separate calibration people.
 b['error_scales']={}
 for stage,cols in [('before',BASE),('after',AFTER)]:
  er=abs(merged.rt-(merged.before_prediction_ms if stage=='before' else merged.prediction_ms));b['error_scales'][stage]=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(merged[cols],er)
 interval_fit(study,b,ca)
 # Duration choice uses only policy people; checkpoint rule sees acquired prefixes.
 durations=[]
 duration_candidates=[('short',.18),('long',.18),('adaptive',.12),('adaptive',.18),('adaptive',.25)] if study.trial_frames else [('summary',.18)]
 for mode,pr in duration_candidates:
  z,_,_=run(study,b,po,{'kind':'always'},mode,pr);s=score(z,False);durations.append(dict(mode=mode,precision=pr,**s))
 long=next((x for x in durations if x['mode']=='long'),durations[0]);acceptable=[x for x in durations if x['MAE_ms']<=long['MAE_ms']+1 and x['Brier']<=long['Brier']+.02];duration=min(acceptable,key=lambda v:(v['total_measurement_minutes'] if v['total_measurement_minutes'] is not None else math.inf,v.get('measure_rate',1)));b['duration']=dict(mode=duration['mode'],precision=duration['precision']);pd.DataFrame(durations).to_csv(report/'duration_selection.csv',index=False)
 # Shared-engine three-way personalization; additional parameter adaptation must beat history.
 personal=[];history_z,_,_=run(study,b,po,{'kind':'always'},b['duration']['mode'],b['duration']['precision'],history='history')
 for cfg in config['personal_configs']:
  b['personal_config']=cfg;z,_,_=run(study,b,po,{'kind':'always'},b['duration']['mode'],b['duration']['precision'],history='personal');a=history_z.set_index(['person','task']);p=z.set_index(['person','task']);diff=pd.DataFrame({'person':z.person,'diff':(abs(p.target_rt-p.prediction_ms)-abs(a.target_rt-a.prediction_ms)).to_numpy()}).groupby('person')['diff'].mean();personal.append(dict(config=cfg,difference_ms=float(diff.mean()),ci95=boot(diff)))
 best=min(personal,key=lambda v:v['difference_ms']);b['personal_config']=best['config'];b['history_mode']='personal' if best['ci95'][1]<0 else 'history';dump(report/'personal_selection.json',dict(candidates=personal,selected=b['history_mode'],selected_config=b['personal_config']))
 # Alarm threshold chosen BEFORE test, with sensitivity first and false-positive cost.
 z,_,_=run(study,b,po,{'kind':'always'},b['duration']['mode'],b['duration']['precision']);alarms=[]
 for threshold in config['alarm_thresholds']:
  zz=z.copy();zz.alarm=zz.event_probability>=threshold;alarms.append(dict(threshold=threshold,**score(zz)))
 eligible=[x for x in alarms if x['sensitivity'] is not None and x['sensitivity']>=.9 and x['false_positive_rate'] is not None and x['false_positive_rate']<=.2];chosen=min(eligible,key=lambda v:v['false_positive_rate']) if eligible else min(alarms,key=lambda v:max(0,.9-(v['sensitivity'] or 0))*5+max(0,(v['false_positive_rate'] if v['false_positive_rate'] is not None else 1.)-.2));b['alarm_threshold']=chosen['threshold'];pd.DataFrame(alarms).to_csv(report/'alarm_selection.csv',index=False)
 # Risk-first policy / budgets. Cuts are fixed from policy-group before scores.
 z,be,_=run(study,b,po,{'kind':'always'},b['duration']['mode'],b['duration']['precision']);baseline=score(z);risk=np.array([v['event_probability'] for v in be]);gains=np.array([v['measurement_gain_ms'] for v in be]);policies=[]
 for kind in config['policy']['types']:
  for budget in config['policy']['budgets']:
   if kind=='always' and budget!=1:continue
   cut=-math.inf if budget==1 else float(np.quantile(gains if kind=='gain' else risk,1-budget));pol=dict(kind=kind,budget=budget,cut=cut,cold_start=True);zz,_,_=run(study,b,po,pol,b['duration']['mode'],b['duration']['precision']);s=score(zz);point=s['sensitivity'] is not None and s['sensitivity']>=.9 and s['false_positive_rate'] is not None and s['false_positive_rate']<=.2 and s['MAE_ms']<=baseline['MAE_ms']+1;ci=s['sensitivity_cluster_ci95'];strong=point and ci is not None and ci[0]>=.9 and s['false_positive_cluster_ci95'][1]<=.2;policies.append(dict(**pol,**s,point_pass=point,evidence_pass=strong))
 pd.DataFrame(policies).to_csv(report/'policy_selection.csv',index=False);valid=[v for v in policies if v['kind'] in ('always','risk') and v['point_pass']];pick=min(valid,key=lambda v:(v['total_measurement_minutes'] if v['total_measurement_minutes'] is not None else math.inf,v.get('measure_rate',1))) if valid else next(v for v in policies if v['kind']=='always');b['policy']={k:pick[k] for k in ['kind','budget','cut','cold_start']};b['policy_point_accepted']=bool(valid);b['policy_evidence_accepted']=pick['evidence_pass']
 # Intervals calibrated again under SELECTED acquisition history and duration.
 interval_fit_policy(study,b,ca)
 introws=[]
 for method in config['intervals']:
  zz,_,_=run(study,b,po,b['policy'],b['duration']['mode'],b['duration']['precision'],method);introws.append(dict(method=method,**score(zz,False)))
 validint=[v for v in introws if v['person_all_tasks_coverage']>=.9];b['interval_method']=min(validint or introws,key=lambda v:v['interval_score'])['method'];pd.DataFrame(introws).to_csv(report/'interval_selection.csv',index=False)
 # Nested learning and numerical stopping, same development people, never test.
 learning=[];previous=None;stopped=False
 for count in sorted(set(min(k,len(tr)) for k in [8,40,72])):
  bb=fit_bundle(study,tr[:count],winner['regression'],winner['classifier'],names);calibrate_probabilities(study,bb,prob);bb['error_scales']=b['error_scales'];interval_fit(study,bb,ca);zz,_,_=run(study,bb,po,{'kind':'always'},'long');s=score(zz,False);stable=False
  if previous:stable=abs(s['MAE_ms']-previous['MAE_ms'])<.15 and abs(s['person_all_tasks_coverage']-previous['person_all_tasks_coverage'])<=.05 and s['sensitivity'] is not None and previous['sensitivity'] is not None and abs(s['sensitivity']-previous['sensitivity'])<=.05 and s['false_positive_rate'] is not None and previous['false_positive_rate'] is not None and abs(s['false_positive_rate']-previous['false_positive_rate'])<=.05
  learning.append(dict(training_people=count,total_people=count+sum(len(v) for k,v in splits.items() if k!='train'),stable=stable,**s));previous=s
  if stable:stopped=True;break
 pd.DataFrame(learning).to_csv(report/'learning_curve.csv',index=False);dump(report/'stopping.json',dict(reason='numerical_stability' if stopped else '128_person_budget_cap',stability_achieved=stopped,max_people=128))
 b.update(splits=splits,source=study.manifest['source'],origin=study.manifest['origin'],population=study.manifest['population'],data_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),protocol=config,training_origin_not_a_feature=True)
 joblib.dump(b,out/'twin.joblib');dump(report/'selection_frozen.json',dict(model_sha256=hashlib.sha256((out/'twin.joblib').read_bytes()).hexdigest(),selected_models=winner,policy=b['policy'],duration=b['duration'],interval=b['interval_method'],history=b['history_mode'],alarm_threshold=b['alarm_threshold'],policy_point_accepted=b['policy_point_accepted'],policy_evidence_accepted=b['policy_evidence_accepted'],test_opened=False));print('SELECTION FROZEN — test may now be evaluated',flush=True)
 return study,b

def interval_fit_policy(study,b,people):
 # Calibration follows selected acquisition history; stage-specific prediction records.
 z,be,af=run(study,b,people,b['policy'],b['duration']['mode'],b['duration']['precision']);lookup=study.labels.set_index(['person','task'])
 for stage,outs in [('before',be),('after',af)]:
  if not outs:continue
  errs=np.array([abs(v['prediction_ms']-lookup.loc[v['person'],v['task']].rt) for v in outs]);ids=[v['person'] for v in outs];cal=b['intervals'][stage];cal['constant']=conformal(pd.DataFrame({'person':ids,'e':errs}).groupby('person').e.max());X=pd.DataFrame([v['model_features'] for v in outs]);sc=np.maximum(3,cal['scale'].predict(X[b['models'][stage]['cols']]));cal['normalized_q']=conformal(pd.DataFrame({'person':ids,'e':errs/sc}).groupby('person').e.max())
 # Conditional cells remain conservatively calibrated from all-measure independent
 # people; insufficient cell support always explicitly falls back to selected-history global.
 for cal in b['intervals'].values():
  for cell in cal['conditional'].values():
   if not cell['eligible']:cell['radius']=cal['constant']
