"""Preregistered, retrospective feature comparisons through Engine.decide.

All groups share the same valid tasks, fixed acquisition schedule and constant
person-level conformal method. Original frozen model remains the default.
"""
import copy,json,math,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LogisticRegression
from v133.model import ConstantBinary,boot
from v134.models import conformal
from v135.data import load,get_measurement
from v135.replay import run
from v135.metrics import score
from v135.train import CalibratedEvent,ZeroGain,calibrate_probabilities
from v135.engine import SHORT
from .features import derived_table,W_COLS,PUBLISHED
from .personal import WakeRateAdapter
from .common import ROOT,REPORT,write,sha,now,clean

GROUPS={'H':['last_rt','history_mean','history_n'],'K':['sin','cos'],'W':W_COLS,'P':['S','x','xc','n'],'E':['co2_excess','dose_ugy','gravity_deviation'],'T':SHORT,'Q':['quality','device_ok','calibration_id','light_coverage']}
VARIANTS={'constant':[],'history':['H'],'clock_sleep':['K','W'],'physiology':['P'],'history_physiology':['H','P'],'history_physiology_environment':['H','P','E'],'history_physiology_clock':['H','P','K'],'published_two_process':PUBLISHED}
from .reference import MeanTargets

def enriched_study(folder):
 study=load(folder);history=pd.read_csv(Path(folder)/'history.csv.gz',dtype={'person':str});d=derived_table(history,study.observations);s=copy.copy(study);s.observations=study.observations.merge(d,on=['person','task'],validate='one_to_one');return s,history

def frame(study,outs,stage):
 labels=study.labels.set_index(['person','task']);rows=[]
 for out in outs:
  label=labels.loc[out['person'],out['task']];r={k:v for k,v in out.items()if k not in ['model_features','physiology','environment','agents','posterior','targets']};r.update(target_rt=label.rt,target_lapse_fraction=label.lapse_fraction,event=bool(label.lapse_fraction>=.1),measured=stage=='after',measurement_attempted=stage=='after',duration_min=get_measurement(study,(out['person'],out['task']),'long',.18).get('duration_min') if stage=='after'else 0.,reference_duration_min=label.reference_duration_min)
  r.update({'target_'+k:label[k]for k in ['response_speed','slow_speed','early_fraction']});r.update({'pred_'+k:v for k,v in out['targets'].items()});r['identified_wake_rate']=out['posterior'].get('identified_wake_rate',False);r['wake_rate_offset']=out['posterior'].get('wake_rate_offset',0.);rows.append(r)
 return pd.DataFrame(rows)

def data_tables(study,base):
 people=sum(study.manifest['splits'].values(),[]);_,be,af=run(study,base,people,{'kind':'always'},'long',.18,'constant',history='history');tables={}
 for stage,outs in [('before',be),('after',af)]:
  feats=pd.DataFrame([dict(person=o['person'],task=o['task'],**o['model_features'])for o in outs if o['interpretable']and o['stage']==stage]);extra=[c for c in study.observations if c not in feats.columns or c in ['person','task']];x=feats.merge(study.observations[extra],on=['person','task'],validate='one_to_one');tables[stage]=x.merge(study.labels,on=['person','task'],suffixes=('','_label'),validate='one_to_one')
 return tables

def columns(name,stage):
 group=VARIANTS[name];cols=group if name=='published_two_process'else[c for g in group for c in GROUPS[g]]
 return list(cols)+(SHORT if stage=='after'else[])

def fit(study,base,tables,name,people,alpha):
 b=copy.deepcopy(base);b.update(models={},population_models={},events={},event_cols={},intervals={},interval_method='constant',history_mode='history',policy={'kind':'always'},gain=ZeroGain(),gain_probability=ConstantBinary(.5),alarm_threshold=.5)
 for stage in ['before','after']:
  d=tables[stage][tables[stage].person.isin(people)];cols=columns(name,stage);y=d[b['target_names']];event=(d.lapse_fraction>=.1).astype(int)
  if not cols:model=MeanTargets(y);classifier=ConstantBinary((event.sum()+1)/(len(event)+2))
  else:
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=alpha)).fit(d[cols],y)
   classifier=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.3,random_state=136,max_iter=1000)).fit(d[cols],event)if event.nunique()>1 else ConstantBinary((event.sum()+1)/(len(event)+2))
  b['models'][stage]=dict(model=model,cols=cols);b['population_models'][stage]=b['models'][stage];b['events'][stage]=CalibratedEvent(classifier,cols);b['event_cols'][stage]=cols
 return b

def calibrate_intervals(study,b,people,history='history'):
 _,be,af=run(study,b,people,{'kind':'always'},'long',.18,'constant',history=history);lookup=study.labels.set_index(['person','task'])
 for stage,outs in [('before',be),('after',af)]:
  records=[dict(person=o['person'],error=abs(o['prediction_ms']-lookup.loc[o['person'],o['task']].rt))for o in outs if o['interpretable']and o['stage']==stage]
  maximum=pd.DataFrame(records).groupby('person').error.max();b['intervals'][stage]={'constant':conformal(maximum)}

def evaluate():
 start=time.monotonic();REPORT.mkdir(exist_ok=True);(ROOT/'models_v136').mkdir(exist_ok=True)
 frozen=ROOT/'models_v135/study/twin.joblib';base=joblib.load(frozen);study,history=enriched_study(ROOT/'data_v135/study');splits=study.manifest['splits'];tables=data_tables(study,base)
 protocol=dict(seed=136221,alphas=[1.,10.,100.],folds=3,variants=VARIANTS,common_environment_in_main_comparison=False,current_measurement_group='T appended to every after model, absent before',interval_method='constant',nominal=.9,fixed_policy='always',duration='long',horizon_h=2,selection_objective='development Brier + .01*MAE',test_scope='Retrospective existing test, already examined in earlier versions',dynamic_acceptance=dict(min_improvement_ms=.15,paired_ci_upper_below=0.,max_person_p90_deterioration_ms=.5),default_model_policy='Original v135 retained; experimental selection cannot silently replace it')
 write(REPORT/'research_protocol.json',protocol);write(REPORT/'feature_groups.json',dict(groups=GROUPS,variants={n:{s:columns(n,s)for s in ['before','after']}for n in VARIANTS},imputation='W missing values use training-only median with explicit missing indicators; no observation is reclassified as normal',feature_roles='person/task/source/origin are routing metadata, not learned predictors'))
 results=[];cv=[];pairs=[];subgroups=[];predictions={};frozen_selection={};candidate_bundles={}
 trainids=splits['train'];folds=list(GroupKFold(3).split(trainids,groups=trainids))
 write(REPORT/'fold_audit.json',[dict(train=[trainids[i]for i in tr],validation=[trainids[i]for i in va])for tr,va in folds])
 for name in VARIANTS:
  print('Fitting comparison',name,flush=True);search=[10.]if name=='constant'else protocol['alphas'];scores=[]
  for alpha in search:
   zall=[]
   for tr,va in folds:
    b=fit(study,base,tables,name,[trainids[i]for i in tr],alpha);_,be,af=run(study,b,[trainids[i]for i in va],{'kind':'always'},'long',.18,'constant')
    for stage,outs in [('before',be),('after',af)]:
     z=frame(study,outs,stage);z=z[(z.interpretable)&(z.stage==stage)];z['stage_label']=stage;zall.append(z)
   zz=pd.concat(zall);s=score(zz,False);scores.append(dict(model=name,alpha=alpha,objective=s['Brier']+.01*s['MAE_ms'],MAE_ms=s['MAE_ms'],Brier=s['Brier']));cv.append(scores[-1])
  best=min(scores,key=lambda r:r['objective']);b=fit(study,base,tables,name,trainids,best['alpha']);calibrate_probabilities(study,b,splits['probability'])
  _,pbe,paf=run(study,b,splits['policy'],{'kind':'always'},'long',.18,'constant');policy=pd.concat([frame(study,pbe,'before'),frame(study,paf,'after')]);eligible=policy[policy.interpretable].copy();cuts=[]
  for threshold in [.05,.1,.15,.2,.3,.4,.5,.6]:
   z=eligible.copy();z.alarm=z.event_probability>=threshold;s=score(z,False);cost=5*max(0,.9-(s['sensitivity']or 0))+max(0,(s['false_positive_rate']if s['false_positive_rate']is not None else 1)-.2);cuts.append((cost,threshold))
  b['alarm_threshold']=min(cuts)[1];calibrate_intervals(study,b,splits['calibration']);candidate_bundles[name]=b;frozen_selection[name]=dict(alpha=best['alpha'],alarm_threshold=b['alarm_threshold'],cols={s:b['models'][s]['cols']for s in ['before','after']})
  write(REPORT/'research_selection.json',dict(choices=frozen_selection,test_used_for_selection=False,test_scope=protocol['test_scope']))
  _,be,af=run(study,b,splits['test'],{'kind':'always'},'long',.18,'constant')
  for stage,outs in [('before',be),('after',af)]:
   z=frame(study,outs,stage);total=len(z);z=z[z.interpretable&(z.stage==stage)].copy();z.to_csv(REPORT/f'{name}_{stage}_predictions.csv',index=False);predictions[name,stage]=z
   results.append(dict(model=name,stage=stage,total_decisions=total,valid_decisions=len(z),**score(z)))
   for scenario in ['nominal','sleep_restriction','recovery','phase_shift','irregular_light','low_history','unseen_combination','invalid_measurement']:
    ids=z.person.str.extract(r'(\d+)$')[0].astype(int);mask=np.ones(len(z),dtype=bool)
    if scenario=='sleep_restriction':mask=(ids%4==2)&(z.issued_h<192)
    elif scenario=='recovery':mask=(ids%4==2)&(z.issued_h>=192)
    elif scenario=='phase_shift':mask=(ids%4==1)&(z.issued_h>=192)
    elif scenario=='irregular_light':mask=ids%4==3
    elif scenario=='low_history':mask=z.history_count<3
    elif scenario=='unseen_combination':mask=(ids%4==3)&(z.issued_h>=192)
    elif scenario=='invalid_measurement':mask=np.zeros(len(z),dtype=bool)
    g=z[mask]
    subgroups.append(dict(model=name,stage=stage,condition=scenario,scope='existing frozen schedule stratum, not a causal intervention',**(score(g,False)if len(g)else dict(tasks=0,reason='Invalid decisions excluded from accuracy; see scenario tests and full outputs'))))
  joblib.dump(b,ROOT/'models_v136'/f'{name}.joblib',compress=3)
 # Existing intercept/S-slope adaptation and one new genuine wake-rate parameter.
 hp=candidate_bundles['history_physiology'];candidates={'history':hp,'personal':copy.deepcopy(hp),'dynamic':copy.deepcopy(hp)};candidates['dynamic']['personal_adapter']=WakeRateAdapter();personal=[];policy_diffs={};policy_replays={}
 for mode,b in candidates.items():
  calibrate_intervals(study,b,splits['calibration'],mode);_,be,af=run(study,b,splits['policy'],{'kind':'always'},'long',.18,'constant',history=mode);p=frame(study,af,'after');policy_replays[mode]=p[p.interpretable&(p.stage=='after')].copy()
 reference=policy_replays['history'].set_index(['person','task'])
 for mode,p in policy_replays.items():
  q=p.set_index(['person','task']);dif=(abs(q.target_rt-q.prediction_ms)-abs(reference.target_rt-reference.prediction_ms)).groupby(level='person').mean();policy_diffs[mode]=dict(mean_ms=float(dif.mean()),ci95=boot(dif),person_p90_ms=float(np.quantile(dif,.9)))
 decision=policy_diffs['dynamic'];accepted=decision['mean_ms']<=-.15 and decision['ci95'][1]<0 and decision['person_p90_ms']<=.5
 write(REPORT/'personal_development_selection.json',dict(development=policy_diffs,new_candidate_accepted=accepted,default_engine_model='v135 frozen',reason='Candidate evaluation is complete; default only changes after a separate versioned release decision',thresholds=protocol['dynamic_acceptance']))
 for mode,b in candidates.items():
  _,be,af=run(study,b,splits['test'],{'kind':'always'},'long',.18,'constant',history=mode);p=frame(study,af,'after');p=p[p.interpretable&(p.stage=='after')].copy();p.to_csv(REPORT/f'personal_{mode}_predictions.csv',index=False);personal.append(dict(mode=mode,identified_wake_rate_fraction=float(p.identified_wake_rate.mean()),**score(p)))
  if mode=='dynamic':joblib.dump(b,ROOT/'models_v136/wake_rate_candidate.joblib',compress=3)
 for stage in ['before','after']:
  ref=predictions['history',stage].set_index(['person','task'])
  for (name,s),p in predictions.items():
   if s!=stage:continue
   q=p.set_index(['person','task']);dif=(abs(q.target_rt-q.prediction_ms)-abs(ref.target_rt-ref.prediction_ms)).groupby(level='person').mean();ci=boot(dif)
   for person,value in dif.items():pairs.append(dict(model=name,stage=s,person=person,difference_ms=value,mean_difference_ms=float(dif.mean()),ci95_low=ci[0],ci95_high=ci[1]))
 pd.DataFrame(results).to_csv(REPORT/'ablation_results.csv',index=False);pd.DataFrame(cv).to_csv(REPORT/'candidate_cv.csv',index=False);pd.DataFrame(pairs).to_csv(REPORT/'paired_person_differences.csv',index=False);pd.DataFrame(subgroups).to_csv(REPORT/'condition_results.csv',index=False);pd.DataFrame(personal).to_csv(REPORT/'personal_parameter_results.csv',index=False);pd.DataFrame([r for r in results if r['model']=='published_two_process']).to_csv(REPORT/'published_baseline_results.csv',index=False)
 summary=dict(comparisons=results,personal=personal,personal_development=policy_diffs,default_model_changed=False,default_model_sha256=sha(frozen),runtime_seconds=time.monotonic()-start,completed_at=now(),test_scope=protocol['test_scope'],baseline_reference='Borbely two-process dynamics with observed sleep (tau_sleep=4.2h,tau_wake=18.2h); sinusoidal C, fixed acrophase 16h; RT/event mapping learned only on training roles. Not SAFTE or complete Akerstedt three-process model; no sleep-inertia component is claimed.')
 write(REPORT/'research_summary.json',summary);return study,base,summary
if __name__=='__main__':evaluate()
