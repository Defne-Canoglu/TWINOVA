"""Secondary checks on actual saved predictions and quality-gated replays."""
import json,copy,time
import numpy as np,pandas as pd,joblib
from v133.model import boot
from v135.replay import run
from v135.metrics import score
from v135.data import get_measurement
from .research import enriched_study,frame,fit,calibrate_intervals,VARIANTS
from .common import ROOT,REPORT,write,now
from .application import Application
from .scenarios import CATALOG

def run_checks():
 study,h=enriched_study(ROOT/'data_v135/study');base=joblib.load(ROOT/'models_v135/study/twin.joblib');rows=[];gains=[];personal=[];learning=[]
 for stage in ['before','after']:
  for name in VARIANTS:
   f=pd.read_csv(REPORT/f'{name}_{stage}_predictions.csv');ids=f.person.str.extract(r'(\d+)$')[0].astype(int);groups={'nominal':ids%4==0,'sleep_restriction':(ids%4==2)&(f.issued_h<192),'recovery':(ids%4==2)&(f.issued_h>=192),'phase_shift':ids%4==1,'irregular_light':ids%4==3,'low_history':f.history_count<3}
   for condition,mask in groups.items():
    z=f[mask];rows.append(dict(model=name,stage=stage,condition=condition,scope='Existing frozen schedule stratum; irregular sleep/light schedule is regime 3; no causal claim',**score(z)))
  ref=pd.read_csv(REPORT/f'history_{stage}_predictions.csv').set_index(['person','task']);hp=pd.read_csv(REPORT/f'history_physiology_{stage}_predictions.csv').set_index(['person','task'])
  for condition,mask in groups.items():
   sub=f.loc[mask,['person','task']];idx=pd.MultiIndex.from_frame(sub);r=ref.loc[idx];p=hp.loc[idx];dif=(abs(p.target_rt-p.prediction_ms)-abs(r.target_rt-r.prediction_ms)).groupby(level='person').mean();ci=boot(dif)
   gains.append(dict(condition=condition,stage=stage,people=len(dif),tasks=len(p),HP_minus_H_ms=float(dif.mean()),ci95_low=ci[0],ci95_high=ci[1]))
 pd.DataFrame(rows).to_csv(REPORT/'condition_results.csv',index=False);pd.DataFrame(gains).to_csv(REPORT/'condition_physiology_differences.csv',index=False)
 ref=pd.read_csv(REPORT/'personal_history_predictions.csv').set_index(['person','task'])
 for mode in ['history','personal','dynamic']:
  p=pd.read_csv(REPORT/f'personal_{mode}_predictions.csv');q=p.set_index(['person','task']);dif=(abs(q.target_rt-q.prediction_ms)-abs(ref.target_rt-ref.prediction_ms)).groupby(level='person').mean();ci=boot(dif)
  for person,v in dif.items():personal.append(dict(mode=mode,person=person,difference_ms=v,mean_difference_ms=float(dif.mean()),ci95_low=ci[0],ci95_high=ci[1]))
  for lo,hi in [(0,3),(3,6),(6,12),(12,28)]:
   z=p[(p.history_count>=lo)&(p.history_count<hi)];offsets=z.wake_rate_offset.to_numpy();learning.append(dict(mode=mode,history_min=lo,history_max_exclusive=hi,identified_fraction=float(z.identified_wake_rate.mean()),offset_min=float(offsets.min()),offset_max=float(offsets.max()),**score(z)))
 pd.DataFrame(personal).to_csv(REPORT/'personal_paired_differences.csv',index=False);pd.DataFrame(learning).to_csv(REPORT/'personal_learning_curve.csv',index=False)
 # Full outputs retained for corrupted current measurements; do not score them as valid after-measurement outcomes.
 quality=[];outputs=[]
 for name in VARIANTS:
  b=joblib.load(ROOT/'models_v136'/f'{name}.joblib')
  for fault in ['none','device_delay','clock_drift','stale_calibration','delayed_input']:
   _,be,af=run(study,b,base['splits']['test'],{'kind':'always'},'long',.18,'constant',scenario=None if fault=='none' else fault)
   eligible=[o for o in af if o['interpretable']and o['stage']=='after'];quality.append(dict(model=name,fault=fault,total_decisions=len(be),requested_after=len(be),actual_after=len(af),no_after_result=len(be)-len(af),valid_after=len(eligible),C=sum(o['state']=='C'for o in af),fallback_before=sum(o['stage']=='before'for o in af),alarm_count=sum(o['alarm']for o in af),scope='Measurement-quality stress; no acceptance of invalid acquisition for accuracy'))
   outputs.extend(dict(model=name,fault=fault,person=o['person'],task=o['task'],state=o['state'],actual_stage=o['stage'],interpretable=o['interpretable'],prediction_ms=o['prediction_ms'],alarm=o['alarm'])for o in af)
 pd.DataFrame(quality).to_csv(REPORT/'measurement_quality_results.csv',index=False);pd.DataFrame(outputs).to_csv(REPORT/'measurement_quality_outputs.csv.gz',index=False)
 # Original interval alternatives use the same frozen prediction bundles and roles; diagnostic only.
 intervals=[]
 for method in ['constant','normalized','conditional','online']:
  _,be,af=run(study,base,base['splits']['test'],base['policy'],base['duration']['mode'],base['duration']['precision'],method)
  for stage,outs in [('before',be),('after',af)]:
   z=frame(study,outs,stage);z=z[z.interpretable&(z.stage==stage)]
   if stage=='after':
    durations={(o['person'],o['task']):get_measurement(study,(o['person'],o['task']),base['duration']['mode'],base['duration']['precision'])['duration_min']for o in outs if o['stage']=='after'}
    z['duration_min']=[durations[p,t]for p,t in zip(z.person,z.task)]
   intervals.append(dict(method=method,stage=stage,**score(z)))
 pd.DataFrame(intervals).to_csv(REPORT/'interval_method_results.csv',index=False)
 # Actual outputs/examples, no hand-coded state or score fixtures in report.
 app=Application();examples={}
 known=json.loads((ROOT/'config_v136/example_requests.json').read_text())
 for entry in known:
  out=app.run(entry['request']);examples[out['state']+'_'+out['stage']]=out
 for scenario in CATALOG:examples['scenario_'+scenario]=app.scenario(dict(person=base['splits']['test'][0],task='13-16',stage='after',lang='en',scenario=scenario))
 write(REPORT/'hypothesis_examples.json',examples)
 write(REPORT/'runtime_and_comms_tests.json',dict(local_runtime=app.health(),scenarios=3,one_way_s=1200,round_trip_s=2400,virtual_time=True,local_api_delayed=False,actual_browser_status='not_checked',browser_reason='This research command does not perform browser layout testing.',layout_screenshots_created=False,offline_scope='No external URLs needed by model/API or embedded guide; no OS network isolation was changed',completed_at=now()))
 return quality
if __name__=='__main__':run_checks()

def run_unseen():
 """Versioned secondary holdout: regression never sees late regime-3 combinations."""
 from .research import data_tables
 from v135.train import calibrate_probabilities
 study,history=enriched_study(ROOT/'data_v135/study');base=joblib.load(ROOT/'models_v135/study/twin.joblib');tables=data_tables(study,base);filtered={};removed={}
 for stage,t in tables.items():
  ids=t.person.str.extract(r'(\d+)$')[0].astype(int);mask=(ids%4==3)&(t.issued_h>=192)&t.person.isin(base['splits']['train']);filtered[stage]=t[~mask].copy();removed[stage]=int(mask.sum())
 write(REPORT/'unseen_combination_protocol.json',dict(scope='Secondary retrospective held-out regression-training combination; not a new person cohort',definition='regime 3 AND issued_h >=192 excluded from regression/classifier training',training_rows_removed=removed,calibration_roles_unchanged=True,alpha='Previously frozen main-analysis group-CV choice; no new tuning',interval='constant person-level',independent_new_test=False))
 results=[];saved={}
 selection=json.loads((REPORT/'research_selection.json').read_text())['choices']
 for name in ['history','history_physiology']:
  b=fit(study,base,filtered,name,base['splits']['train'],selection[name]['alpha']);calibrate_probabilities(study,b,base['splits']['probability']);b['alarm_threshold']=selection[name]['alarm_threshold'];calibrate_intervals(study,b,base['splits']['calibration']);_,be,af=run(study,b,base['splits']['test'],{'kind':'always'},'long',.18,'constant')
  for stage,outs in [('before',be),('after',af)]:
   z=frame(study,outs,stage);ids=z.person.str.extract(r'(\d+)$')[0].astype(int);z=z[(ids%4==3)&(z.issued_h>=192)&z.interpretable&(z.stage==stage)];saved[name,stage]=z;z.to_csv(REPORT/f'unseen_{name}_{stage}_predictions.csv',index=False);results.append(dict(model=name,stage=stage,condition='unseen_training_combination',**score(z)))
 differences=[]
 for stage in ['before','after']:
  a=saved['history',stage].set_index(['person','task']);b=saved['history_physiology',stage].set_index(['person','task']);d=(abs(b.target_rt-b.prediction_ms)-abs(a.target_rt-a.prediction_ms)).groupby(level='person').mean();ci=boot(d);differences.append(dict(stage=stage,people=len(d),tasks=len(a),HP_minus_H_ms=float(d.mean()),ci95_low=ci[0],ci95_high=ci[1]))
 pd.DataFrame(results).to_csv(REPORT/'unseen_combination_results.csv',index=False);pd.DataFrame(differences).to_csv(REPORT/'unseen_combination_differences.csv',index=False)
