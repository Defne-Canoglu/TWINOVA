"""Acquisition orchestration; all numerical decisions go through Engine.decide."""
import copy
import numpy as np,pandas as pd
from .engine import Engine,BASE,SHORT
from .data import get_measurement
from v133.hardware import VirtualRig,Device

def run(study,bundle,people,policy=None,duration='adaptive',precision=.18,interval='constant',history=None,scenario=None,before_only_task=None):
 engine=Engine(bundle);obs=study.observations[study.observations.person.isin(people)].copy();labels=study.labels.set_index(['person','task']);pending=[];outputs=[];before_rows=[];after_rows=[];pol=policy or bundle.get('policy',{'kind':'always'})
 if scenario in ('sensor_noise','short_missing','long_missing','delayed_input','device_delay','clock_drift','stale_calibration','unseen_combination'):
  obs=stress_observations(obs,scenario)
 if scenario in ('sleep_restriction','recovery','phase_shift','irregular_light'):
  ids=obs.person.str.extract(r'(\d+)$')[0].astype(int)
  mask=(ids%4==2)&(obs.issued_h<8*24) if scenario=='sleep_restriction' else (ids%4==2)&(obs.issued_h>=8*24) if scenario=='recovery' else (ids%4==1)&(obs.issued_h>=8*24) if scenario=='phase_shift' else ids%4==3
  obs=obs[mask].copy()
 for time,g in obs.groupby('issued_h',sort=True):
  remaining=[]
  for label in pending:
   if label['available_h']<=time:engine.ingest(label,float(time))
   else:remaining.append(label)
  pending=remaining;observations=g.to_dict('records');before=engine.decide(observations,'before',pol,interval,history,False);before_rows.extend(before)
  measured=[];measurements=[];indices=[]
  for i,(o,b) in enumerate(zip(observations,before)):
   if not b['measure_recommended'] or before_only_task==(o['person'],o['task']):continue
   key=(o['person'],o['task']);m=get_measurement(study,key,duration,precision);a={**o,**{k:v for k,v in m.items() if k!='duration_min'}};a['now_h']=float(m['short_available_h'])
   if scenario in ('device_delay','unseen_combination'):
    a['short_rt']+=60;a['device_ok']=bool(VirtualRig(Device(),135).run(300,n=30,fault='hid_delay',now_h=0)['verdict']['passed'])
   engine.bind_before(b);measured.append(a);measurements.append(m);indices.append(i)
  after=engine.decide(measured,'after',{'kind':'always'},interval,history,True);after_rows.extend(after);by_index={idx:(z,mm) for idx,z,mm in zip(indices,after,measurements)}
  for i,(o,b) in enumerate(zip(observations,before)):
   z,mm=by_index.get(i,(b,None));take=i in by_index and z['interpretable'];row={k:v for k,v in z.items() if k not in ('targets','model_features','posterior','agents','physiology','environment')};row.update(before_probability=b['event_probability'],before_prediction_ms=b['prediction_ms'],measured=bool(take),measurement_attempted=i in by_index,duration_min=float(mm['duration_min']) if mm and mm['duration_min'] is not None else None,personal_intercept=z['posterior']['intercept'],personal_slope=z['posterior']['slope'],identified_slope=z['posterior']['identified_slope'],**{'pred_'+k:v for k,v in z['targets'].items()});outputs.append(row)
   if take:
    # This target is evaluator-held; it is delivered to the engine only at its availability time.
    label=labels.loc[o['person'],o['task']].to_dict();label.pop('reference_duration_min',None);label.update(person=o['person'],task=o['task'])
    if scenario=='delayed_label':label['available_h']+=24
    pending.append(label)
 z=pd.DataFrame(outputs);y=study.labels[['person','task','rt','lapse_fraction','response_speed','slow_speed','early_fraction','reference_duration_min']].rename(columns=lambda c:'target_'+c if c not in ('person','task') else c);z=z.merge(y,on=['person','task'],validate='one_to_one');z.rename(columns={'target_reference_duration_min':'reference_duration_min'},inplace=True);z['event']=z.target_lapse_fraction>=.1;z['legacy_event']=z.target_rt>=400
 return z,before_rows,after_rows

def stress_observations(obs,scenario):
 d=obs.copy();late=d.issued_h>=9*24
 if scenario=='sleep_restriction':d.loc[late,'S']=np.minimum(.98,d.loc[late,'S']+.18)
 elif scenario=='recovery':d.loc[late,'S']=np.maximum(.05,d.loc[late,'S']-.15)
 elif scenario=='phase_shift':
  x=d.x.copy();d.x=d.xc;d.xc=-x
 elif scenario=='irregular_light':d.n=np.clip(d.n+.1*np.sin(d.issued_h),0,1)
 elif scenario=='sensor_noise':
  rng=np.random.default_rng(135);d.S=np.clip(d.S+rng.normal(0,.08,len(d)),0,1);d.x+=rng.normal(0,.1,len(d))
 elif scenario in ('short_missing','long_missing','delayed_input','unseen_combination'):
  if scenario in ('delayed_input','unseen_combination'):
   for col in ['S','x','xc','n','co2_excess','dose_ugy','gravity_deviation','sin','cos']:
    previous=d.groupby('person')[col].shift(1);d.loc[late,col]=previous[late]
  d.loc[late,'light_coverage']=.8 if scenario=='short_missing' else 0.;d.loc[late,'quality']='flagged'
 elif scenario in ('clock_drift','stale_calibration'):
  verdict=VirtualRig(Device(),135).run(300,n=30,fault='drift' if scenario=='clock_drift' else 'stale_calibration',now_h=0)['verdict'];d.loc[late,'device_ok']=bool(verdict['passed'])
 return d
