"""Conditional input replay: no target labels or UI states are fabricated."""
import copy
import numpy as np
from v135.replay import run
from .features import recompute_state,history_at

CATALOG={
 'nominal':dict(tr='Recorded condition',en='Recorded condition'),
 'sleep_restriction':dict(tr='Sleep restriction',en='Sleep restriction'),
 'phase_shift':dict(tr='Sleep/light schedule shift',en='Sleep/light schedule shift'),
 'irregular_light':dict(tr='Irregular light',en='Irregular light'),
 'short_missing':dict(tr='Son girdide eksiklik',en='Recent input missing'),
 'device_delay':dict(tr='Cihaz gecikmesi',en='Device delay'),
 'stale_calibration':dict(tr='Eski kalibrasyon',en='Stale calibration'),
 'co2_rise':dict(tr='CO₂ input increase',en='CO₂ input increase'),
 'radiation_exposure':dict(tr='Radyasyon maruziyet girdisi',en='Radiation exposure input')}

def transform(history,person,issued,scenario):
 if scenario not in CATALOG:raise ValueError('Unknown scenario')
 h=history_at(history,person,issued).copy();mask=h.start_h>=max(0,issued-48)
 if scenario=='sleep_restriction':
  h.loc[mask&(h.start_h%24>=3)&(h.start_h%24<6),'asleep']=0
  h.loc[mask&(h.asleep==0),'eye_lux']=np.maximum(h.loc[mask&(h.asleep==0),'eye_lux'],30)
 elif scenario=='phase_shift':
  rows=h.loc[mask].copy();values={round(float(r.start_h%24),6):(r.asleep,r.eye_lux)for r in rows.itertuples()}
  for idx,r in rows.iterrows():
   key=round(float((r.start_h-4)%24),6)
   if key in values:h.loc[idx,['asleep','eye_lux']]=values[key]
 elif scenario=='irregular_light':h.loc[mask,'eye_lux']=np.where((h.loc[mask,'start_h']%6)<3,5.,450.)
 elif scenario=='short_missing':h.loc[h.end_h>issued-1,'quality']='missing'
 elif scenario=='co2_rise':h.loc[h.end_h>issued-2,'co2_ppm']=2500.
 elif scenario=='radiation_exposure':h.loc[h.end_h>issued-2,'dose_ugy_h']=10.
 return h

def decide(app,request,scenario):
 if scenario not in CATALOG:raise ValueError('Unknown scenario')
 baseline=app.base_run(request);person=baseline['person'];task=baseline['task'];issued=baseline['issued_h']
 if scenario in ['nominal','device_delay','stale_calibration']:
  out=app.base_run({**request,'fault':'none'if scenario=='nominal'else scenario});h=history_at(app.history,person,out['decision_h']);return out,h,dict(id=scenario,kind='existing_measurement_replay',engine_forced_state=False,changed_inputs=[]if scenario=='nominal'else['device calibration/check'],counterfactual_target_available=False)
 h=transform(app.history,person,issued,scenario);state=recompute_state(h,person,issued)
 obs=app.study.observations[(app.study.observations.person==person)&(app.study.observations.issued_h<=issued)].copy();row=obs.task==task
 for k,v in state.items():obs.loc[row,k]=v
 if scenario=='short_missing':obs.loc[row,'quality']='flagged'
 outside=[]
 for k in ['S','x','xc','n','co2_excess','dose_ugy','gravity_deviation']:
  bounds=app.training_bounds[k]
  if not bounds[0]-1e-8<=state[k]<=bounds[1]+1e-8:outside.append(k)
 if outside:obs.loc[row,'quality']='flagged'
 study=copy.copy(app.study);study.observations=obs
 original_before=app.base_run({**request,'stage':'before'})
 _,before,_=run(study,app.b,[person],app.b['policy'],app.b['duration']['mode'],app.b['duration']['precision'],app.b['interval_method'],before_only_task=(person,task))
 out=next(x for x in before if x['task']==task)
 # Output is an unmeasured conditional forecast; old observed short/target data
 # are not recast as observations from the changed physical scenario.
 return out,h,dict(id=scenario,kind='conditional_input_replay',stage='before',window_h=48,changed_inputs=list(state),input_state=state,engine_forced_state=False,outside_training_range=outside,original_past_outcomes_retained_as_context=True,counterfactual_target_available=False,clinical_emergency_model=False,baseline_same_stage={k:original_before[k]for k in ['state','prediction_ms','event_probability','radius_ms']},scenario_same_stage={k:out[k]for k in ['state','prediction_ms','event_probability','radius_ms']},raw_timeline_window_h=[max(0,issued-48),issued],raw_timeline=h[h.end_h>issued-48][['start_h','end_h','asleep','eye_lux','co2_ppm','dose_ugy_h','quality']].iloc[::4].to_dict('records'),assumptions=['Population physiology parameters unchanged','Earlier observed performance remains original context','No counterfactual outcome accuracy is claimed'])
