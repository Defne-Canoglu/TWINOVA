"""One strict numerical decision kernel for UI/API/CLI/CV and every replay."""
import math,hashlib,json
from dataclasses import dataclass,field
import numpy as np,pandas as pd
from v133.data import BASE
from .trials import summarize,prefix

SHORT=['short_rt','short_response_speed','short_lapse_fraction','short_slow_speed','short_early_fraction','short_n','trial_metrics_available']
AFTER=BASE+SHORT
TARGETS=['rt','response_speed','lapse_fraction','slow_speed','early_fraction']
META={'person','task','issued_h','now_h','target_h','quality','device_ok','short_available_h','protocol','calibration_id','instrument_sd_ms'}
DERIVED={'observed_sleep_h24','covered_h24','hours_since_wake','reference_S','reference_C','S_rate_m150','S_rate_m075','S_rate_zero','S_rate_p075','S_rate_p150'}
ALLOWED=set(BASE+SHORT)|META|DERIVED

@dataclass
class State:
 labels:list=field(default_factory=list)
 snapshots:dict=field(default_factory=dict)
 online_q:dict=field(default_factory=dict)

class Engine:
 @staticmethod
 def measurement_decision(rows,now_h,mode='adaptive',precision=.18,lapse_ms=500,max_trials=60,summary_cache=None):
  s,acquired=prefix(rows,now_h,mode,precision,lapse_ms,summary_cache);n=len(acquired);checkpoint=n in (24,36) or n>=max_trials;enough=s is not None and s['speed_se']<=precision and s['lapse_ci_width']<=precision;candidate=n>=max_trials or (mode=='short' and n>=24) or (mode=='adaptive' and checkpoint and n>=24 and enough);minimum_reached=bool(acquired and now_h>=float(acquired[0]['stimulus_h'])+2/60-1e-10);stop=candidate and minimum_reached
  action='quality' if stop and s is None else 'stop' if stop else 'extend';return dict(action=action,summary=s,acquired=acquired,acquired_trials=n,precision=precision,reason='await_minimum_two_minutes' if candidate and not minimum_reached else 'maximum_protocol_trials' if n>=max_trials else 'fixed_short' if mode=='short' and stop else 'precision_sufficient' if stop else 'collect_more_observed_trials')
 def bind_before(self,decision):
  st=self.states.setdefault(decision['person'],State());st.snapshots['B:'+str(decision['task'])]=dict(issued_h=decision['issued_h'],S=decision['model_features'].get('S',.5),general_prediction=decision['general_prediction_ms'],prediction_ms=decision['prediction_ms'],radius_ms=decision['radius_ms'],stage='before',model_features=decision['model_features'])
 def __init__(self,bundle):self.b=bundle;self.states={}
 def ingest(self,label,now):
  allowed={'person','task','available_h','quality','rt','response_speed','lapse_fraction','slow_speed','early_fraction'}
  if set(label)-allowed:raise ValueError('label schema')
  if not np.isfinite(label['available_h']) or label['available_h']>now:raise ValueError('future label')
  st=self.states.setdefault(label['person'],State());key=str(label['task'])
  if key not in st.snapshots:raise ValueError('unacquired label cannot update history')
  if any(str(z['task'])==key for z in st.labels):return
  if label['quality']!='valid':return
  snap=st.snapshots[key];st.labels.append(dict(label,issued_h=snap['issued_h'],S=snap['S'],general_prediction=snap['general_prediction'],stage=snap['stage'],model_features=snap['model_features'],**({'dynamic_context':snap['dynamic_context']} if 'dynamic_context' in snap else {})))
  for update in [snap]+([st.snapshots['B:'+key]] if 'B:'+key in st.snapshots else []):
   stage=update['stage'];q=st.online_q.get(stage,self.b.get('intervals',{}).get(stage,{}).get('constant',math.inf));n=len(st.labels);eta=2/max(1,n)**.6
   if np.isfinite(q):st.online_q[stage]=max(0.,q+eta*(float(abs(label['rt']-update['prediction_ms'])>update['radius_ms'])-.1))
 def context(self,obs,history_mode='history'):
  q=dict(obs);st=self.states.setdefault(str(q['person']),State());past=sorted([z for z in st.labels if z['available_h']<=q['now_h'] and z['issued_h']<q['issued_h']],key=lambda z:z['issued_h'])
  if history_mode=='population':past=[]
  q['last_rt']=past[-1]['rt'] if past else 330.;q['history_mean']=float(np.mean([z['rt'] for z in past])) if past else 330.;q['history_n']=len(past)
  return q,past
 def decide(self,observations,stage='before',policy=None,interval_method=None,history_mode=None,record=True):
  if isinstance(observations,dict):return self.decide([observations],stage,policy,interval_method,history_mode,record)[0]
  if stage not in ('before','after'):raise ValueError('decision stage')
  contexts=[];histories=[]
  mode=history_mode or self.b.get('history_mode','history')
  for obs in observations:
   if set(obs)-ALLOWED:raise ValueError('target/unknown observation field')
   if stage=='before' and any(k in obs for k in SHORT):raise ValueError('unacquired short measurement')
   if obs['now_h']<obs['issued_h'] or not np.isfinite([obs[k] for k in BASE if k not in ('last_rt','history_mean','history_n')]+[obs['now_h'],obs['issued_h']]).all():raise ValueError('observation time/domain')
   if stage=='after' and (obs.get('short_available_h',math.inf)>obs['now_h'] or not all(k in obs for k in SHORT)):raise ValueError('short measurement unavailable')
   if stage=='after' and not np.isfinite([obs[k] for k in SHORT]).all():raise ValueError('nonfinite acquired measurement')
   q,past=self.context(obs,mode);contexts.append(q);histories.append(past)
  if not contexts:return []
  if stage=='after' and any(q['quality']!='valid' or not q.get('device_ok',True) or q['light_coverage']<1 for q in contexts):
   outputs=[]
   for obs in observations:
    if obs['quality']!='valid' or not obs.get('device_ok',True) or obs['light_coverage']<1:
     stripped={k:v for k,v in obs.items() if k not in SHORT and k!='short_available_h'};out=self.decide(stripped,'before',policy,interval_method,history_mode,False);out['measurement_rejected']=True;outputs.append(out)
    else:outputs.append(self.decide(obs,'after',policy,interval_method,history_mode,record))
   return outputs
  d=pd.DataFrame(contexts);spec=(self.b.get('population_models',self.b['models']) if mode=='population' else self.b['models'])[stage];prediction=np.asarray(spec['model'].predict(d[spec['cols']]))
  if prediction.ndim==1:prediction=prediction[:,None]
  risk=self.b['events'][stage].predict_proba(d[self.b['event_cols'][stage]])[:,1]
  gains=self.b['gain'].predict(d[BASE]);gainprob=self.b['gain_probability'].predict_proba(np.asarray(gains).reshape(-1,1))[:,1]
  results=[];pol=policy or self.b.get('policy',dict(kind='always',cut=0));method=interval_method or self.b.get('interval_method','constant')
  for i,(q,past) in enumerate(zip(contexts,histories)):
   estimate={name:float(prediction[i,j]) for j,name in enumerate(self.b['target_names'])};general=estimate['rt'];
   for name in ['lapse_fraction','early_fraction']:
    if name in estimate:estimate[name]=float(np.clip(estimate[name],0,1))
   post=dict(intercept=0.,slope=0.,accepted=len(past),identified_slope=False)
   if stage=='after' and mode=='personal' and past:
    cfg=self.b['personal_config'];recent=[z for z in past if q['now_h']-z['available_h']<=72 and z['stage']=='after'];post['accepted']=len(recent)
    if len(recent)>=2:
     X=np.array([[1,z['S']-.5] for z in recent]);res=np.array([z['rt']-z['general_prediction'] for z in recent]);w=np.exp(-np.array([q['now_h']-z['available_h'] for z in recent])/cfg['decay_h']);precision=np.diag([1/cfg['intercept_var'],1/cfg['slope_var']]);cov=np.linalg.inv(precision+X.T@(w[:,None]*X)/cfg['noise_var']);theta=cov@(X.T@(w*res))/cfg['noise_var'];identified=len(recent)>=6 and np.std(X[:,1])>.04
     if not identified:theta[1]=0.
     estimate['rt']+=float(theta@[1,q['S']-.5]);post.update(intercept=float(theta[0]),slope=float(theta[1]),identified_slope=bool(identified))
   if stage=='after' and mode=='dynamic' and self.b.get('personal_adapter') is not None:
    estimate,post=self.b['personal_adapter'].adjust(q,past,spec,estimate)
   valid=q['quality']=='valid' and q.get('device_ok',True) and q['light_coverage']>=1
   score=float(risk[i]);index=int(round((q['issued_h']%24)/8)) # deterministic schedule; no target or future index
   budget=pol.get('budget',1.);kind=pol['kind']
   if kind=='always':take=True
   elif kind=='fixed':
    j=int(q['issued_h']//24)*2+(0 if q['issued_h']%24<12 else 1);take=math.floor((j+1)*budget)>math.floor(j*budget)
   elif kind=='random':take=int(hashlib.sha256(f"135:{q['person']}:{q['task']}".encode()).hexdigest()[:8],16)/2**32<budget
   elif kind=='gain':take=float(gains[i])>=pol['cut']
   else:take=score>=pol['cut'] or (len(past)<2 and pol.get('cold_start',True))
   take=bool(take and valid);alarm=bool(score>=self.b.get('alarm_threshold',.5));cal=self.b.get('intervals',{}).get(stage,{});rad=cal.get('constant',math.inf);fallback=None
   if method=='normalized' and 'scale' in cal:rad=cal['normalized_q']*max(3.,float(cal['scale'].predict(d.iloc[[i]][list(getattr(cal['scale'],'feature_names_in_',spec['cols']))])[0]))
   if method=='conditional':
    group=f"{'am' if q['issued_h']%24<12 else 'pm'}:{int(score>=.5)}";cell=cal.get('conditional',{}).get(group)
    if cell and cell['eligible']:rad=cell['radius']
    else:fallback='population_constant_insufficient_people'
   if method=='online':rad=self.states[q['person']].online_q.get(stage,rad)
   p=estimate['rt'];lo=p-rad;hi=p+rad
   out=dict(version='13.5.0',person=q['person'],task=q['task'],stage=stage,issued_h=q['issued_h'],decision_h=q['now_h'],target_h=q['target_h'],prediction_ms=p,general_prediction_ms=general,targets=estimate,event_probability=score,measurement_gain_ms=float(gains[i]),measurement_gain_probability=float(gainprob[i]),alarm=alarm,measure_recommended=take,interpretable=valid,state='C' if not valid else 'N' if len(past)<2 else 'B' if alarm else 'A',prediction_interval=[float(lo) if np.isfinite(lo) else None,float(hi) if np.isfinite(hi) else None],radius_ms=float(rad),interval_method=method,interval_fallback=fallback,history_count=len(past),posterior=post,model_features={k:float(q[k]) for k in spec['cols']},last_label_available_h=max([z['available_h'] for z in past],default=None),instrument_sd_ms=q.get('instrument_sd_ms'),calibration_id=q.get('calibration_id'),target_revealed=False,clinical_validation=False,physical_validation=False,mission_clearance=None)
   out['agents']=[dict(role='physiology',passed=bool(q['light_coverage']>=1),evidence={'S':q['S'],'x':q['x'],'xc':q['xc']}),dict(role='hardware',passed=bool(q.get('device_ok',True)),evidence={'calibration_id':q.get('calibration_id'),'instrument_sd_ms':q.get('instrument_sd_ms')}),dict(role='learning',passed=all(z['available_h']<=q['now_h'] for z in past),evidence={'available_labels':len(past)}),dict(role='reference',passed=len(past)>=2,evidence={'past_personal_tasks':len(past)}),dict(role='planner',passed=bool(valid),evidence={'acquire':take,'event_probability':score,'gain_ms':float(gains[i])})]
   out['physiology']={k:float(q[k]) for k in ['S','x','xc','n']};out['environment']={k:float(q[k]) for k in ['co2_excess','dose_ugy','gravity_deviation']};out['personal_reference_band']=[float(np.mean([z['rt'] for z in past])-1.645*np.std([z['rt'] for z in past])),float(np.mean([z['rt'] for z in past])+1.645*np.std([z['rt'] for z in past]))] if len(past)>=2 else None
   out['snapshot_hash']=hashlib.sha256(json.dumps(out,sort_keys=True,allow_nan=True).encode()).hexdigest()
   if stage=='after' and record and take and valid:self.states[q['person']].snapshots[str(q['task'])]=dict(issued_h=q['issued_h'],S=q['S'],general_prediction=general,prediction_ms=p,radius_ms=rad,stage=stage,model_features=out['model_features'])
   if stage=='after' and record and take and valid and self.b.get('personal_adapter') is not None:
    self.states[q['person']].snapshots[str(q['task'])]['dynamic_context']={k:q[k] for k in ['S_rate_m150','S_rate_m075','S_rate_zero','S_rate_p075','S_rate_p150']}
   results.append(out)
  return results
