"""Available-history summaries. Missing/forward-carried time is not observed time."""
import math
import numpy as np,pandas as pd
from v133.physiology import advance,Parameters

RATE_GRID=(-.15,-.075,0.,.075,.15)
RATE_COLS=['S_rate_m150','S_rate_m075','S_rate_zero','S_rate_p075','S_rate_p150']
W_COLS=['observed_sleep_h24','covered_h24','hours_since_wake']
PUBLISHED=['reference_S','reference_C']
DERIVED=W_COLS+RATE_COLS+PUBLISHED

def history_at(history,person,decision):
 g=history[(history.person==person)&(history.end_h<=decision+1e-10)&(history.available_h<=decision+1e-10)].sort_values(['start_h','end_h']).copy()
 if len(g)>1 and (g.start_h.to_numpy()[1:]<g.end_h.to_numpy()[:-1]-1e-10).any():raise ValueError('Overlapping available intervals')
 return g

def window_summary(g,decision,window=24.):
 if not math.isfinite(decision)or window<=0:raise ValueError('Invalid summary time/window')
 g=g[(g.available_h<=decision)&(g.end_h<=decision)].sort_values('start_h')
 if len(g)>1 and (g.start_h.to_numpy()[1:]<g.end_h.to_numpy()[:-1]-1e-10).any():raise ValueError('Overlapping summary intervals')
 records=[];total=0.;sleep=0.;onsets=0;last=None
 for i,r in g.iterrows():
  lo=max(float(r.start_h),decision-window);hi=min(float(r.end_h),decision)
  if hi<=lo:continue
  observed=r.quality=='valid' and np.isfinite(r.asleep)and r.asleep in (0,1)
  rid=str(r.get('record_id',f'{r.person}:{float(r.start_h):g}:{float(r.end_h):g}'))
  records.append(dict(id=rid,start_h=lo,end_h=hi,available_h=float(r.available_h),quality=str(r.quality)))
  if observed:
   total+=hi-lo;sleep+=(hi-lo)*r.asleep
   if r.asleep==1 and last is not None and last[0]==0 and abs(lo-last[1])<1e-8:onsets+=1
   last=(r.asleep,hi)
  else:last=None
 return dict(window_h=float(window),observed_h=float(total),sleep_h=float(sleep),coverage_fraction=float(min(1,total/window)),sleep_onsets=onsets,records=records)

def since_wake(g,decision):
 wake=None;previous=None
 for r in g.itertuples():
  if r.quality!='valid':previous=None;wake=None;continue
  if previous and abs(r.start_h-previous.end_h)<1e-8 and previous.asleep==1 and r.asleep==0:wake=float(r.start_h)
  if r.asleep==1:wake=None
  previous=r
 return float(decision-wake)if wake is not None and previous and abs(previous.end_h-decision)<1e-8 else None

def homeostat(value,asleep,hours,wake_scale=1.):
 if not 0<=value<=1 or asleep not in (0,1)or hours<0 or not np.isfinite([value,hours,wake_scale]).all()or wake_scale<=0:raise ValueError('Homeostat domain')
 return value*np.exp(-hours/4.2)if asleep else 1-(1-value)*np.exp(-hours/(18.2*wake_scale))

def derived_table(history,observations,sol_length_h=24.):
 if sol_length_h<=0:raise ValueError('Invalid external day')
 rows=[]
 for person,tasks in observations.groupby('person',sort=False):
  h=history[history.person==person].sort_values('start_h');state=np.full(len(RATE_GRID),.4);index=0;records=h.to_dict('records');last=None;missing=0.
  for task in tasks.sort_values('issued_h').itertuples():
   t=float(task.issued_h)
   while index<len(records)and records[index]['end_h']<=t and records[index]['available_h']<=t:
    r=records[index];dt=r['end_h']-r['start_h'];asleep=r['asleep']
    if r['quality']=='missing':
     missing+=dt
     if last is None or missing>2:raise ValueError('Unresolved missing input for derived state')
     asleep=last['asleep']
    else:last=r;missing=0.
    state=np.array([homeostat(s,asleep,dt,1+offset)for s,offset in zip(state,RATE_GRID)]);index+=1
   available=history_at(h,person,t);w=window_summary(available,t,24.)
   sw=since_wake(available,t)
   rows.append(dict(person=person,task=task.task,observed_sleep_h24=w['sleep_h'],covered_h24=w['observed_h'],hours_since_wake=sw if sw is not None else np.nan,reference_S=state[2],reference_C=float(np.cos(2*np.pi*((t%sol_length_h)-16)/sol_length_h)),**dict(zip(RATE_COLS,state))))
 return pd.DataFrame(rows)

def recompute_state(history,person,decision):
 g=history_at(history,person,decision);state=np.array([-.1,-1.1,.45,.4]);end=float(g.start_h.min())if len(g)else 0.;excess=dose=gravity=0.;last=None;missing=0.;seen=[]
 for r in g.to_dict('records'):
  if abs(r['start_h']-end)>1e-8:raise ValueError('History gap')
  dt=r['end_h']-r['start_h']
  if r['quality']=='missing':
   missing+=dt
   if last is None or missing>2:raise ValueError('Unresolved missing physiological inputs')
   r={**r,**{k:last[k]for k in ['asleep','eye_lux','co2_ppm','dose_ugy_h','gravity_g']}}
  else:last=r.copy();missing=0.
  state=advance(state,r['eye_lux'],r['asleep'],dt,max_step_h=.05)
  excess+=max(0,r['co2_ppm']-1000)*dt;dose+=r['dose_ugy_h']*dt;gravity+=abs(1-r['gravity_g'])*dt;seen.append((r['available_h'],r['quality']=='valid'));end=r['end_h']
 if abs(end-decision)>1e-8:raise ValueError('Incomplete history at decision')
 recent=[v for t,v in seen if t>decision-6]
 return dict(S=float(state[3]),x=float(state[0]),xc=float(state[1]),n=float(state[2]),co2_excess=excess,dose_ugy=dose,gravity_deviation=gravity,light_coverage=float(np.mean(recent))if recent else 0.)
