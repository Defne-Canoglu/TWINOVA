"""One trial contract and one prefix-only summary for every source."""
import math,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta
FIELDS=['source','person','task','session','trial','protocol','device','stimulus_h','response_h','available_h','raw_rt_ms','rt_ms','correction_ms','no_response','early','quality','calibration_id','calibration_available_h','calibration_valid_until_h']

def summarize(rows,lapse_ms=500):
 d=pd.DataFrame(rows)
 if d.empty:return None
 all_trials=d.copy();d=d[d.quality=='valid'];valid=d[~d.early.astype(bool)]
 if len(valid)==0:return None
 rt=valid.rt_ms.to_numpy(float);speed=1000/rt;slow=np.sort(rt)[-max(1,math.ceil(.1*len(rt))):];k=int(((rt>=lapse_ms)|valid.no_response.astype(bool).to_numpy()).sum());n=len(rt)
 return dict(rt=float(np.median(rt)),response_speed=float(speed.mean()),lapse_fraction=k/n,slow_speed=float(np.mean(1000/slow)),early_fraction=float(d.early.mean()),n=n,lapses=k,speed_se=float(np.std(speed,ddof=1)/np.sqrt(n)) if n>1 else math.inf,lapse_ci_width=float(beta.ppf(.975,k+1,n-k+1)-beta.ppf(.025,k+1,n-k+1)),duration_min=float((all_trials.stimulus_h.max()-all_trials.stimulus_h.min())*60+5/60),available_h=float(d.available_h.max()))

def validate(trials,tasks,source_ids,protocols=None):
 d=trials
 if not set(FIELDS)<=set(d):raise ValueError('trial schema')
 if d.duplicated(['person','task','session','trial']).any():raise ValueError('duplicate trial')
 keys=set(zip(tasks.person,tasks.task));lookup=tasks.set_index(['person','task'])
 if not set(zip(d.person,d.task))<=keys or not set(d.source)<=set(source_ids):raise ValueError('trial identity/source')
 if not d.session.isin(['short','target']).all() or not d.quality.isin(['valid','flagged','missing']).all():raise ValueError('trial session/quality')
 if not d[['no_response','early']].isin([0,1,False,True]).all().all():raise ValueError('trial flags')
 nums=['stimulus_h','available_h','raw_rt_ms','rt_ms','correction_ms','calibration_available_h','calibration_valid_until_h']
 if not np.isfinite(d[nums]).all().all() or (d.rt_ms<=0).any() or not np.allclose(d.raw_rt_ms-d.correction_ms,d.rt_ms,atol=1e-8):raise ValueError('trial numbers/correction')
 if not ((d.stimulus_h<=d.available_h)&(d.calibration_available_h<=d.stimulus_h)&(d.calibration_valid_until_h>=d.stimulus_h)).all():raise ValueError('trial availability/calibration')
 responded=~d.no_response.astype(bool)
 if not ((d.loc[~responded,'available_h']-d.loc[~responded,'stimulus_h'])*3600000>=d.loc[~responded,'rt_ms']-.01).all():raise ValueError('timeout available before observation')
 if not np.isfinite(d.loc[responded,'response_h']).all() or not ((d.loc[responded,'response_h']>=d.loc[responded,'stimulus_h'])&(d.loc[responded,'available_h']>=d.loc[responded,'response_h'])).all():raise ValueError('trial response time')
 if not np.allclose((d.loc[responded,'response_h']-d.loc[responded,'stimulus_h'])*3600000,d.loc[responded,'raw_rt_ms'],atol=.001):raise ValueError('trial clock/RT mismatch')
 for (p,t,session),g in d.groupby(['person','task','session'],sort=False):
  r=lookup.loc[p,t];g=g.sort_values('trial')
  if not (np.diff(g.stimulus_h)>0).all() or not g.device.eq(r.device).all():raise ValueError('trial order/device')
  expected=(protocols or {}).get(session,{}).get('name','ENGINEERING_SHORT_5MIN' if session=='short' else 'ENGINEERING_TARGET_10MIN')
  if not g.protocol.eq(expected).all():raise ValueError('trial protocol')
  if session=='short' and (g.stimulus_h.min()<r.issued_h or g.available_h.max()>r.short_available_h):raise ValueError('short trial timing')
  if session=='target' and (g.stimulus_h.min()<r.target_h or g.available_h.max()>r.target_available_h):raise ValueError('target trial timing')
  s=summarize(g,(protocols or {}).get(session,{}).get('lapse_ms',500));field='short_rt' if session=='short' else 'target_rt'
  if s is not None and not np.isclose(s['rt'],r[field],atol=1e-7):raise ValueError('trial/session summary mismatch')
  if session=='target' and s is not None and (s['lapses']!=r.target_lapses or s['n']!=r.target_n):raise ValueError('trial/session lapse count mismatch')
 return d

def prefix(rows,now_h,mode='adaptive',precision=.18,lapse_ms=500,summary_cache=None):
 """Sequential acquisition: later rows cannot influence a stopping decision."""
 acquired=[];stop=None
 records=rows.sort_values('trial').to_dict('records') if isinstance(rows,pd.DataFrame) else sorted(rows,key=lambda r:r['trial'])
 for r in records:
  if r['available_h']>now_h:break
  acquired.append(r);n=len(acquired)
  if n not in (24,36,60):continue
  s=summary_cache[n] if summary_cache is not None and n in summary_cache else summarize(acquired,lapse_ms)
  enough=s is not None and s['speed_se']<=precision and s['lapse_ci_width']<=precision
  if (mode=='short' and n==24) or (mode=='adaptive' and enough) or n==60:stop=s;break
 if stop is None:stop=summarize(acquired,lapse_ms)
 return stop,acquired
