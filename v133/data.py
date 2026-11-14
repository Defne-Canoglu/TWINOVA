from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from .physiology import advance
from v132.contracts import digest
ROOT=Path(__file__).resolve().parents[1]
BASE=['S','x','xc','n','co2_excess','dose_ugy','gravity_deviation','last_rt','history_mean','history_n','sin','cos','light_coverage']
AFTER=BASE+['short_rt']

def generate(folder,seed=133110,people=64,days=14):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'manifest.json').exists():raise ValueError('refusing to overwrite study')
 if people!=64 or days<4:raise ValueError('locked sample:64 people; >=4 days')
 rng=np.random.default_rng(seed);ids=[f'SYN133-{i:03}' for i in range(people)];roles=dict(train=ids[:32],probability=ids[32:38],policy=ids[38:44],calibration=ids[44:54],test=ids[54:]);state=np.tile([-.1,-1.1,.45,.4],(people,1));state[:,:2]+=rng.normal(0,.12,(people,2));trait=rng.normal(310,24,people);chronotype=rng.normal(0,.75,people);history=[];states={};truth=[];half=int(days*48)
 for step in range(half):
  start=step*.5;end=start+.5;clock=(start+chronotype)%24;asleep=(clock>=22)|(clock<6);light=np.where(asleep,0,np.where((clock>=8)&(clock<18),350.,30.));light*=rng.lognormal(0,.1,people);state=advance(state,light,asleep,.5,max_step_h=.05);observed=light*rng.lognormal(0,.04,people);co2=np.maximum(400,rng.normal(950,100,people));dose=np.full(people,.1);gravity=np.full(people,1.)
  for i,p in enumerate(ids):history.append(dict(source='TWINOVA133_LAB',person=p,start_h=start,end_h=end,available_h=end,eye_lux=observed[i],light_position='eye',light_unit='lux',asleep=int(asleep[i]),co2_ppm=co2[i],dose_ugy_h=dose[i],gravity_g=gravity[i],quality='valid'))
  states[round(end,4)]=state.copy()
 for day in range(days):
  for hour in (8,16):
   now=day*24+hour
   if now+2>days*24:continue
   a=states[now];future=states[now+2]
   for i,p in enumerate(ids):
    stress=12*max(0,day-8)/5 if i%3==0 else 0
    short=np.clip(trait[i]+65*a[i,3]-20*a[i,0]+stress+rng.normal(0,8),100,1500);target=np.clip(trait[i]+65*future[i,3]-20*future[i,0]+stress+rng.normal(0,10),100,1500);lapse_p=1/(1+np.exp(-(target-450)/45));truth.append(dict(person=p,task=f'{day:02}-{hour}',issued_h=now,short_available_h=now+.04,target_h=now+2,target_available_h=now+2.2,short_rt=short,target_rt=target,target_lapses=int(rng.binomial(60,lapse_p)),target_n=60,device='VIRTUAL_D1' if i%2==0 else 'VIRTUAL_D2',protocol='LAB_SHORT_THEN_INDEPENDENT_PVT',quality='valid',origin='synthetic_engineering'))
 pd.DataFrame(history).to_csv(folder/'history.csv.gz',index=False);pd.DataFrame(truth).to_csv(folder/'tasks.csv',index=False)
 manifest=dict(version='13.3.0',source='TWINOVA133_LAB',origin='synthetic_engineering',population='artificial_people',clock='study_hours',seed=seed,people=people,days=days,splits=roles,light='eye photopic lux; no spectral conversion',physiology='Forger99 population prior plus sleep homeostat; synthetic PVT link',files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['history.csv.gz','tasks.csv']})
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest

def load(folder):
 folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
 if set(m['files'])!={'history.csv.gz','tasks.csv'}:raise ValueError('required study hashes')
 if m['origin'] not in ('synthetic_engineering','real_archive','mixed') or m['clock'] not in ('study_hours','UTC_epoch_hours') or not m['source'] or not m['population']:raise ValueError('study provenance')
 for n,sha in m['files'].items():
  p=(folder/n).resolve()
  if folder.resolve() not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('study file hash/path')
 h=pd.read_csv(folder/'history.csv.gz',dtype={'source':str,'person':str});t=pd.read_csv(folder/'tasks.csv',dtype={'person':str,'task':str,'device':str,'protocol':str});required={'source','person','start_h','end_h','available_h','eye_lux','light_position','light_unit','asleep','co2_ppm','dose_ugy_h','gravity_g','quality'}
 if not required<=set(h) or not {'person','task','issued_h','short_available_h','target_h','target_available_h','short_rt','target_rt','target_lapses','target_n','device','protocol','quality','origin'}<=set(t):raise ValueError('study schema')
 if h.duplicated(['person','start_h']).any() or t.duplicated(['person','task']).any():raise ValueError('duplicate observation/task')
 expected_sources=set(m.get('sources',[m['source']]))
 allowed_origins={'synthetic_engineering','real_archive'} if m['origin']=='mixed' else {m['origin']}
 if not set(h.source)<=expected_sources or not set(t.origin)<=allowed_origins or not h.light_position.eq('eye').all() or not h.light_unit.eq('lux').all():raise ValueError('identity/light semantics')
 if not h.asleep.isin([0,1]).all() or not h.quality.isin(['valid','flagged','missing']).all() or not t.quality.isin(['valid','flagged']).all():raise ValueError('quality/sleep state')
 nums=['start_h','end_h','available_h','eye_lux','co2_ppm','dose_ugy_h','gravity_g'];tn=['issued_h','short_available_h','target_h','target_available_h','short_rt','target_rt','target_lapses','target_n']
 if not np.isfinite(h[nums]).all().all() or not np.isfinite(t[tn]).all().all() or (h[nums[3:]]<0).any().any() or not ((h.start_h<h.end_h)&(h.end_h<=h.available_h)).all() or not ((t.issued_h<t.short_available_h)&(t.short_available_h<t.target_h)&(t.target_h<t.target_available_h)).all() or not (t.short_rt>0).all() or not (t.target_rt>0).all() or not ((t.target_lapses>=0)&(t.target_lapses<=t.target_n)&(t.target_n>0)).all():raise ValueError('time/numeric domain')
 if not np.allclose(t.target_h-t.issued_h,float((t.target_h-t.issued_h).iloc[0])):raise ValueError('one fixed forecast horizon per study required')
 ids=[]
 if set(m['splits'])!={'train','probability','policy','calibration','test'}:raise ValueError('five person roles required')
 for group,persons in m['splits'].items():
  if not persons or set(ids)&set(persons):raise ValueError('person role leakage/empty role')
  ids+=persons
 if set(ids)!=set(t.person) or set(ids)!=set(h.person):raise ValueError('cohort mismatch')
 for _,g in h.sort_values('start_h').groupby('person'):
  a=g.start_h.to_numpy();b=g.end_h.to_numpy()
  if len(g)>1 and (a[1:]<b[:-1]).any():raise ValueError('overlapping history')
 return m,h,t

def features(folder):
 m,h,t=load(folder);rows=[];audit=[]
 for person,g in h.groupby('person',sort=False):
  last_measured=None;missing_hours=0.;y=np.array([-.1,-1.1,.45,.4]);end=float(g.start_h.min());light_integral=0;excess=0;dose=0;grav=0;seen=[];g=g.sort_values('end_h');index=0;hist=g.to_dict('records');tasks=t[t.person==person].sort_values('issued_h')
  for r in tasks.itertuples():
   now=r.issued_h
   while index<len(hist) and hist[index]['end_h']<=now and hist[index]['available_h']<=now:
    q=hist[index].copy();dt=q['end_h']-q['start_h']
    if q['quality']=='missing':
     missing_hours+=dt
     if last_measured is None or missing_hours>2:raise ValueError('unresolved missing physiological inputs')
     for field in ('eye_lux','asleep','co2_ppm','dose_ugy_h','gravity_g'):q[field]=last_measured[field]
    else:last_measured=q.copy();missing_hours=0.
    if abs(q['start_h']-end)>1e-8:raise ValueError('unresolved history gap; supply explicit flagged interval')
    y=advance(y,q['eye_lux'],q['asleep'],dt,max_step_h=.05);end=q['end_h'];excess+=max(0,q['co2_ppm']-1000)*dt;dose+=q['dose_ugy_h']*dt;grav+=abs(1-q['gravity_g'])*dt;seen.append((q['available_h'],q['quality']=='valid'));index+=1
   if abs(end-now)>1e-8:raise ValueError('decision has incomplete/future history')
   past=tasks[(tasks.target_available_h<=now)&(tasks.issued_h<now)&(tasks.quality=='valid')];recent=[x for x in seen if x[0]>now-6];coverage=np.mean([x[1] for x in recent]) if recent else 0.;last=float(past.iloc[-1].target_rt) if len(past) else 330.;mean=float(past.target_rt.mean()) if len(past) else 330.
   record=r._asdict();record.update(S=y[3],x=y[0],xc=y[1],n=y[2],co2_excess=excess,dose_ugy=dose,gravity_deviation=grav,last_rt=last,history_mean=mean,history_n=len(past),sin=np.sin(now*np.pi/12),cos=np.cos(now*np.pi/12),light_coverage=coverage);rows.append(record);audit.append(dict(person=person,task=r.task,decision_h=now,last_input_available_h=seen[-1][0],prior_outcome_max_available_h=float(past.target_available_h.max()) if len(past) else None,origin=m['origin']))
 return m,pd.DataFrame(rows),pd.DataFrame(audit)
