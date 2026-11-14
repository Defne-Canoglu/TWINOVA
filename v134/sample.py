"""Bounded cohort: independent sleep/phase observation equation, same data contract."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def generate(folder,seed=134111,people=128,days=14):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'manifest.json').exists():raise ValueError('Never overwrite a study')
 if people!=128:raise ValueError('Preregistered bounded cohort is 128 people')
 rng=np.random.default_rng(seed);ids=[f'C134-{i:03d}' for i in range(people)]
 roles=dict(train=ids[:72],probability=ids[72:78],policy=ids[78:88],calibration=ids[88:108],test=ids[108:])
 trait=rng.normal(310,24,people);sensitivity=np.clip(rng.normal(70,14,people),35,110);phase=rng.normal(0,1,people);s=np.full(people,.4);history=[];truth={};tasks=[]
 for step in range(days*48):
  start=step*.5;end=start+.5;day=int(start/24);clock=(start+phase+np.where(np.arange(people)%4==1,min(day*.2,2),0))%24
  regime=np.arange(people)%4;bed=np.where((regime==2)&(day<8),1.,23.);wake=np.where((regime==3)&(day%3==0),4.,6.)
  asleep=np.where(bed>wake,(clock>=bed)|(clock<wake),(clock>=bed)&(clock<wake))
  light=np.where(asleep,0,np.where((clock>=8)&(clock<18),350.,30.))*rng.lognormal(0,.25,people)
  s=np.where(asleep,s*np.exp(-.5/4.2),1-(1-s)*np.exp(-.5/18.2))
  # Independent target mechanism: analytic daily phase, nonlinear sleep response,
  # personal slopes and late drift; no Forger state is read by this generator.
  rt=trait+sensitivity*s+12*s*s-18*np.cos((end+phase-15)*np.pi/12)+np.where(regime==0,max(0,day-8)*1.7,0)
  truth[round(end,4)]=rt.copy()
  for i,p in enumerate(ids):
   missing=(i%11==0 and step%173==50);observed=light[i]*rng.lognormal(0,.08)
   history.append(dict(source='TWINOVA134_STUDY',person=p,start_h=start,end_h=end,available_h=end,eye_lux=observed,light_position='eye',light_unit='lux',asleep=int(asleep[i]),co2_ppm=max(400,rng.normal(950+100*(regime[i]==2),110)),dose_ugy_h=.1,gravity_g=1.,quality='missing' if missing else 'valid'))
 for day in range(days):
  for hour in (8,16):
   now=day*24+hour
   for i,p in enumerate(ids):
    short=float(np.clip(truth[now][i]+rng.normal(0,8),100,1500));target=float(np.clip(truth[now+2][i]+rng.normal(0,10),100,1500))
    tasks.append(dict(person=p,task=f'{day:02d}-{hour}',issued_h=now,short_available_h=now+.04,target_h=now+2,target_available_h=now+2.2,short_rt=short,target_rt=target,target_lapses=rng.binomial(60,1/(1+np.exp(-(target-450)/45))),target_n=60,device='VIRTUAL_D1' if i%2==0 else 'VIRTUAL_D2',protocol='LAB_SHORT_THEN_INDEPENDENT_PVT',quality='valid',origin='synthetic_engineering'))
 pd.DataFrame(history).to_csv(folder/'history.csv.gz',index=False);pd.DataFrame(tasks).to_csv(folder/'tasks.csv',index=False)
 m=dict(version='13.4.0',source='TWINOVA134_STUDY',origin='synthetic_engineering',population='artificial_people',clock='study_hours',measurement_provider='virtual',habitat_scenario=True,seed=seed,people=people,days=days,splits=roles,generator='analytic phase + exact sleep homeostat + nonlinear heterogeneous response; different from Forger observation features',files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['history.csv.gz','tasks.csv']})
 (folder/'manifest.json').write_text(json.dumps(m,indent=2));return m
