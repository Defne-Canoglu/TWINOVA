import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from v134.sample import generate as environment
from v133.hardware import Device,VirtualRig
from v134.timing import fit_correction,correct
from .trials import summarize

def generate(folder,seed=135221):
 folder=Path(folder);m=environment(folder,seed=seed);h=pd.read_csv(folder/'history.csv.gz');t=pd.read_csv(folder/'tasks.csv');h.person=h.person.str.replace('C134','C135');t.person=t.person.str.replace('C134','C135');h.source='TWINOVA135_STUDY';m['splits']={k:[p.replace('C134','C135') for p in v] for k,v in m['splits'].items()};rng=np.random.default_rng(seed+1);rows=[];calcache={}
 for i,r in enumerate(t.itertuples()):
  # Daily independent optical/contact calibration; not inferred from participant RT.
  key=(r.device,int(r.issued_h//24))
  if key not in calcache:
   rig=VirtualRig(Device(name=r.device,display_hz=60 if r.device.endswith('1') else 120),seed+i);z=rig.run(300,n=40,now_h=key[1]*24);calcache[key]=fit_correction(z['app'],z['raw'],z['metadata'],key[1]*24)
  cal=calcache[key];bias=cal['bias_ms']
  for session,n,start,latent,protocol in [('short',60,r.issued_h,r.short_rt,'ENGINEERING_SHORT_5MIN'),('target',120,r.target_h,r.target_rt,'ENGINEERING_TARGET_10MIN')]:
   p=.01+.27*expit((latent-365)/18);base=rng.lognormal(np.log(latent),.11,n);lapses=rng.random(n)<p;base[lapses]+=rng.uniform(190,480,lapses.sum());early=rng.random(n)<.006;no=(rng.random(n)<.003)&~early;base[early]=80;base[no]=5000
   group=[]
   for j,rt in enumerate(base):
    stimulus=start+j*5/3600;raw=float(rt+bias+rng.normal(0,.2));raw=float(5000+bias) if no[j] else raw;corrected,sd=correct(raw,cal,stimulus);response=stimulus+raw/3600000 if not no[j] else None;available=stimulus+(5 if no[j] else raw/1000)/3600
    group.append(dict(source='TWINOVA135_STUDY',person=r.person,task=r.task,session=session,trial=j,protocol=protocol,device=r.device,stimulus_h=stimulus,response_h=response,available_h=available,raw_rt_ms=raw,rt_ms=corrected,correction_ms=bias,no_response=bool(no[j]),early=bool(early[j]),quality='valid',calibration_id=f'{r.device}-DAY{key[1]}',calibration_available_h=cal['available_h'],calibration_valid_until_h=cal['available_h']+24,instrument_sd_ms=sd))
   rows.extend(group);s=summarize(group)
   if session=='short':t.loc[i,'short_rt']=s['rt'];t.loc[i,'short_available_h']=max(start+n*5/3600,max(x['available_h'] for x in group))+.0001
   else:t.loc[i,'target_rt']=s['rt'];t.loc[i,'target_available_h']=max(start+n*5/3600,max(x['available_h'] for x in group))+.02;t.loc[i,'target_lapses']=s['lapses'];t.loc[i,'target_n']=s['n']
 t.protocol='ENGINEERING_SHORT5_TARGET10';h.to_csv(folder/'history.csv.gz',index=False);t.to_csv(folder/'tasks.csv',index=False);pd.DataFrame(rows).to_csv(folder/'trials.csv.gz',index=False)
 m.update(version='13.5.0',source='TWINOVA135_STUDY',seed=seed,generator='v134 analytic physiology environment; independent trial-level lognormal/long-response mixture; all session outcomes derived from corrected trials',trial_files={'trials.csv.gz':hashlib.sha256((folder/'trials.csv.gz').read_bytes()).hexdigest()},protocols={'short':dict(name='ENGINEERING_SHORT_5MIN',lapse_ms=500,max_trials=60,timeout_ms=5000,slot_seconds=5),'target':dict(name='ENGINEERING_TARGET_10MIN',lapse_ms=500,max_trials=120,timeout_ms=5000,slot_seconds=5)},files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['history.csv.gz','tasks.csv']})
 (folder/'manifest.json').write_text(json.dumps(m,indent=2));write_calibrations(folder,seed);return json.loads((folder/'manifest.json').read_text())

def write_calibrations(folder,seed=135221):
 folder=Path(folder);m=json.loads((folder/'manifest.json').read_text());t=pd.read_csv(folder/'tasks.csv');index={};files={}
 for i,r in enumerate(t.itertuples()):
  key=f'{r.device}-DAY{int(r.issued_h//24)}'
  if key in index:continue
  z=VirtualRig(Device(name=r.device,display_hz=60 if r.device.endswith('1') else 120),seed+i).run(300,n=40,now_h=int(r.issued_h//24)*24)
  base=folder/'hardware'/key;base.mkdir(parents=True,exist_ok=True);z['app'].to_csv(base/'app.csv',index=False);(base/'raw.txt').write_text(z['raw']);(base/'metadata.json').write_text(json.dumps(z['metadata'],indent=2));index[key]={name:str((base/file).relative_to(folder)) for name,file in [('app','app.csv'),('raw','raw.txt'),('metadata','metadata.json')]}
  for name in index[key].values():files[name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
 m['calibration_evidence']=index;m['calibration_files']=files;(folder/'manifest.json').write_text(json.dumps(m,indent=2))
