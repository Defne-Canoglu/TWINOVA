import json,hashlib
from pathlib import Path
from dataclasses import dataclass
import numpy as np,pandas as pd,joblib
from v133.data import features,load as load_base,BASE
from .trials import validate,summarize,prefix
from .engine import SHORT,Engine
from .adapter import calibrations

@dataclass
class Study:
 manifest:dict
 observations:pd.DataFrame
 labels:pd.DataFrame
 short:dict
 trial_frames:dict
 audit:pd.DataFrame

def load(folder,rebuild=False):
 folder=Path(folder);rawmanifest=json.loads((folder/'manifest.json').read_text());bound=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest();studycache=folder/'study_cache.joblib';binding=folder/'study_cache_binding.json'
 if not rebuild and studycache.exists() and binding.exists():
  certificate=json.loads(binding.read_text())
  if certificate.get('schema')==2 and certificate.get('manifest')==bound and certificate.get('cache')==hashlib.sha256(studycache.read_bytes()).hexdigest():
   for mapping in ['files','trial_files','calibration_files']:
    for name,sha in rawmanifest.get(mapping,{}).items():
     p=(folder/name).resolve()
     if folder.resolve() not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('cached study source hash/path mismatch')
   return joblib.load(studycache)
 m,h,t=load_base(folder);sources=m.get('sources',[m['source']]);trialdata=None
 for n,sha in m.get('trial_files',{}).items():
  p=(folder/n).resolve()
  if folder.resolve() not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('trial path/hash')
  if n=='trials.csv.gz':trialdata=pd.read_csv(p,dtype={'person':str,'task':str,'device':str,'calibration_id':str})
 if trialdata is not None:validate(trialdata,t,sources,m.get('protocols'))
 calibration_map=calibrations(folder,m)
 if trialdata is not None:
  for key,g in trialdata.groupby('calibration_id',sort=False):
   if key not in calibration_map:raise ValueError('Missing independent calibration evidence')
   cal=calibration_map[key]
   if not np.allclose(g.correction_ms,cal['bias_ms'],atol=1e-7):raise ValueError('trial correction/evidence mismatch')
 cache=folder/'environment_features.csv.gz';cachekey=folder/'environment_cache.json';identity=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()
 valid=False
 if cache.exists() and cachekey.exists():
  c=json.loads(cachekey.read_text());valid=c['input']==identity and c['features']==hashlib.sha256(cache.read_bytes()).hexdigest()
 if valid and not rebuild:
  d=pd.read_csv(cache,dtype={'person':str,'task':str});audit=pd.read_csv(folder/'feature_audit.csv')
 else:
  _,d,audit=features(folder);d.to_csv(cache,index=False);audit.to_csv(folder/'feature_audit.csv',index=False);cachekey.write_text(json.dumps(dict(input=identity,features=hashlib.sha256(cache.read_bytes()).hexdigest())))
 obs=d[['person','task','issued_h','target_h','protocol','quality',*BASE]].copy();obs['now_h']=obs.issued_h;obs['device_ok']=True;obs['calibration_id']='summary_record';obs['instrument_sd_ms']=0.
 if m['measurement_provider']=='native' and not calibration_map:obs['device_ok']=False
 labels=[];short={};frames={};groups={} if trialdata is None else {k:g for k,g in trialdata.groupby(['person','task','session'],sort=False)}
 for r in d.itertuples():
  key=(r.person,r.task);tar=groups.get((*key,'target'));sh=groups.get((*key,'short'))
  metrics=summarize(tar,m.get('protocols',{}).get('target',{}).get('lapse_ms',500)) if tar is not None else dict(rt=r.target_rt,lapse_fraction=r.target_lapses/r.target_n,response_speed=None,slow_speed=None,early_fraction=None)
  labels.append(dict(person=r.person,task=r.task,available_h=float(r.target_available_h),quality=r.quality,reference_duration_min=metrics.get('duration_min'),**{k:metrics[k] for k in ['rt','response_speed','lapse_fraction','slow_speed','early_fraction']}))
  if sh is not None:
   frames[key]=sh
   cal_id=str(sh.iloc[0].calibration_id);cal=calibration_map[cal_id];obs.loc[r.Index,'calibration_id']=cal_id;obs.loc[r.Index,'instrument_sd_ms']=float(sh.iloc[0].get('instrument_sd_ms',0));obs.loc[r.Index,'device_ok']=bool(cal['available_h']<=r.issued_h<=cal['available_h']+24)
   records=sh.sort_values('trial').to_dict('records');lapse_ms=m.get('protocols',{}).get('short',{}).get('lapse_ms',500);prefix_summaries={n:summarize(records[:n],lapse_ms) for n in (24,36,60) if n<=len(records)}
   for mode,precision in [('short',.18),('long',.18),('adaptive',.12),('adaptive',.18),('adaptive',.25)]:
    decision=Engine.measurement_decision(records,float(r.short_available_h),mode,precision,lapse_ms,m.get('protocols',{}).get('short',{}).get('max_trials',60),prefix_summaries);s=decision['summary'];acquired=decision['acquired'];short[key,mode,precision]=measurement(s,acquired)
  else:
   short[key,'summary',.18]=dict(short_rt=float(r.short_rt),short_response_speed=0.,short_lapse_fraction=0.,short_slow_speed=0.,short_early_fraction=0.,short_n=0.,trial_metrics_available=0.,short_available_h=float(r.short_available_h),calibration_id='summary_record',instrument_sd_ms=0.,duration_min=None)
 lab=pd.DataFrame(labels)
 if lab[['rt','lapse_fraction']].isna().any().any():raise ValueError('required target summaries')
 study=Study(m,obs,lab,short,frames,audit);joblib.dump(study,studycache,compress=3);binding.write_text(json.dumps(dict(schema=2,manifest=bound,cache=hashlib.sha256(studycache.read_bytes()).hexdigest())));return study

def measurement(s,acquired):
 if not s:raise ValueError('no valid trial prefix')
 return dict(short_rt=s['rt'],short_response_speed=s['response_speed'],short_lapse_fraction=s['lapse_fraction'],short_slow_speed=s['slow_speed'],short_early_fraction=s['early_fraction'],short_n=s['n'],trial_metrics_available=1.,short_available_h=max(s['available_h'],float(acquired[0]['stimulus_h'])+s['duration_min']/60),calibration_id=acquired[-1]['calibration_id'],instrument_sd_ms=float(acquired[-1].get('instrument_sd_ms',0)),duration_min=s['duration_min'])

def get_measurement(study,key,mode='adaptive',precision=.18):
 if key in study.trial_frames:return study.short[key,mode,precision]
 return study.short[key,'summary',.18]
