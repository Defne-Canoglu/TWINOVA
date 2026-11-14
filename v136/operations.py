"""Record-derived progress, current-vs-suite quality, and virtual communication."""
import heapq,json,math
from pathlib import Path
import pandas as pd
from .common import ROOT,sha,clean

class CommunicationQueue:
 def __init__(self,one_way_s=1200):
  if not math.isfinite(one_way_s)or not 0<=one_way_s<=7200:raise ValueError('Communication-delay domain')
  self.one_way_s=float(one_way_s);self.time_s=0.;self.events=[];self.counter=0;self.delivered=[]
 def submit(self,payload):
  self.counter+=1;heapq.heappush(self.events,(self.time_s+self.one_way_s,self.counter,'ground_received',payload));return self.counter
 def advance(self,seconds):
  if not math.isfinite(seconds)or seconds<0:raise ValueError('Virtual clock cannot reverse')
  target=self.time_s+seconds
  while self.events and self.events[0][0]<=target:
   t,i,kind,payload=heapq.heappop(self.events);self.delivered.append(dict(time_s=t,id=i,kind=kind,payload=payload))
   if kind=='ground_received':heapq.heappush(self.events,(t+self.one_way_s,i,'reply_received',{'acknowledged':True}))
  self.time_s=target;return self.snapshot()
 def snapshot(self):return dict(scenario=True,clock='virtual_seconds',time_s=self.time_s,one_way_s=self.one_way_s,round_trip_s=2*self.one_way_s,pending=len(self.events),delivered=list(self.delivered),local_inference_delayed=False)

def progress(records,decision_time,day_length_h=24.,clock='study_hours',timezone_offset_h=0.):
 if day_length_h<=0 or not math.isfinite(day_length_h):raise ValueError('Day length')
 unique={}
 for r in records:
  if r.get('available_h',math.inf)>decision_time or not r.get('completed',False)or not r.get('valid',False):continue
  key=(str(r['person']),str(r['task']));unique[key]=r
 days={int(math.floor((float(r['issued_h'])+timezone_offset_h)/day_length_h))for r in unique.values()};current=int(math.floor((decision_time+timezone_offset_h)/day_length_h));end=current if current in days else current-1;streak=0
 while end in days:streak+=1;end-=1
 return dict(mode='record_review',new_measurements_created=0,completed_records=len(unique),completed_days=len(days),consecutive_observed_days=streak,day_length_h=day_length_h,clock=clock,timezone_offset_h=timezone_offset_h,last_record_available_h=max([r['available_h']for r in unique.values()],default=None),missing_days_not_counted=True,mission_clearance=None)

def _hardware_summary(report=None):
 p=Path(report or ROOT/'reports_v136/hardware_bench/bench_results.csv')
 if not p.is_file():return dict(status='unavailable',cases=None,matches_expectation=None,scope='virtual',physical_run=False,reason='No benchmark artifact')
 proof=p.with_name('execution.json')
 if not proof.exists():return dict(status='unverified',cases=None,matches_expectation=None,scope='virtual',physical_run=False,reason='No current suite execution certificate')
 cert=json.loads(proof.read_text());from datetime import datetime,timezone
 age=(datetime.now(timezone.utc)-datetime.fromisoformat(cert['completed_at'])).total_seconds()/86400
 if not 0<=age<=cert.get('freshness_days',30) or cert.get('report_sha256')!=sha(p) or any(sha(ROOT/k)!=v for k,v in cert.get('source_hashes',{}).items()):return dict(status='stale',cases=None,matches_expectation=None,scope='virtual',physical_run=False,reason='Stale or mismatched suite evidence')
 f=pd.read_csv(p);required={'passed','expected_pass','matches_expectation','fault','device','condition'}
 if not required<=set(f):return dict(status='invalid',cases=None,matches_expectation=None,scope='virtual',physical_run=False)
 def boolean(s):
  if s.dtype==bool:return s
  x=s.astype(str).str.lower()
  if not x.isin(['true','false']).all():raise ValueError('Invalid benchmark boolean')
  return x.eq('true')
 passed=boolean(f.passed);expected=boolean(f.expected_pass);matches=boolean(f.matches_expectation);recomputed=passed==expected
 verified=bool((matches==recomputed).all());return clean(dict(status='reported'if verified else'inconsistent',cases=len(f),measurement_accepts=int(passed.sum()),measurement_rejects=int((~passed).sum()),matches_expectation=int(recomputed.sum())if verified else None,report_sha256=sha(p),report_modified_utc=cert['completed_at'],version='13.6.0',scope='discrete-event virtual rig, not firmware instruction emulation',physical_run=False,firmware_build='see archived build evidence',rows=f.to_dict('records'),report_path=str(p.relative_to(ROOT))if p.is_relative_to(ROOT)else p.name))


def hardware_summary(report=None):
 try:return _hardware_summary(report)
 except (ValueError,KeyError,TypeError,OSError) as e:return dict(status='invalid',cases=None,matches_expectation=None,scope='virtual',physical_run=False,reason=str(e))
