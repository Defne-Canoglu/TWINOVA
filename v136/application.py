"""Read-only extension facade: the original numerical application is unchanged."""
import copy,time
import numpy as np,pandas as pd
from v135.application import Application as Base
from v135.adapter import calibrations
from v135.data import get_measurement
from .common import clean,ROOT,sha
from .evidence import assess
from .operations import progress,hardware_summary
from .scenarios import decide as scenario_decide

class Application(Base):
 def __init__(self,data=None,model=None):
  super().__init__(data,model);self.history=pd.read_csv(self.data/'history.csv.gz',dtype={'person':str});self.calibrations=calibrations(self.data,self.study.manifest);self._hardware=hardware_summary();self.started=time.monotonic();train=self.study.observations[self.study.observations.person.isin(self.b['splits']['train'])];self.training_bounds={k:(float(train[k].min()),float(train[k].max()))for k in ['S','x','xc','n','co2_excess','dose_ugy','gravity_deviation']}
 def hardware(self):
  self._hardware=hardware_summary();return self._hardware
 def base_run(self,request,ledger=None):return super().run(request,ledger)
 def health(self):return dict(service='TWINOVA',extension_version='13.6.0',engine='v135.engine.Engine.decide',local_inference=True,model_loaded=True,model_sha256=self.hash,data_sha256=self.b['data_sha256'],checked_uptime_s=round(time.monotonic()-self.started,3),live_sensor_subscription=False,physical_validation=False)
 def enrich(self,out,history=None,scenario=None):
  d=copy.deepcopy(out);now=float(d['decision_h']);cal=self.calibrations.get(d.get('calibration_id'),{});cal={**cal,'valid_until_h':cal.get('available_h',0)+cal.get('max_age_h',24)}if cal else{}
  d['evidence_assessment']=assess(d,history if history is not None else self.history,self.study.observations,cal)
  records=[]
  for r in self.study.observations[(self.study.observations.person==d['person'])&(self.study.observations.issued_h<=d['issued_h'])].itertuples():
   m=get_measurement(self.study,(r.person,r.task),self.b['duration']['mode'],self.b['duration']['precision'])
   valid=bool(r.device_ok and r.quality=='valid'and r.light_coverage>=1)
   records.append(dict(person=r.person,task=r.task,issued_h=float(r.issued_h),available_h=float(m['short_available_h']),completed=True,valid=valid))
  d['routine_progress']=progress(records,now,clock=self.study.manifest['clock'])
  d['hardware_validation']={k:v for k,v in self.hardware().items()if k!='rows'}
  d['runtime_status']=dict(version='13.6.0',local_inference=True,decision_engine='v135.engine.Engine.decide',model_loaded=True,checked_for_this_response=True,live_sensor_subscription=False)
  source_h=history if history is not None else self.history;g=source_h[(source_h.person==d['person'])&(source_h.end_h<=d['issued_h'])&(source_h.available_h<=d['issued_h'])];span=float(d['issued_h']-g.start_h.min())if len(g)else 0.;observed=float((g.loc[g.quality=='valid','end_h']-g.loc[g.quality=='valid','start_h']).sum())
  d['environment_context']=dict(observed_time_fraction=observed/span if span>0 else None,coverage_window_h=span,issued_h=d['issued_h'],integrated_from_h=float(self.history[self.history.person==d['person']].start_h.min()),unit_by_field={'co2_excess':'ppm·h','dose_ugy':'µGy','gravity_deviation':'g·h'},health_alarm_thresholds_defined=False,source_scope='Accumulated model inputs, not instantaneous sensors')
  if scenario:d['scenario']=scenario
  return clean(d)
 def run(self,request,ledger=None,evidence=True):
  d=self.base_run(request,ledger);return self.enrich(d)if evidence else d
 def scenario(self,request):
  if not isinstance(request,dict)or set(request)-{'person','task','stage','lang','scenario'}:raise ValueError('Scenario request fields')
  q={k:v for k,v in request.items()if k!='scenario'};kind=request.get('scenario','nominal');out,h,meta=scenario_decide(self,q,kind)
  if 'lang'not in out:
   ref=self.base_run({**q,'stage':'before'});out={**ref,**out,'requested_stage':'before','measurement_acquired':False}
   out['next_action']=('Review input suitability; scenario output is not mission clearance.'if out['lang']=='tr'else'Review input suitability; scenario output is not mission clearance.')
   from v132.contracts import digest
   out['application_snapshot_hash']=digest({k:v for k,v in clean(out).items()if k not in ['application_snapshot_hash','next_action','lang']})
  return self.enrich(out,h,meta)
