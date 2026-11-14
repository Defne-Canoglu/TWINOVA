"""Study/transport facade; numerical inference and replay share Engine.decide."""
import json,hashlib,copy
from pathlib import Path
import joblib,pandas as pd
from v133.data import ROOT
from v136.common import clean
from v132.ledger import Ledger
from v132.contracts import digest
from .data import load
from .replay import run
from .engine import Engine

class Application:
 def __init__(self,data=None,model=None):
  self.data=Path(data or ROOT/'data_v135/study');self.model=Path(model or ROOT/'models_v135/study/twin.joblib');self.b=joblib.load(self.model);self.hash=hashlib.sha256(self.model.read_bytes()).hexdigest()
  if self.b['data_sha256']!=hashlib.sha256((self.data/'manifest.json').read_bytes()).hexdigest():raise ValueError('Model/data mismatch')
  self.study=load(self.data)
 def prefix_decision(self,request):
  if not isinstance(request,dict) or set(request)-{'person','task','trials'}:raise ValueError('prefix request fields')
  person=str(request.get('person',self.b['splits']['test'][0]));task=str(request.get('task'));n=int(request.get('trials',24))
  if not 1<=n<=60:raise ValueError('prefix count')
  key=(person,task)
  if key not in self.study.trial_frames:raise ValueError('No observed trial table; no invented duration decision')
  rows=self.study.trial_frames[key].sort_values('trial').head(n);now=float(rows.available_h.max());config=self.study.manifest.get('protocols',{}).get('short',{});decision=Engine.measurement_decision(rows,now,'adaptive',self.b['duration']['precision'],config.get('lapse_ms',500),config.get('max_trials',60));decision.pop('acquired');return clean(dict(person=person,task=task,as_of_h=now,target_revealed=False,**decision))
 def run(self,request,ledger=None):
  if not isinstance(request,dict) or set(request)-{'person','task','stage','lang','fault'}:raise ValueError('Unknown request / target injection')
  person=str(request.get('person',self.b['splits']['test'][0]));g=self.study.observations[self.study.observations.person==person].sort_values('issued_h')
  if g.empty:raise ValueError('Unknown person')
  task=str(request.get('task',g.iloc[-1].task));stage=request.get('stage','after');lang=request.get('lang','en');fault=request.get('fault','none')
  if stage not in ('before','after') or lang not in ('tr','en') or fault not in ('none','device_delay','clock_drift','stale_calibration','delayed_input'):raise ValueError('Stage/language/fault')
  selected=g[g.task==task]
  if selected.empty:raise ValueError('Unknown task')
  # Historical-only replay ends at requested issue time. Future tasks/labels are absent.
  study=copy.copy(self.study);study.observations=g[g.issued_h<=selected.iloc[0].issued_h].copy()
  z,be,af=run(study,self.b,[person],self.b['policy'],self.b['duration']['mode'],self.b['duration']['precision'],self.b['interval_method'],scenario=None if fault=='none' else fault,before_only_task=(person,task) if stage=='before' else None)
  before=next(v for v in be if v['task']==task);after=next((v for v in af if v['task']==task),None);out=copy.deepcopy(before if stage=='before' or after is None else after)
  out.update(measurement_acquired=after is not None and after['interpretable'],requested_stage=stage,model_sha256=self.hash,data_sha256=self.b['data_sha256'],study_id=self.b['source'],origin=self.b['origin'],population=self.b['population'],duration_policy=self.b['duration'],policy_point_accepted=self.b['policy_point_accepted'],policy_evidence_accepted=self.b['policy_evidence_accepted'])
  numeric=clean(out);numeric['application_snapshot_hash']=digest(numeric)
  if ledger:
   if stage=='after':self.run({**request,'stage':'before'},ledger)
   ledgerobj=Ledger(ledger);episode=digest(dict(model=self.hash,person=person,task=task,fault=fault));ledgerobj.put(episode+':'+stage,episode,'initial' if stage=='before' else 'revision',float(out['decision_h']),numeric)
   if stage=='before':ledgerobj.put(episode+':decision',episode,'decision',float(out['decision_h']),dict(measure=bool(out['measure_recommended']),event_probability=out['event_probability'],gain_probability=out['measurement_gain_probability']))
  out=copy.deepcopy(numeric);out['lang']=lang;out['next_action']=('Check the measurement chain.' if out['state']=='C' else 'Acquire the planned measurement.' if stage=='before' and out['measure_recommended'] else 'Monitor impairment probability; this is not mission clearance.') if lang=='tr' else ('Check the measurement chain.' if out['state']=='C' else 'Acquire the planned measurement.' if stage=='before' and out['measure_recommended'] else 'Monitor impairment probability; this is not mission clearance.')
  return out
