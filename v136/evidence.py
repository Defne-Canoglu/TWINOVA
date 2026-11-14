"""Deterministic associative evidence; never changes decisions or asks for tests."""
import math
import numpy as np,pandas as pd
from .features import history_at,window_summary
from .common import circular_difference,clean

VERSION='evidence-1.0'
LABELS={
 'H1':('Insufficient or irregular sleep','Insufficient or irregular sleep'),
 'H2':('Model-estimated phase misalignment','Model-estimated phase misalignment'),
 'H3':('Increased cognitive workload','Increased cognitive workload'),
 'H4':('Device delay / measurement conditions','Device delay / measurement conditions'),
 'H5':('Co-occurring factors','Co-occurring factors')}
def item(key,time,scope='association_only'):
 return dict(hypothesis_id=key,label_tr=LABELS[key][0],label_en=LABELS[key][1],support_level='not_evaluable',interpretation_scope=scope,supporting_evidence=[],contradicting_evidence=[],missing_evidence=[],coverage_fraction=None,window_h=None,evidence_ids=[],decision_time=float(time),last_available_time=None,rule_version=VERSION,limitations=['Association, not a cause probability'],reason_codes=[])
def fact(code,value,unit,ids,available):return dict(code=code,value=clean(value),unit=unit,evidence_ids=ids,available_h=available)
def assess(decision,history,observations,calibration=None,workload=None,sol_length_h=24.):
 d=decision;person=d['person'];now=float(d['decision_h']);g=history_at(history,person,now)
 result={key:item(key,now,'measurement_quality'if key=='H4'else'association_only')for key in LABELS}
 h1=result['H1'];windows={str(w):window_summary(g,now,w)for w in (24,48,72)};w=windows['24'];ids=[r['id']for r in w['records']if r['quality']=='valid'];last=max([r['available_h']for r in w['records']],default=None)
 h1.update(coverage_fraction=w['coverage_fraction'],window_h=24.,evidence_ids=ids,last_available_time=last,window_summaries=windows)
 prior=[]
 for days in range(1,8):
  t=now-days*24
  if t<0:continue
  summary=window_summary(history_at(history,person,t),t,24.)
  if summary['coverage_fraction']>=.9:prior.append(summary['sleep_h'])
 if w['coverage_fraction']<.9:h1['missing_evidence'].append('At least 90% observed time in the last 24 h');h1['reason_codes'].append('INSUFFICIENT_SLEEP_COVERAGE')
 elif len(prior)<3:h1['missing_evidence'].append('Three previous adequately observed 24 h personal windows');h1['reason_codes'].append('PERSONAL_SLEEP_REFERENCE_LIMITED')
 else:
  ref=float(np.median(prior));spread=max(.5,float(np.median(abs(np.asarray(prior)-ref)))*1.4826);deficit=ref-w['sleep_h'];f=fact('SLEEP_DEFICIT_FROM_PERSONAL_MEDIAN',deficit,'h',ids,last)
  h1['support_level']='strong'if deficit>=max(1.5,2*spread)else'limited'if deficit>=spread or w['sleep_onsets']>=3 else'weak'
  (h1['supporting_evidence']if h1['support_level']!='weak'else h1['contradicting_evidence']).append(f)
  h1['reference']=dict(median_sleep_h=ref,windows=len(prior),robust_scale_h=spread);h1['limitations'].append('Engineering deviation rule; not a universal sleep requirement')
  if w['sleep_onsets']>=3:h1['supporting_evidence'].append(fact('OBSERVED_SLEEP_ONSETS',w['sleep_onsets'],'count',ids,last))
 # Timing dispersion is descriptive, not another independent sleep measurement.
 def sleep_midpoint(summary):
  records=[r for r in summary['records']if r['quality']=='valid'];angles=[];weights=[]
  for r in records:
   g0=g[(g.start_h<=r['start_h'])&(g.end_h>=r['end_h'])]
   if len(g0) and g0.iloc[0].asleep==1:angles.append(((r['start_h']+r['end_h'])/2)%24*2*np.pi/24);weights.append(r['end_h']-r['start_h'])
  return float(np.angle(np.average(np.exp(1j*np.array(angles)),weights=weights))*24/(2*np.pi)%24)if weights else None
 h1['sleep_midpoint_clock_h']=sleep_midpoint(w)
 timing_prior=[sleep_midpoint(window_summary(history_at(history,person,now-days*24),now-days*24,24.))for days in range(1,8)if now-days*24>=0]
 timing_prior=[v for v in timing_prior if v is not None]
 if len(timing_prior)>=3 and h1['sleep_midpoint_clock_h']is not None:
  ref_mid=float(np.angle(np.mean(np.exp(1j*np.asarray(timing_prior)*2*np.pi/24)))*24/(2*np.pi)%24);h1['sleep_timing_context']=dict(reference_midpoint_clock_h=ref_mid,current_minus_reference_h=float(circular_difference(h1['sleep_midpoint_clock_h'],ref_mid)),reference_windows=len(timing_prior),independent_of_H1=False)
 h1['timing_convention']='Duration-weighted circular midpoint of observed sleep in the last 24 h; external 24 h study clock'
 h1['model_context']=dict(S=d.get('physiology',{}).get('S'),independent_measurement=False)
 h2=result['H2'];h2.update(coverage_fraction=w['coverage_fraction'],window_h=24.,evidence_ids=ids,last_available_time=last)
 p=d.get('physiology',{});x,xc=p.get('x'),p.get('xc');obs=observations[(observations.person==person)&(observations.issued_h<float(d['issued_h']))].copy()
 # Compare only the same external-clock slot in earlier records.
 if 'quality'in obs:obs=obs[obs.quality=='valid']
 if 'light_coverage'in obs:obs=obs[obs.light_coverage>=.9]
 if len(obs):obs=obs[np.abs(circular_difference(obs.issued_h.to_numpy()%sol_length_h,float(d['issued_h'])%sol_length_h,sol_length_h))<1.5]
 if not all(isinstance(v,(float,int))and math.isfinite(v)for v in [x,xc])or w['coverage_fraction']<.9:h2['missing_evidence'].append('Finite model oscillator and adequate input coverage');h2['reason_codes'].append('PHASE_INPUT_UNAVAILABLE')
 elif len(obs)<3:h2['missing_evidence'].append('Three previous same-clock model states');h2['reason_codes'].append('PHASE_REFERENCE_LIMITED')
 else:
  recent=obs.tail(7);phases=np.arctan2(-recent.xc.to_numpy(),recent.x.to_numpy());reference=float(np.angle(np.mean(np.exp(1j*phases))));phase=math.atan2(-xc,x);delta=abs(float(circular_difference(phase,reference,2*np.pi)))*24.2/(2*np.pi)
  h2['support_level']='strong'if delta>=1.5 else'limited'if delta>=.75 else'weak';h2['reference']=dict(phase_radians=reference,previous_records=len(recent),convention='atan2(-xc,x); same external-clock history; tau 24.2 h',measured_DLMO=False)
  (h2['supporting_evidence']if delta>=.75 else h2['contradicting_evidence']).append(fact('MODEL_PHASE_DEVIATION',delta,'h',ids,last));h2['limitations'].append('Shared sleep/light input with H1; not independent biological phase evidence')
 h3=result['H3'];h3['reason_codes'].append('NO_VALIDATED_WORKLOAD_FIELD');h3['missing_evidence'].append('Time-stamped validated workload values and their reference')
 if workload is not None and len(workload):
  required={'person','available_h','value','reference','measurement_type','record_id'}
  if required<=set(workload):
   q=workload[(workload.person==person)&(workload.available_h<=now)&(workload.available_h>now-24)&(workload.measurement_type=='validated_workload')]
   q=q[np.isfinite(q.value)&np.isfinite(q.reference)&(q.reference>0)]
   if len(q):
    ratio=float(np.median(q.value/q.reference));h3.update(support_level='strong'if ratio>=1.5 else'limited'if ratio>1.1 else'weak',coverage_fraction=None,window_h=24.,evidence_ids=q.record_id.astype(str).tolist(),last_available_time=float(q.available_h.max()),missing_evidence=[],reason_codes=['EXISTING_VALIDATED_WORKLOAD'])
    (h3['supporting_evidence']if ratio>1.1 else h3['contradicting_evidence']).append(fact('WORKLOAD_TO_REFERENCE',ratio,'ratio',h3['evidence_ids'],h3['last_available_time']))
 h4=result['H4'];agents=d.get('agents',[]);hardware=next((a for a in agents if a.get('role')=='hardware'),None);cal=calibration or {};cal_id=d.get('calibration_id');h4['evidence_ids']=[str(cal_id)]if cal_id else[];h4['last_available_time']=cal.get('available_h')
 if hardware is not None and isinstance(hardware.get('passed'),bool):
  passed=hardware['passed'];valid_time=cal.get('available_h')is not None and cal['available_h']<=now<=cal.get('valid_until_h',cal['available_h']+24)
  h4['support_level']='weak'if passed and valid_time else'limited'if not passed else'not_evaluable'
  if not passed:h4['supporting_evidence'].append(fact('DEVICE_CHECK_FAILED',True,'boolean',h4['evidence_ids'],h4['last_available_time']));h4['reason_codes'].append('DEVICE_ISSUE_NOT_SPECIFIC_DELAY')
  elif valid_time:h4['contradicting_evidence'].append(fact('DEVICE_CHECK_AND_CALIBRATION_VALID',True,'boolean',h4['evidence_ids'],h4['last_available_time']))
  else:h4['missing_evidence'].append('Calibration availability/validity interval')
 else:h4['missing_evidence'].append('Current-record device check')
 h4['limitations'].extend(['A passing check does not rule out every fault','Bench-suite results are not current-record evidence'])
 h5=result['H5'];active=[k for k in ['H1','H2','H3','H4']if result[k]['support_level']in ['strong','limited']];domains={'H1':'sleep_light','H2':'sleep_light','H3':'workload','H4':'device'}
 known=[k for k in ['H1','H2','H3','H4']if result[k]['support_level']!='not_evaluable'];h5.update(components=active,available_components=known,dependency_groups=domains,missing_evidence=[k for k in ['H1','H2','H3','H4']if k not in known],window_h=24.,coverage_fraction=None)
 if len(known)>=2:h5['support_level']='strong'if len(active)>=2 and len({domains[k]for k in active})>=2 else'limited'if len(active)>=2 else'weak'
 h5['reason_codes']=['CO_OCCURRENCE_NOT_INTERACTION'];h5['limitations'].append('H1/H2 share one sleep-light evidence domain; no summed probability')
 h5['evidence_ids']=sorted({v for k in active for v in result[k]['evidence_ids']});h5['last_available_time']=max([result[k]['last_available_time']for k in active if result[k]['last_available_time']is not None],default=None)
 appropriate=d.get('interpretable')is True and d.get('state')in ['A','B','N']
 if not appropriate:
  for k in ['H1','H2','H3','H5']:result[k]['interpretation_scope']='context_only';result[k]['limitations'].append('Performance-change attribution withheld: input/measurement not interpretable')
 return clean(dict(schema_version=1,rule_version=VERSION,person=person,task=d['task'],decision_time=now,performance_attribution_allowed=appropriate,causal_claim=False,probabilities_calibrated=False,measurement_requests=0,hypotheses=list(result.values())))
