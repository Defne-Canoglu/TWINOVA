import unittest,copy,math,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from v133.data import BASE
from v134.models import conformal
from v135.engine import Engine,SHORT,AFTER
from v135.trials import summarize,prefix,validate
from v133.model import ConstantBinary

class Predictor:
 def predict(self,x):
  rt=.8*x.short_rt.to_numpy()+60 if 'short_rt' in x else 300+20*x.S.to_numpy()+.1*x.history_mean.to_numpy();return np.column_stack([rt,np.full(len(x),3.),np.full(len(x),.05),np.full(len(x),2.),np.zeros(len(x))])
class Gain:
 def predict(self,x):return 20*x.S.to_numpy()
class Scale:
 def predict(self,x):return np.full(len(x),5.)

def bundle():
 return dict(models={k:dict(model=Predictor(),cols=c) for k,c in [('before',BASE),('after',AFTER)]},events={'before':ConstantBinary(.2),'after':ConstantBinary(.7)},event_cols={'before':BASE,'after':AFTER},target_names=['rt','response_speed','lapse_fraction','slow_speed','early_fraction'],gain=Gain(),gain_probability=ConstantBinary(.1),alarm_threshold=.5,history_mode='history',policy={'kind':'always'},personal_config=dict(intercept_var=9,slope_var=25,noise_var=144,decay_h=48),interval_method='constant',intervals={k:dict(constant=20.,scale=Scale(),normalized_q=4.,conditional={}) for k in ['before','after']})
def observation(task='a',now=8.,stage='before'):
 d={k:0. for k in BASE};d.update(S=.5,light_coverage=1.,person='p',task=task,issued_h=now,now_h=now,target_h=now+2,quality='valid',device_ok=True,protocol='demo',calibration_id='cal',instrument_sd_ms=.2)
 if stage=='after':d.update(short_rt=350.,short_response_speed=3.,short_lapse_fraction=.1,short_slow_speed=2.,short_early_fraction=0.,short_n=24.,trial_metrics_available=1.,short_available_h=now+.04,now_h=now+.04)
 return d

def rows(n=60):
 return [dict(source='s',person='p',task='a',session='short',trial=i,protocol='ENGINEERING_SHORT_5MIN',device='d',stimulus_h=8+i*5/3600,response_h=8+i*5/3600+.3/3600,available_h=8+i*5/3600+.3/3600,raw_rt_ms=305.,rt_ms=300.,correction_ms=5.,no_response=False,early=False,quality='valid',calibration_id='cal',calibration_available_h=7.,calibration_valid_until_h=31.) for i in range(n)]

class Core(unittest.TestCase):
 def test_target_injection_rejected(self):
  for key in ['target_rt','target_lapses','target_response_speed','origin','source']:
   o=observation();o[key]=400
   with self.assertRaises(ValueError):Engine(bundle()).decide(o)
 def test_unacquired_short_rejected(self):
  o=observation();o['short_rt']=300
  with self.assertRaises(ValueError):Engine(bundle()).decide(o)
 def test_after_before_available_rejected(self):
  o=observation(stage='after');o['now_h']=8
  with self.assertRaises(ValueError):Engine(bundle()).decide(o,'after')
 def test_nan_short_rejected(self):
  o=observation(stage='after');o['short_rt']=math.nan
  with self.assertRaises(ValueError):Engine(bundle()).decide(o,'after')
 def test_future_label_rejected(self):
  e=Engine(bundle());e.decide(observation(stage='after'),'after')
  with self.assertRaises(ValueError):e.ingest(dict(person='p',task='a',available_h=10.,quality='valid',rt=380.),9.)
 def test_unacquired_label_rejected(self):
  with self.assertRaises(ValueError):Engine(bundle()).ingest(dict(person='p',task='a',available_h=7.,quality='valid',rt=400.),8)
 def test_history_only_after_label_available(self):
  e=Engine(bundle());e.decide(observation(stage='after'),'after');e.ingest(dict(person='p',task='a',available_h=10.,quality='valid',rt=380.),10.);out=e.decide(observation('b',16.));self.assertEqual(out['history_count'],1);self.assertEqual(out['model_features']['last_rt'],380.)
 def test_repeated_label_idempotent(self):
  e=Engine(bundle());e.decide(observation(stage='after'),'after');l=dict(person='p',task='a',available_h=10.,quality='valid',rt=380.);e.ingest(l,10.);e.ingest(l,10.);self.assertEqual(len(e.states['p'].labels),1)
 def test_supplied_history_cannot_override(self):
  a=observation();b=copy.deepcopy(a);b.update(last_rt=9999,history_mean=9999,history_n=99);self.assertEqual(Engine(bundle()).decide(a)['prediction_ms'],Engine(bundle()).decide(b)['prediction_ms'])
 def test_event_and_gain_separate(self):
  out=Engine(bundle()).decide(observation());self.assertEqual(out['event_probability'],.2);self.assertEqual(out['measurement_gain_probability'],.1);self.assertEqual(out['measurement_gain_ms'],10.)
 def test_invalid_measurement_falls_back(self):
  a=observation(stage='after');a['device_ok']=False;out=Engine(bundle()).decide(a,'after');self.assertEqual(out['state'],'C');self.assertEqual(out['stage'],'before');self.assertFalse(out['measure_recommended'])
 def test_online_updates_only_after_label(self):
  e=Engine(bundle());before=e.decide(observation());e.bind_before(before);after=e.decide(observation(stage='after'),'after');self.assertFalse(e.states['p'].online_q);e.ingest(dict(person='p',task='a',available_h=10.,quality='valid',rt=500.),10.);self.assertGreater(e.states['p'].online_q['after'],20.);self.assertGreater(e.states['p'].online_q['before'],20.)
 def test_conditional_fallback(self):
  o=Engine(bundle()).decide(observation(),interval_method='conditional');self.assertEqual(o['interval_fallback'],'population_constant_insufficient_people');self.assertEqual(o['radius_ms'],20.)
 def test_conditional_supported_cell(self):
  b=bundle();b['intervals']['before']['conditional']['am:0']=dict(eligible=True,radius=30.);o=Engine(b).decide(observation(),interval_method='conditional');self.assertIsNone(o['interval_fallback']);self.assertEqual(o['radius_ms'],30.)
 def test_normalized_radius(self):self.assertEqual(Engine(bundle()).decide(observation(),interval_method='normalized')['radius_ms'],20.)
 def test_infinite_quantile_preserved(self):
  b=bundle();b['intervals']['before']['constant']=math.inf;o=Engine(b).decide(observation());self.assertEqual(o['prediction_interval'],[None,None])
 def test_finite_sample_quantile(self):self.assertEqual(conformal(range(20)),18);self.assertTrue(np.isinf(conformal(range(8))))
 def test_reciprocal_metric_not_reciprocal_median(self):
  d=rows(3)
  for r,rt in zip(d,[100,300,900]):r['rt_ms']=rt
  s=summarize(d);self.assertNotAlmostEqual(s['response_speed'],1000/s['rt'])
 def test_shared_duration_kernel_minimum_time(self):
  r=rows(24);d=Engine.measurement_decision(r,r[-1]['available_h'],'adaptive',.18);self.assertEqual(d['action'],'extend');d=Engine.measurement_decision(r,8+2/60,'adaptive',.18);self.assertEqual(d['action'],'stop')
 def test_prefix_future_poison_invariance(self):
  a=rows();b=copy.deepcopy(a)
  for r in b[24:]:r['rt_ms']=30000
  x,used=prefix(a,9.,'adaptive',.18);y,used2=prefix(b,9.,'adaptive',.18);self.assertEqual(len(used),24);self.assertEqual(x,y)
 def test_prefix_availability_gate(self):
  d=rows();s,used=prefix(d,d[10]['available_h'],'long');self.assertEqual(len(used),11);self.assertEqual(s['n'],11)
 def test_timeout_is_lapse_and_early_excluded(self):
  d=rows(3);d[0].update(rt_ms=80,early=True);d[1].update(rt_ms=5000,no_response=True);s=summarize(d);self.assertEqual(s['n'],2);self.assertEqual(s['lapses'],1)
 def test_agents_preserved(self):self.assertEqual({x['role'] for x in Engine(bundle()).decide(observation())['agents']},{'physiology','hardware','learning','reference','planner'})
 def test_no_automatic_mission_clearance(self):self.assertIsNone(Engine(bundle()).decide(observation())['mission_clearance'])
 def test_population_ignores_acquired_history(self):
  e=Engine(bundle());e.decide(observation(stage='after'),'after');e.ingest(dict(person='p',task='a',available_h=10,quality='valid',rt=900),10);o=e.decide(observation('b',16),history_mode='population');self.assertEqual(o['history_count'],0)
 def test_random_policy_deterministic(self):
  p=dict(kind='random',budget=.5);a=Engine(bundle()).decide(observation(),policy=p);b=Engine(bundle()).decide(observation(),policy=p);self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
