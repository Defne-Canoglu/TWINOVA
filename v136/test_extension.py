"""Boundary-focused verification of the extension, independent of its reports."""
import copy,json,math,tempfile,unittest,subprocess,sys,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np,pandas as pd
from http.server import ThreadingHTTPServer
from v135.metrics import score
from v134.models import conformal
from v133.model import boot
from .application import Application
from .common import ROOT,REPORT,sha,circular_difference,write
from .evidence import assess
from .features import history_at,window_summary,homeostat,recompute_state
from .operations import progress,CommunicationQueue,hardware_summary
from .phase import pair_profiles,compare_profiles,reference_rhs
from .stability import bootstrap_scores
from .personal import WakeRateAdapter
from .scenarios import CATALOG,transform

class Extension(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.app=Application();cls.q=dict(person=cls.app.b['splits']['test'][0],task='13-16',stage='after',lang='tr');cls.d=cls.app.run(cls.q)
 def hypotheses(self,d=None,history=None,observations=None,**kw):
  d=d or self.d;return {h['hypothesis_id']:h for h in assess(d,history if history is not None else self.app.history,observations if observations is not None else self.app.study.observations,**kw)['hypotheses']}
 def test_default_output_identity(self):
  base=self.app.run(self.q,evidence=False);enriched=self.app.run(self.q);self.assertEqual(base,{k:enriched[k]for k in base})
 def test_evidence_does_not_create_measurements(self):
  out=self.app.run(self.q);self.assertEqual(out['evidence_assessment']['measurement_requests'],0);self.assertEqual(out['routine_progress']['new_measurements_created'],0)
 def test_no_new_model_or_probability_in_explanation(self):
  ev=self.d['evidence_assessment'];self.assertFalse(ev['causal_claim']);self.assertFalse(ev['probabilities_calibrated']);self.assertEqual(len(ev['hypotheses']),5)
 def test_schema_every_hypothesis(self):
  keys={'hypothesis_id','label_tr','label_en','support_level','interpretation_scope','supporting_evidence','contradicting_evidence','missing_evidence','coverage_fraction','evidence_ids','decision_time','last_available_time','rule_version','limitations','reason_codes'}
  for h in self.d['evidence_assessment']['hypotheses']:self.assertTrue(keys<=set(h));self.assertIn(h['support_level'],['strong','limited','weak','not_evaluable'])
 def test_H3_not_task_count(self):
  self.assertEqual(self.hypotheses()['H3']['support_level'],'not_evaluable')
 def test_H3_existing_validated_support(self):
  w=pd.DataFrame([dict(person=self.d['person'],available_h=self.d['decision_h']-1,value=80,reference=40,measurement_type='validated_workload',record_id='W1')]);self.assertEqual(self.hypotheses(workload=w)['H3']['support_level'],'strong')
 def test_H3_counterevidence(self):
  w=pd.DataFrame([dict(person=self.d['person'],available_h=self.d['decision_h']-1,value=20,reference=40,measurement_type='validated_workload',record_id='W1')]);h=self.hypotheses(workload=w)['H3'];self.assertEqual(h['support_level'],'weak');self.assertTrue(h['contradicting_evidence'])
 def test_H3_future_not_used(self):
  w=pd.DataFrame([dict(person=self.d['person'],available_h=self.d['decision_h']+1,value=80,reference=40,measurement_type='validated_workload',record_id='W1')]);self.assertEqual(self.hypotheses(workload=w)['H3']['support_level'],'not_evaluable')
 def test_H3_invalid_not_normal(self):
  w=pd.DataFrame([dict(person=self.d['person'],available_h=self.d['decision_h']-1,value=float('nan'),reference=0,measurement_type='validated_workload',record_id='W1')]);self.assertEqual(self.hypotheses(workload=w)['H3']['support_level'],'not_evaluable')
 def test_H1_sleep_reduction_support(self):
  h=self.app.history.copy();mask=(h.person==self.d['person'])&(h.end_h>self.d['issued_h']-24)&(h.end_h<=self.d['issued_h']);h.loc[mask,'asleep']=0;self.assertEqual(self.hypotheses(history=h)['H1']['support_level'],'strong')
 def test_H1_missing_not_wake(self):
  h=self.app.history.copy();h.loc[(h.person==self.d['person'])&(h.end_h>self.d['issued_h']-24),'quality']='missing';x=self.hypotheses(history=h)['H1'];self.assertEqual(x['support_level'],'not_evaluable');self.assertEqual(x['coverage_fraction'],0)
 def test_H1_early_reference_unavailable(self):
  d=self.app.run({**self.q,'task':'00-8'});self.assertEqual(self.hypotheses(d=d)['H1']['support_level'],'not_evaluable')
 def test_H2_rotated_state_support(self):
  d=copy.deepcopy(self.d);d['physiology']['x']*=-1;d['physiology']['xc']*=-1;self.assertEqual(self.hypotheses(d=d)['H2']['support_level'],'strong')
 def test_H2_nonfinite_missing(self):
  d=copy.deepcopy(self.d);d['physiology']['x']=float('nan');self.assertEqual(self.hypotheses(d=d)['H2']['support_level'],'not_evaluable')
 def test_C_context_without_device_accusation(self):
  d=copy.deepcopy(self.d);d.update(state='C',interpretable=False);h=self.hypotheses(d=d)
  for k in ['H1','H2','H3','H5']:self.assertEqual(h[k]['interpretation_scope'],'context_only')
  self.assertNotEqual(h['H4']['support_level'],'strong')
 def test_H4_failed_not_specific_delay(self):
  d=copy.deepcopy(self.d);d['agents']=[dict(role='hardware',passed=False)];h=self.hypotheses(d=d)['H4'];self.assertEqual(h['support_level'],'limited');self.assertEqual(h['supporting_evidence'][0]['value'],True)
 def test_H4_valid_calibration_counterevidence(self):
  h=self.hypotheses(calibration=dict(available_h=self.d['issued_h']-1,valid_until_h=self.d['decision_h']+1))['H4'];self.assertEqual(h['support_level'],'weak')
 def test_H4_no_calibration_unknown(self):self.assertEqual(self.hypotheses()['H4']['support_level'],'not_evaluable')
 def test_H5_shared_sources_not_independent(self):
  d=copy.deepcopy(self.d);d['physiology']['x']*=-1;d['physiology']['xc']*=-1;h=self.app.history.copy();h.loc[(h.person==d['person'])&(h.end_h>d['issued_h']-24),'asleep']=0;z=self.hypotheses(d=d,history=h)['H5'];self.assertEqual(z['support_level'],'limited');self.assertNotIn('probability',z)
 def test_future_history_poison(self):
  h=self.app.history.copy();mask=h.available_h>self.d['decision_h'];h.loc[mask,'asleep']=0;self.assertEqual(self.hypotheses(),self.hypotheses(history=h))
 def test_available_evidence_time(self):
  for h in self.d['evidence_assessment']['hypotheses']:
   if h['last_available_time']is not None:self.assertLessEqual(h['last_available_time'],self.d['decision_h'])
 def test_overlap_rejected(self):
  h=history_at(self.app.history,self.d['person'],24);dup=pd.concat([h,h.iloc[[0]]]);self.assertRaises(ValueError,history_at,dup,self.d['person'],24)
 def test_forward_fill_not_observed_coverage(self):
  h=history_at(self.app.history,self.d['person'],24);h.loc[h.end_h>23,'quality']='missing';w=window_summary(h,24);self.assertLess(w['coverage_fraction'],1);self.assertGreater(w['coverage_fraction'],.9)
 def test_long_missing_refused(self):
  h=self.app.history.copy();h.loc[(h.person==self.d['person'])&(h.end_h>self.d['issued_h']-3),'quality']='missing';self.assertRaises(ValueError,recompute_state,h,self.d['person'],self.d['issued_h'])
 def test_homeostat_analytic(self):
  self.assertAlmostEqual(homeostat(.4,True,4.2),.4/math.e);self.assertAlmostEqual(homeostat(.4,False,18.2),1-.6/math.e)
 def test_day_boundary_circular(self):self.assertAlmostEqual(abs(float(circular_difference(23+50/60,10/60))),1/3)
 def test_progress_duplicate_future_failed(self):
  r=dict(person='P',task='T',issued_h=23,available_h=23,completed=True,valid=True);x=progress([r,r,{**r,'task':'F','available_h':99},{**r,'task':'BAD','valid':False}],24);self.assertEqual(x['completed_records'],1);self.assertEqual(x['consecutive_observed_days'],1)
 def test_progress_missing_day_break(self):
  records=[dict(person='P',task=str(t),issued_h=t,available_h=t,completed=True,valid=True)for t in [1,73]];self.assertEqual(progress(records,74)['consecutive_observed_days'],1)
 def test_progress_timezone(self):
  r=dict(person='P',task='T',issued_h=23,available_h=23,completed=True,valid=True);self.assertEqual(progress([r],24,timezone_offset_h=2)['consecutive_observed_days'],1)
 def test_virtual_comms_roundtrip(self):
  c=CommunicationQueue(1200);c.submit({'record':'r'});self.assertEqual(c.advance(1199)['delivered'],[]);self.assertEqual(c.advance(1)['delivered'][0]['kind'],'ground_received');x=c.advance(1200);self.assertEqual(x['pending'],0);self.assertEqual(x['delivered'][-1]['kind'],'reply_received');self.assertFalse(x['local_inference_delayed'])
 def test_comms_invalid(self):
  for v in [-1,float('nan'),7201]:self.assertRaises(ValueError,CommunicationQueue,v)
 def test_hardware_truth_vs_acceptance(self):
  h=hardware_summary();self.assertEqual(h['matches_expectation'],48);self.assertEqual(h['measurement_accepts'],4);self.assertEqual(h['measurement_rejects'],44);self.assertFalse(h['physical_run'])
 def test_hardware_missing_unknown(self):self.assertEqual(hardware_summary(ROOT/'not_exists.csv')['status'],'unavailable')
 def test_hardware_corrupted_proof_safe(self):
  with tempfile.TemporaryDirectory()as td:
   p=Path(td)/'bench_results.csv';p.write_text('passed,expected_pass,matches_expectation,fault,device,condition\nTrue,True,True,none,D,idle\n');p.with_name('execution.json').write_text('{broken');self.assertEqual(hardware_summary(p)['status'],'invalid')
 def test_H5_independent_domains_support(self):
  d=copy.deepcopy(self.d);d['agents']=[dict(role='hardware',passed=False)];h=self.app.history.copy();h.loc[(h.person==d['person'])&(h.end_h>d['issued_h']-24),'asleep']=0;self.assertEqual(self.hypotheses(d=d,history=h)['H5']['support_level'],'strong')
 def test_H5_insufficient_domains_unknown(self):
  d=self.app.run({**self.q,'task':'00-8'});self.assertEqual(self.hypotheses(d=d)['H5']['support_level'],'not_evaluable')
 def test_scenario_does_not_edit_source(self):
  original=sha(self.app.data/'history.csv.gz');out=self.app.scenario({**self.q,'scenario':'radiation_exposure'});self.assertFalse(out['scenario']['engine_forced_state']);self.assertNotEqual(out['state'],'B');self.assertEqual(sha(self.app.data/'history.csv.gz'),original)
 def test_radiation_does_not_change_light(self):
  h=transform(self.app.history,self.d['person'],self.d['issued_h'],'radiation_exposure');o=history_at(self.app.history,self.d['person'],self.d['issued_h']);np.testing.assert_array_equal(h.eye_lux,o.eye_lux)
 def test_short_missing_retains_flag(self):self.assertEqual(self.app.scenario({**self.q,'scenario':'short_missing'})['state'],'C')
 def test_scenario_target_injection(self):self.assertRaises(ValueError,self.app.scenario,{**self.q,'scenario':'nominal','rt':10})
 def test_reference_no_false_biological_phase(self):
  q=json.loads((REPORT/'phase_consistency_results.json').read_text());self.assertEqual(q['biological']['matched_people'],0);self.assertIsNone(q['biological']['measured_phase_error_h'])
 def test_phase_pair_strict_identity(self):
  common=dict(person='p',clock='mission',timezone='UTC',protocol='test',start_h=0,end_h=24);l={**common,'unit':'lux','prediction_available_h':0};m={**common,'unit':'pg/mL','dim_light_confirmed':True,'prediction_available_before_assay':True};self.assertEqual(len(pair_profiles([l],[m])[0]),1)
  for k in ['person','clock','timezone']:
   self.assertEqual(len(pair_profiles([{**l,k:'other'}],[m])[0]),0)
  self.assertEqual(len(pair_profiles([l],[{**m,'person':None}])[0]),0)
 def test_phase_no_overlap_or_future_prediction(self):
  l=dict(person='p',clock='m',timezone='UTC',protocol='P',unit='lux',start_h=0,end_h=5,prediction_available_h=0);m={**l,'unit':'pg/mL','start_h':6,'end_h':8,'dim_light_confirmed':True,'prediction_available_before_assay':True};self.assertEqual(pair_profiles([l],[m])[0],[])
 def test_phase_comparison_fixture(self):
  common=dict(person='p',clock='mission',timezone='UTC',protocol='test',start_h=0,end_h=24);l={**common,'unit':'lux','prediction_available_h':0,'time_h':[0,4,8,12,16,20,24],'model_x':[1,0,-1,0,1,0,-1]};m={**common,'unit':'pg/mL','dim_light_confirmed':True,'prediction_available_before_assay':True,'time_h':[0,1,2,3,4],'values_pg_ml':[0,1,3,5,6]};rows,_=compare_profiles([l],[m]);self.assertEqual(len(rows),6);self.assertTrue(all(abs(r['circular_error_h'])<=12 for r in rows))
 def test_bootstrap_rank_independent(self):
  values=np.arange(1,21);q=bootstrap_scores(values,10,replicates=1,seed=99);rng=np.random.default_rng(99);draw=values[rng.integers(0,20,10)].tolist();draw.sort();self.assertEqual(q[0],draw[9])
 def test_small_calibration_infinite(self):self.assertTrue(np.isinf(bootstrap_scores([1,2,3,4],4)).all())
 def test_repeated_bootstrap_people_not_collapsed(self):self.assertEqual(bootstrap_scores([2],10,replicates=1)[0],2)
 def test_metrics_independent_fixture(self):
  # Errors 1,3,2,4; person means 2 and 3. Coverage all-tasks: first false, second true.
  f=pd.DataFrame(dict(person=['a','a','b','b'],event=[1,1,0,0],alarm=[1,0,1,0],event_probability=[.8,.2,.7,.1],target_rt=[101,103,102,104],prediction_ms=[100]*4,radius_ms=[2,2,4,4],measured=[True]*4,duration_min=[2]*4,reference_duration_min=[5]*4,interpretable=[True]*4));z=score(f,False);self.assertEqual(z['MAE_ms'],2.5);self.assertEqual(z['task_coverage'],.75);self.assertEqual(z['person_all_tasks_coverage'],.5);self.assertEqual([z[k]for k in ['TP','FN','FP','TN']],[1,1,1,1])
 def test_personal_minimum_information(self):
  x,info=WakeRateAdapter().adjust({'now_h':1},[],None,{'rt':300});self.assertEqual(x['rt'],300);self.assertFalse(info['identified_wake_rate'])
 def test_no_test_people_in_training_folds(self):
  test=set(self.app.b['splits']['test']);folds=json.loads((REPORT/'fold_audit.json').read_text())
  for f in folds:self.assertFalse(set(f['train'])&set(f['validation']));self.assertFalse(test&set(f['train']+f['validation']))
 def test_research_common_task_keys(self):
  for stage in ['before','after']:
   ref=pd.read_csv(REPORT/f'history_{stage}_predictions.csv');hp=pd.read_csv(REPORT/f'history_physiology_{stage}_predictions.csv');self.assertEqual(list(zip(ref.person,ref.task)),list(zip(hp.person,hp.task)))
 def test_baseline_data_model_reports_immutable(self):
  hashes=json.loads((REPORT/'baseline_file_hashes.json').read_text())
  for name,expected in hashes.items():self.assertEqual(sha(ROOT/name),expected,name)

class HTTP(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from .service import Handler
  class Quiet(Handler):
   def log_message(self,*a):return
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Quiet);threading.Thread(target=cls.server.serve_forever,daemon=True).start();cls.base=f'http://127.0.0.1:{cls.server.server_port}'
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close()
 def get(self,path):return json.load(urllib.request.urlopen(self.base+path,timeout=30))
 def test_api_cli_same_decision(self):
  q={'person':'C135-108','task':'13-16','stage':'after','lang':'tr'};out=subprocess.check_output([sys.executable,'-m','v136.cli','--request',json.dumps(q)],cwd=ROOT,text=True);api=self.get('/api/v135?person=C135-108&task=13-16&stage=after&lang=tr');self.assertEqual(json.loads(out),api)
 def test_health_runtime(self):self.assertTrue(self.get('/api/v136/health')['model_loaded'])
 def test_duplicate_query_rejected(self):
  with self.assertRaises(urllib.error.HTTPError)as c:self.get('/api/v135?task=00-8&task=00-8')
  self.assertEqual(c.exception.code,400)
 def test_api_bad_json_rejected(self):
  q=urllib.request.Request(self.base+'/api/v135',data=b'{broken',headers={'Content-Type':'application/json'})
  with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(q)
 def test_mars_cli_same_data(self):
  a=self.get('/api/v136/mars?scenario=earth');b=json.loads(subprocess.check_output([sys.executable,'-m','v136.cli','--mars','earth'],cwd=ROOT,text=True));self.assertEqual(a,b)
 def test_comms_does_not_block_inference(self):
  c=self.get('/api/v136/comms?one_way_s=1200&advance_s=0');self.assertEqual(c['pending'],1);self.assertTrue(self.get('/api/v136/health')['local_inference'])
 def test_catalog_offline_embedded(self):
  text=urllib.request.urlopen(self.base+'/').read().decode();self.assertNotIn('<script src=',text);self.assertIn('SPACE_RESOURCES',text);self.assertIn('radiation',text)

if __name__=='__main__':unittest.main(verbosity=2)
