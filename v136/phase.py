"""Documented Forger RHS solved independently with SciPy + matched-data audit."""
import sys,math
from pathlib import Path
import numpy as np,pandas as pd
from scipy.integrate import solve_ivp
from v133.physiology import advance,Parameters
from .common import ROOT,REPORT,write,circular_difference,sha

def reference_rhs(t,y,lux):
 x,xc,n,S=y;a=.05*(lux/9500.)**.5;B=33.75*(1-n)*a*(1-.4*x)*(1-.4*xc)
 return [np.pi/12*(xc+B),np.pi/12*(.23*(xc-4*xc**3/3)-x*((24/(.99669*24.2))**2+.55*B)),60*(a*(1-n)-.0075*n),0.]

def pair_profiles(light_profiles,melatonin_profiles):
 required={'person','clock','timezone','protocol','unit','start_h','end_h'};matched=[];audit=[]
 def valid(p):
  return required<=set(p) and all(p[k] is not None and str(p[k]).strip() for k in ['person','clock','timezone','protocol','unit']) and np.isfinite([p['start_h'],p['end_h']]).all() and p['start_h']<p['end_h']
 for m in melatonin_profiles:
  reason=None;candidates=[]
  if not valid(m):reason='Missing person/clock/timezone/protocol/unit or valid time window'
  else:
   candidates=[l for l in light_profiles if valid(l) and l['person']==m['person'] and l['clock']==m['clock'] and l['timezone']==m['timezone'] and l['unit']=='lux' and l['start_h']<m['end_h'] and l['end_h']>=m['start_h'] and l.get('prediction_available_h',float('inf'))<=m['start_h']]
   if not candidates:reason='No available same-person same-clock overlapping light history'
   elif m['unit']!='pg/mL' or m.get('dim_light_confirmed')is not True:reason='Assay unit/dim-light protocol not confirmed'
   elif not m.get('prediction_available_before_assay',False):reason='No prospectively available phase prediction'
  status='not_evaluable' if reason else 'matched'
  if not reason:matched.append((candidates[0],m))
  audit.append(dict(profile=m.get('profile',m.get('person','unknown')),status=status,reason=reason or 'Identity, clock, overlap, availability and protocol verified'))
 return matched,audit

def compare_profiles(light_profiles,melatonin_profiles):
 # Optional explicit matched-data contract; never guesses linkage in raw CSV.
 matched,audit=pair_profiles(light_profiles,melatonin_profiles);rows=[]
 sys.path.insert(0,str(ROOT/'runtime'));from astrotwin.phase_v8 import melatonin_crossing
 for light,assay in matched:
  t=np.asarray(light.get('time_h',[]),float);x=np.asarray(light.get('model_x',[]),float)
  mt=np.asarray(assay.get('time_h',[]),float);values=np.asarray(assay.get('values_pg_ml',[]),float)
  if len(t)!=len(x) or len(t)<3 or len(mt)!=len(values) or len(mt)<3 or not np.isfinite(np.r_[t,x,mt,values]).all():
   audit.append(dict(profile=assay.get('profile',assay['person']),status='not_evaluable',reason='Missing finite phase/assay series'));continue
  candidates=np.where((x[1:-1]<=x[:-2])&(x[1:-1]<x[2:]))[0]+1
  if not len(candidates):audit.append(dict(profile=assay.get('profile',assay['person']),status='not_evaluable',reason='No interior model CBT minimum'));continue
  for threshold in [2.3,4.]:
   crossing=melatonin_crossing(mt,values,threshold=threshold,unit='pg/mL');measured=crossing.get('crossing_h')
   # Support the released crossing API name, explicitly test below.
   if measured is None:measured=crossing.get('time_h')
   if measured is None:continue
   for offset in [6.,7.,8.]:
    predictions=t[candidates]-offset;chosen=float(predictions[np.argmin(abs(predictions-(assay['start_h']+assay['end_h'])/2))])
    rows.append(dict(person=assay['person'],threshold_pg_ml=threshold,cbt_dlmo_offset_h=offset,predicted_dlmo_h=chosen,observed_crossing_h=float(measured),circular_error_h=float(circular_difference(chosen,measured)),scope='Assay-window selected forecast cycle; model-offset sensitivity, not causal validation'))
 return rows,audit

def phase_checks():
 initial=np.array([-.1,-1.1,.45,.4]);rows=[]
 for lux in [0.,30.,350.,1000.]:
  reference=solve_ivp(lambda t,y:reference_rhs(t,y,lux),(0,8),initial,rtol=1e-11,atol=1e-12).y[:,-1]
  for step in [.05,.025,.0125]:
   actual=advance(initial,lux,False,8.,max_step_h=step);error=float(np.max(abs(actual[:3]-reference[:3])));rows.append(dict(lux=lux,max_step_h=step,max_state_error=error,phase_error_rad=float(abs(circular_difference(np.arctan2(-actual[1],actual[0]),np.arctan2(-reference[1],reference[0]),2*np.pi)))))
 assert max(r['max_state_error']for r in rows)<1e-4
 source=ROOT/'sampled_raw/melatonin.csv';f=pd.read_csv(source);profiles=[dict(profile='sampled_raw/melatonin.csv',unit='pg/mL',clock='calendar_datetime',person=None)]
 # The released sample has no identity, matching light history or dim-light
 # protocol. Inference of these properties from the file name is forbidden.
 _,audit=pair_profiles([],profiles);crossings=[]
 sys.path.insert(0,str(ROOT/'runtime'));from astrotwin.phase_v8 import melatonin_crossing
 t=pd.to_datetime(f.datetime,format='mixed');hours=(t-t.iloc[0]).dt.total_seconds()/3600
 for threshold in [2.3,4.]:crossings.append(dict(threshold_pg_ml=threshold,**melatonin_crossing(hours,f.melatonin,threshold=threshold,unit='pg/mL')))
 pd.DataFrame(audit).to_csv(REPORT/'phase_matching_audit.csv',index=False);pd.DataFrame(rows).to_csv(REPORT/'phase_numerical_reference.csv',index=False)
 from .mars import simulate
 import joblib
 fmodel=simulate(joblib.load(ROOT/'models_v135/study/twin.joblib'),'earth',days=4);last=fmodel[fmodel.time_h>72];cbt=float(last.loc[last.x.idxmin(),'time_h']);pd.DataFrame([dict(offset_h=o,model_cbt_min_h=cbt,assumed_dlmo_h=(cbt-o)%24,scope='Model convention sensitivity; not an observed phase error')for o in [6.,7.,8.]]).to_csv(REPORT/'phase_offset_sensitivity.csv',index=False)
 write(REPORT/'phase_consistency_results.json',dict(numerical_reference=dict(method='Independent SciPy RK45 documented Forger99 RHS; homeostat checked separately',source='https://arcascope.github.io/circadian/api/models.html',results=rows,passed=True),biological=dict(status='not_evaluable',matched_people=0,reason='Available assay sample lacks person linkage, matching light series and confirmed dim-light protocol',measured_phase_error_h=None),assay_threshold_checks=crossings,assay_sha256=sha(source),convention='atan2(-xc,x) is a geometrical model angle; model CBT minimum is minimum x; DLMO=CBTmin-7h is a reference assumption, not observed DLMO',cbt_to_dlmo_offset_sensitivity_h=[6,7,8],circular_test_2350_0010_h=float(abs(circular_difference(23+50/60,10/60)))));return rows
if __name__=='__main__':phase_checks()


def biological_summary(light_profiles,melatonin_profiles):
 rows,audit=compare_profiles(light_profiles,melatonin_profiles)
 if not rows:return dict(status='not_evaluable',people=0,valid_crossings=0,audit=audit,reason='No protocol-valid matched crossing/forecast pair',circular_MAE_h=None)
 f=pd.DataFrame(rows);results=[]
 from v133.model import boot
 for (threshold,offset),g in f.groupby(['threshold_pg_ml','cbt_dlmo_offset_h']):
  pm=g.groupby('person').circular_error_h.apply(lambda v:float(abs(v).mean()));ci=boot(pm);results.append(dict(threshold_pg_ml=float(threshold),offset_h=float(offset),people=len(pm),pairs=len(g),person_macro_circular_MAE_h=float(pm.mean()),ci95=ci))
 return dict(status='computed',people=int(f.person.nunique()),valid_crossings=len(f.drop_duplicates(['person','threshold_pg_ml','observed_crossing_h'])),audit=audit,results=results,scope='Matched assay comparison under explicit offset assumptions; not clinical validation')
