"""Day-length experiment: mission clock differs from intrinsic tau and gravity."""
import numpy as np,pandas as pd,joblib,os
from v133.physiology import advance,Parameters
from v135.engine import Engine
from .common import ROOT,REPORT,write,circular_difference,sha

SOL_H=24.6597
DEFINITIONS={
 'earth':dict(sol_h=24.,light_period_h=24.,alternative=False),
 'mars_earth_light':dict(sol_h=SOL_H,light_period_h=24.,alternative=False),
 'mars_scheduled_light':dict(sol_h=SOL_H,light_period_h=SOL_H,alternative=True)}
def simulate(bundle,name,days=35.,step=.25):
 if name not in DEFINITIONS or days<1 or not 0<step<=.5:raise ValueError('Mars scenario')
 cfg=DEFINITIONS[name];sol=cfg['sol_h'];state=np.array([-.1,-1.1,.45,.4]);engine=Engine(bundle);rows=[];target=days*sol
 for t in np.arange(0,target,step):
  clock=24*(t%sol)/sol;asleep=bool(clock>=22 or clock<6);light_clock=24*(t%cfg['light_period_h'])/cfg['light_period_h'];lux=0. if asleep else 350. if 8<=light_clock<18 else 30.
  if cfg['alternative']and not asleep and 12<=clock<20:lux=450.
  state=advance(state,lux,asleep,step,max_step_h=.05);end=float(t+step);phase=float(np.arctan2(-state[1],state[0]));row=dict(scenario=name,time_h=end,mission_sol=end/sol,mission_clock_h=24*(end%sol)/sol,eye_lux=lux,asleep=asleep,S=state[3],x=state[0],xc=state[1],n=state[2],phase_radians=phase,phase_angle_hours=float(circular_difference(phase*24.2/(2*np.pi),end%sol,sol)),intrinsic_tau_h=24.2,sol_length_h=sol,gravity_g=1.,prediction_ms=None,radius_ms=None,interpretable=None)
  # Regular elapsed-time checkpoints, with explicit mission-clock features.
  if abs((end/8)-round(end/8))<1e-8:
   q=dict(person='MARS_RESEARCH',task=f'{end:g}',issued_h=end,now_h=end,target_h=end+2,protocol='RESEARCH_DAY_LENGTH',quality='valid',device_ok=True,calibration_id='model_only_no_measurement',instrument_sd_ms=0.,S=state[3],x=state[0],xc=state[1],n=state[2],co2_excess=0.,dose_ugy=0.,gravity_deviation=0.,last_rt=330.,history_mean=330.,history_n=0.,sin=np.sin(2*np.pi*end/sol),cos=np.cos(2*np.pi*end/sol),light_coverage=1.)
   d=engine.decide(q,'before',record=False);row.update(prediction_ms=d['prediction_ms'],radius_ms=d['radius_ms'],interpretable=d['interpretable'])
  rows.append(row)
 return pd.DataFrame(rows)
def run_mars():
 model_path=os.environ.get('TWINOVA_MODEL',ROOT/'models_v135/study/twin.joblib');b=joblib.load(model_path);frames=[simulate(b,n)for n in DEFINITIONS];f=pd.concat(frames,ignore_index=True);f.to_csv(REPORT/'mars_sol_results.csv',index=False);summary=[]
 for name,g in f.groupby('scenario',sort=False):
  late=g[g.mission_sol>=14];residual=np.unwrap(late.phase_radians.to_numpy())-(late.time_h.to_numpy()*2*np.pi/DEFINITIONS[name]['sol_h']);daily=pd.Series(residual*DEFINITIONS[name]['sol_h']/(2*np.pi),index=np.floor(late.mission_sol)).groupby(level=0).mean();summary.append(dict(scenario=name,**DEFINITIONS[name],intrinsic_tau_h=24.2,mean_sleep_pressure=float(late.S.mean()),phase_drift_h_per_sol=float(np.polyfit(daily.index,daily.values,1)[0])if len(daily)>1 else None,mean_model_forecast_ms=float(late.prediction_ms.dropna().mean()),post_warmup_sols=21,drift_convention='unwrapped oscillator angle minus external-day angle, hours per sol'))
 write(REPORT/'mars_summary.json',dict(model_sha256=sha(model_path),scenarios=summary,initial_conditions=[-.1,-1.1,.45,.4],days=35,warmup_sols=14,step_h=.25,scope='Model day-length/light-timing experiment; no counterfactual observed targets; not individual entrainment validation',gravity_unchanged_to_isolate_day_length=True,light_timing_not_treatment_advice=True,interval_scope='Earth-cohort calibration; Mars coverage not established',source='https://pmc.ncbi.nlm.nih.gov/articles/PMC1934931/'));return summary
if __name__=='__main__':run_mars()
