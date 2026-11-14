"""Event-level virtual plant and identical native serial decoder for real capture.
No firmware binary execution or hardware-in-loop run is claimed by the simulator.
"""
from dataclasses import dataclass,asdict
import math,csv,io
import numpy as np,pandas as pd
from v132.timing import pair,calibrate
FAULTS=('none','display_delay','hid_delay','drift','missing_optical','missing_contact','duplicate','clock_reset','overflow','disconnect','stale_calibration','actuator_stuck')
@dataclass(frozen=True)
class Device:
 name:str='VIRTUAL_D1';display_hz:float=60.;scan_fraction:float=.3;diode_tau_ms:float=.02;threshold_fraction:float=.5;contact_debounce_ms:float=.1;hid_poll_ms:float=1.;usb_ms:float=1.;clock_ppm:float=25.;tick_us:float=.5
 def __post_init__(self):
  if not self.name or not all(math.isfinite(v) for k,v in asdict(self).items() if k!='name') or self.display_hz<=0 or not 0<=self.scan_fraction<=1 or not 0<self.threshold_fraction<1 or min(self.diode_tau_ms,self.contact_debounce_ms,self.hid_poll_ms,self.usb_ms,self.tick_us)<=0 or abs(self.clock_ppm)>10000:raise ValueError('device configuration')
class Supervisor:
 """Independent supervisor: raw optical/contact records drive its verdict."""
 def inspect(self,app,raw,meta,now_h):
  try:
   if not np.isfinite(meta['available_h']) or not 0<=now_h-meta['available_h']<=24:raise ValueError('stale calibration')
   ref=decode(raw,meta);q=calibrate(app,ref,meta);q['errors']=[]
  except (ValueError,KeyError,TypeError) as e:q=dict(passed=False,matched=0,unmatched=len(app),errors=[str(e)],kind=meta.get('reference_kind'),physical_certification=False)
  return q

def decode(raw,meta):
 needed={'device','condition','reference_kind','clock_id','clock_calibration_id','tick_us','reference_uncertainty_us'}
 if not needed<=set(meta) or meta['reference_kind'] not in ('engineering_simulation','physical_reference') or not meta['clock_calibration_id'] or not math.isfinite(meta['tick_us']) or meta['tick_us']<=0 or not math.isfinite(meta['reference_uncertainty_us']) or meta['reference_uncertainty_us']<0:raise ValueError('reference metadata')
 rows=[];last=None;epoch=0;armed=set()
 for line in raw.splitlines():
  if not line or line=='trial,edge,ref_ticks,clock_id,dropped':continue
  a=line.split(',')
  if a[0]=='ARMED':
   if len(a)!=2:raise ValueError('armed protocol')
   armed.add(int(a[1]));continue
  if len(a)!=5:raise ValueError('serial protocol/disconnect')
  trial=int(a[0]);edge=a[1];tick=int(a[2]);lost=int(a[4])
  if trial not in armed or edge not in ('S','R') or lost or a[3]!=meta['clock_id'] or not 0<=tick<2**32:raise ValueError('unarmed/unknown/dropped/clock event')
  if last is not None and tick<last:
   if last>3*2**30 and tick<2**30:epoch+=2**32
   else:raise ValueError('clock reset/reordered trace')
  last=tick;rows.append(dict(device=meta['device'],condition=meta['condition'],trial=trial,edge=edge,ref_us=(epoch+tick)*meta['tick_us'],clock_id=meta['clock_id'],reference_kind=meta['reference_kind'],reference_uncertainty_us=meta['reference_uncertainty_us']))
 if not rows:raise ValueError('no reference edges')
 return pd.DataFrame(rows)

class VirtualRig:
 def __init__(self,device=Device(),seed=133110):self.device=device;self.rng=np.random.default_rng(seed)
 def run(self,reaction_ms,n=60,fault='none',condition='idle',now_h=0.):
  if fault not in FAULTS or not 100<=reaction_ms<=2000 or not 2<=n<=1000:raise ValueError('rig trial/configuration')
  d=self.device;lines=['trial,edge,ref_ticks,clock_id,dropped'];apps=[];truth=[];rate=1+d.clock_ppm/1e6;physical_tick=d.tick_us/rate
  for i in range(n):
   command=i*3e6+10000;frame=1e6/d.display_hz;display=math.ceil(command/frame)*frame;optical=display+d.scan_fraction*frame;diode=optical-1000*d.diode_tau_ms*math.log(1-d.threshold_fraction);rt=reaction_ms+self.rng.normal(0,.25);contact=optical+rt*1000;response=contact+1000*d.contact_debounce_ms
   app_onset=display;app_response=math.ceil(response/(1000*d.hid_poll_ms))*1000*d.hid_poll_ms+1000*d.usb_ms
   if fault=='display_delay':diode+=40000;response+=40000;app_response+=40000
   if fault=='hid_delay':app_response+=60000
   if fault=='drift':app_response+=i*1000
   if fault=='actuator_stuck':response=None
   lines.append(f'ARMED,{i}')
   ts=int(round(diode/physical_tick))%2**32
   if fault=='clock_reset' and i>0:ts=100
   lost=1 if fault=='overflow' else 0
   onset=f'{i},S,{ts},ATMEGA328P_TIMER1,{lost}'
   if fault!='missing_optical':lines.append(onset)
   if fault=='duplicate':lines.append(onset)
   if response is not None and fault!='missing_contact':lines.append(f'{i},R,{int(round(response/physical_tick))%2**32},ATMEGA328P_TIMER1,{lost}')
   apps.append(dict(device=d.name,condition=condition,trial=i,app_rt_ms=(app_response-app_onset)/1000,protocol='PVT_CONTACT',display_hz=d.display_hz,browser='virtual_browser',input_path='isolated_contact'))
   truth.append(dict(trial=i,command_us=command,display_us=display,optical_us=optical,diode_us=diode,contact_us=contact,app_response_us=app_response,origin='synthetic_engineering'))
  if fault=='disconnect':lines=['DISCONNECTED']
  meta=dict(device=d.name,condition=condition,reference_kind='engineering_simulation',clock_id='ATMEGA328P_TIMER1',clock_calibration_id='VIRTUAL_TIMEBASE_133',tick_us=physical_tick,reference_uncertainty_us=.5,protocol='PVT_CONTACT',available_h=now_h-25 if fault=='stale_calibration' else now_h)
  app=pd.DataFrame(apps);raw='\n'.join(lines)+'\n';verdict=Supervisor().inspect(app,raw,meta,now_h)
  return dict(app=app,raw=raw,metadata=meta,truth=pd.DataFrame(truth),verdict=verdict)
