"""Real serial session for the browser photodiode/contact bench; explicit configuration only."""
import threading,time,json,math
from pathlib import Path
import pandas as pd
from .hardware import Supervisor
class HardwareSession:
 def __init__(self,port,metadata,output,serial_factory=None):
  if metadata.get('reference_kind')!='physical_reference' or metadata.get('clock')!='UTC_epoch_hours':raise ValueError('physical reference metadata required')
  if metadata.get('example_only',False) or str(metadata.get('clock_calibration_id','')).startswith('REPLACE') or not math.isfinite(metadata.get('available_h',float('nan'))) or not 0<=time.time()/3600-metadata['available_h']<=24:raise ValueError('actual current clock calibration required')
  if not metadata.get('clock_calibration_id') or not 0<metadata.get('tick_us',0)<10 or not math.isfinite(metadata.get('reference_uncertainty_us',float('nan'))):raise ValueError('clock calibration metadata required')
  self.meta=metadata;self.path=Path(output)
  if self.path.exists():raise ValueError('new output directory required')
  if serial_factory is None:
   import serial
   serial_factory=serial.Serial
  self.connection=serial_factory(port,115200,timeout=.1);self.path.mkdir(parents=True);self.lines=[];self.apps=[];self.lock=threading.RLock();self.stop=threading.Event();self.pending=None;self.armed=set();self.status='connected';self.thread=threading.Thread(target=self.reader,daemon=True);self.thread.start()
 def reader(self):
  try:
   while not self.stop.is_set():
    raw=self.connection.readline()
    if not raw:continue
    line=raw.decode('ascii',errors='strict').strip()
    with self.lock:
     self.lines.append(line)
     with (self.path/'reference_serial.txt').open('a') as f:f.write(line+'\n')
     if line.startswith('ARMED,'):self.armed.add(int(line.split(',')[1]))
  except Exception as e:self.status='disconnected: '+type(e).__name__;self.stop.set()
 def arm(self,trial,delay_us):
  if isinstance(trial,bool) or not isinstance(trial,int) or trial<0 or isinstance(delay_us,bool) or not isinstance(delay_us,int) or not 100000<=delay_us<=1000000:raise ValueError('trial/delay')
  with self.lock:
   if self.stop.is_set() or self.pending is not None or trial in self.armed:raise ValueError('serial busy/disconnected/duplicate trial')
   self.pending=trial;self.connection.write(f'A,{trial},{delay_us}\n'.encode('ascii'))
  deadline=time.monotonic()+2
  while time.monotonic()<deadline:
   with self.lock:
    if trial in self.armed:return dict(trial=trial,armed=True)
   if self.stop.wait(.01):break
  self.pending=None;raise ValueError('ARMED acknowledgement timeout')
 def record(self,trial,rt_ms,display_hz):
  if not all(math.isfinite(v) for v in (rt_ms,display_hz)) or not 0<rt_ms<=10000 or not 1<=display_hz<=1000:raise ValueError('app timing range')
  with self.lock:
   if self.pending!=trial or any(r['trial']==trial for r in self.apps):raise ValueError('not pending or duplicate trial')
   self.apps.append(dict(device=self.meta['device'],condition=self.meta['condition'],trial=trial,app_rt_ms=rt_ms,protocol=self.meta['protocol'],display_hz=display_hz,browser='user_browser',input_path='isolated_contact'));self.pending=None;pd.DataFrame(self.apps).to_csv(self.path/'app.csv',index=False);(self.path/'metadata.json').write_text(json.dumps(self.meta,indent=2))
   return dict(recorded=True,trial=trial)
 def result(self,now_h):
  with self.lock:
   if not self.apps:return dict(passed=False,errors=['no completed app trials'],physical_certification=False)
   return Supervisor().inspect(pd.DataFrame(self.apps),'\n'.join(self.lines),self.meta,now_h)
 def close(self):
  self.stop.set();self.thread.join(timeout=1);self.connection.close()
