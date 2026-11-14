"""Native real trace adapter; no synthetic fallback. Capture keeps ARMED/edge stream."""
import argparse,json,math,time
from pathlib import Path
import pandas as pd
from .hardware import decode,Supervisor

def capture(port,path,meta,seconds,arm_delay_us=None):
 if not math.isfinite(seconds) or seconds<=0 or seconds>3600:raise ValueError('duration')
 if meta.get('example_only',False) or meta['reference_kind']!='physical_reference':raise ValueError('explicit physical reference required')
 if arm_delay_us is not None and not 100000<=arm_delay_us<=1000000:raise ValueError('actuator test delay must be 100-1000 ms')
 if Path(path).exists():raise ValueError('refusing to overwrite raw trace')
 import serial
 with serial.Serial(port,115200,timeout=.1) as connection,Path(path).open('x') as out:
  until=time.monotonic()+seconds;trial=0;next_arm=time.monotonic()+2
  while time.monotonic()<until:
   now=time.monotonic()
   if arm_delay_us is not None and now>=next_arm:
    connection.write(f'A,{trial},{arm_delay_us}\n'.encode('ascii'));trial+=1;next_arm=now+3
   line=connection.readline().decode('ascii',errors='strict')
   if line:out.write(line);out.flush()
 Path(path).with_suffix('.metadata.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);v=sub.add_parser('verify');v.add_argument('--app',required=True);v.add_argument('--raw',required=True);v.add_argument('--metadata',required=True);v.add_argument('--now-h',type=float,required=True);v.add_argument('--out',required=True)
 c=sub.add_parser('capture');c.add_argument('--port',required=True);c.add_argument('--out',required=True);c.add_argument('--metadata',required=True);c.add_argument('--seconds',type=float,default=330);c.add_argument('--arm-delay-us',type=int)
 a=p.parse_args();m=json.loads(Path(a.metadata).read_text())
 if a.mode=='capture':capture(a.port,a.out,m,a.seconds,a.arm_delay_us)
 else:
  q=Supervisor().inspect(pd.read_csv(a.app),Path(a.raw).read_text(),m,a.now_h);Path(a.out).write_text(json.dumps(q,indent=2));print(json.dumps(q,indent=2));raise SystemExit(0 if q['passed'] else 1)
