"""Full virtual bench suite including all fault modes and matched native traces."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from .hardware import VirtualRig,Device,FAULTS

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for name,hz in [('VIRTUAL_D1',60),('VIRTUAL_D2',120)]:
  for condition in ['idle','stress']:
   for fault in FAULTS:
    q=VirtualRig(Device(name=name,display_hz=hz,usb_ms=1 if condition=='idle' else 2),seed=133).run(300,n=60,fault=fault,condition=condition,now_h=1);stem=f'{name}_{condition}_{fault}';q['app'].to_csv(out/(stem+'_app.csv'),index=False);(out/(stem+'_serial.txt')).write_text(q['raw']);(out/(stem+'_meta.json')).write_text(json.dumps(q['metadata'],indent=2));q['truth'].to_csv(out/(stem+'_plant_truth.csv'),index=False);passed=q['verdict']['passed'];rows.append(dict(device=name,condition=condition,fault=fault,passed=passed,expected_pass=fault=='none',matches_expectation=passed==(fault=='none'),matched=q['verdict']['matched'],errors=';'.join(q['verdict'].get('errors',[])),origin='synthetic_engineering',scope='discrete event plant; not firmware instruction emulation or physical hardware run'))
 pd.DataFrame(rows).to_csv(out/'bench_results.csv',index=False)
 if not all(r['matches_expectation'] for r in rows):raise ValueError('unexpected hardware fault response')
 return rows
if __name__=='__main__':print(len(run('reports_v133/bench')),'bench scenarios verified')
