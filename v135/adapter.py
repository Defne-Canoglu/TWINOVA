"""Independent reference evidence; provider selection never uses data origin."""
import json,hashlib
from pathlib import Path
import pandas as pd
from v134.timing import fit_correction

def calibrations(folder,manifest):
 folder=Path(folder);provider=manifest.get('measurement_provider')
 if provider not in ('virtual','native'):raise ValueError('Explicit measurement provider required')
 for n,sha in manifest.get('calibration_files',{}).items():
  p=(folder/n).resolve()
  if folder.resolve() not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('reference file path/hash')
 out={}
 for key,index in manifest.get('calibration_evidence',{}).items():
  paths={k:(folder/v).resolve() for k,v in index.items()}
  if any(folder.resolve() not in p.parents for p in paths.values()):raise ValueError('reference path')
  meta=json.loads(paths['metadata'].read_text())
  if provider=='native' and (meta['reference_kind']!='physical_reference' or any(token in str(meta.get('clock_calibration_id','')).upper() for token in ('VIRTUAL','REPLACE','EXAMPLE'))):raise ValueError('native channel requires actual reference-kind evidence and non-placeholder timebase calibration')
  out[key]=fit_correction(pd.read_csv(paths['app']),paths['raw'].read_text(),meta,meta['available_h'])
 return out
