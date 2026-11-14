import hashlib,json,math,os,platform,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports_v136'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v)for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,Path):return str(x)
 if isinstance(x,float)and not math.isfinite(x):return None
 return x
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=p.parent,delete=False)as f:f.write(data);temp=f.name
 os.replace(temp,p)
def now():return datetime.now(timezone.utc).isoformat()
def environment():return dict(python=sys.version,platform=platform.platform(),utc=now())
def circular_difference(a,b,period=24.):return (np.asarray(a)-np.asarray(b)+period/2)%period-period/2
