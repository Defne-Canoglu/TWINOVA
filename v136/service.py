import argparse,json,os,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
def strict_json(raw):
 def pairs(items):
  out={}
  for key,value in items:
   if key in out:raise ValueError('Duplicate JSON field')
   out[key]=value
  return out
 def invalid(value):raise ValueError('Nonfinite JSON number')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

from .application import Application
from .common import ROOT,clean
from .scenarios import CATALOG
from .operations import CommunicationQueue

_application=None;_lock=threading.RLock()
def app():
 global _application
 with _lock:
  if _application is None:_application=Application(os.environ.get('TWINOVA_DATA'),os.environ.get('TWINOVA_MODEL'))
  return _application
def query(path):
 q=parse_qs(path.query,keep_blank_values=True)
 if any(len(v)!=1 for v in q.values()):raise ValueError('Duplicate parameter')
 return {k:v[0]for k,v in q.items()}
class Handler(BaseHTTPRequestHandler):
 def setup(self):
  super().setup();self.connection.settimeout(15)

 def send(self,value,status=200):
  data=json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
  self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
 def page(self,path):
  data=path.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8' if path.suffix=='.html' else 'text/plain; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

 def do_GET(self):
  path=urlparse(self.path)
  try:
   if path.path=='/':return self.page(ROOT/'ui/v136.html')
   if path.path=='/api/v135-config':
    a=app();return self.send(dict(people=a.b['splits']['test'],tasks=sorted(a.study.observations.task.unique()),faults=['none','device_delay','clock_drift','stale_calibration','delayed_input'],scenarios=CATALOG,extension_version='13.6.0'))
   if path.path=='/api/v135':return self.send(app().run(query(path),os.environ.get('TWINOVA_LEDGER')))
   if path.path=='/api/v135-prefix':return self.send(app().prefix_decision(query(path)))
   if path.path=='/api/v136/scenario':return self.send(app().scenario(query(path)))
   if path.path=='/api/v136/health':return self.send(app().health())
   if path.path=='/api/v136/hardware':return self.send(app().hardware())
   if path.path=='/api/v136/comms':
    q=query(path)
    if set(q)-{'one_way_s','advance_s'}:raise ValueError('Communication request fields')
    c=CommunicationQueue(float(q.get('one_way_s',1200)));c.submit({'kind':'decision_summary_only'});return self.send(c.advance(float(q.get('advance_s',0))))
   if path.path=='/api/v136/research':
    p=ROOT/'reports_v136/research_summary.json';return self.send(json.loads(p.read_text())if p.exists()else dict(status='not_run'))
   if path.path=='/api/v136/mars':
    from .mars import DEFINITIONS,simulate
    import pandas as pd
    q=query(path)
    if set(q)-{'scenario'}:raise ValueError('Mars request fields')
    name=q.get('scenario','earth')
    if name not in DEFINITIONS:raise ValueError('Mars scenario')
    p=ROOT/'reports_v136/mars_sol_results.csv'
    f=pd.read_csv(p)if p.exists()else simulate(app().b,name)
    f=f[f.scenario==name].iloc[::4]
    summary_path=ROOT/'reports_v136/mars_summary.json';summary=json.loads(summary_path.read_text())if summary_path.exists()else {'scope':'Model-only simulation'}
    if p.exists() and summary.get('model_sha256')!=app().hash:raise ValueError('Mars results belong to a different model; rerun python -m v136.mars with the active TWINOVA_MODEL')
    return self.send(clean(dict(definition=DEFINITIONS[name],series=f[['time_h','S','phase_radians','eye_lux']].to_dict('records'),summary=summary)))
   if path.path=='/report136':
    self.send_response(302);self.send_header('Location','/artifacts136/final_results.html');self.send_header('Content-Length','0');self.end_headers();return
   if path.path.startswith('/docs_v136/'):
    p=(ROOT/'docs'/path.path.removeprefix('/docs_v136/')).resolve()
    if (ROOT/'docs').resolve()not in p.parents or p.suffix not in ['.md','.html']:raise ValueError('Documentation path')
    return self.page(p)
   if path.path.startswith('/artifacts136/'):
    p=(ROOT/'reports_v136'/path.path.removeprefix('/artifacts136/')).resolve()
    if (ROOT/'reports_v136').resolve()not in p.parents:raise ValueError('Artifact path')
    if p.suffix not in {'.html','.png','.svg','.json','.csv','.md','.txt'}:raise ValueError('Artifact type')
    data=p.read_bytes();self.send_response(200);self.send_header('Content-Type',{'.png':'image/png','.svg':'image/svg+xml','.html':'text/html; charset=utf-8','.json':'application/json','.csv':'text/csv; charset=utf-8','.md':'text/plain; charset=utf-8','.txt':'text/plain; charset=utf-8'}[p.suffix]);self.send_header('Content-Length',str(len(data)));self.end_headers();return self.wfile.write(data)
  except (ValueError,KeyError,TypeError,OSError)as e:return self.send(dict(error=str(e),result=None),400)
  return self.send({"error":"Not found","result":None},404)
 def do_POST(self):
  path=urlparse(self.path)
  if path.path not in ['/api/v135','/api/v136/scenario']:return self.send({"error":"Not found","result":None},404)
  try:
   n=int(self.headers.get('Content-Length',0))
   if not 0<n<=4000:raise ValueError('Request size')
   q=strict_json(self.rfile.read(n));return self.send(app().scenario(q)if path.path.endswith('scenario')else app().run(q,os.environ.get('TWINOVA_LEDGER')))
  except (ValueError,KeyError,TypeError,OSError)as e:return self.send(dict(error=str(e),result=None),400)
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);args=p.parse_args();app();print('TWINOVA 13.6 http://127.0.0.1:'+str(args.port),flush=True);ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
if __name__=='__main__':main()
