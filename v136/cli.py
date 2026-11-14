import argparse,json,os
from .application import Application
from .operations import CommunicationQueue
def main():
 p=argparse.ArgumentParser();p.add_argument('--request',default='{}');p.add_argument('--scenario');p.add_argument('--mars',choices=['earth','mars_earth_light','mars_scheduled_light']);p.add_argument('--without-evidence',action='store_true');p.add_argument('--one-way-s',type=float);p.add_argument('--advance-s',type=float,default=0);args=p.parse_args()
 if args.mars:
  from .mars import DEFINITIONS
  from .common import REPORT,ROOT,sha
  import pandas as pd
  summary=json.loads((REPORT/'mars_summary.json').read_text())
  if summary.get('model_sha256')!=sha(os.environ.get('TWINOVA_MODEL',ROOT/'models_v135/study/twin.joblib')):raise ValueError('Mars results belong to a different model; rerun v136.mars')
  f=pd.read_csv(REPORT/'mars_sol_results.csv');f=f[f.scenario==args.mars].iloc[::4];out=dict(definition=DEFINITIONS[args.mars],series=f[['time_h','S','phase_radians','eye_lux']].to_dict('records'),summary=json.loads((REPORT/'mars_summary.json').read_text()))
 elif args.one_way_s is not None:
  c=CommunicationQueue(args.one_way_s);c.submit({'kind':'decision_summary_only'});out=c.advance(args.advance_s)
 else:
  app=Application();q=json.loads(args.request);out=app.scenario({**q,'scenario':args.scenario})if args.scenario else app.run(q,evidence=not args.without_evidence)
 print(json.dumps(out,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
