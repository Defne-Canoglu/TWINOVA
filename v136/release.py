"""Maintained GitHub entry point. Research models never replace the default model."""
import argparse,unittest,json,hashlib,sys
from .common import ROOT,REPORT

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 for name in ['demo','verify','experiments','report','integrity']:g.add_argument('--'+name,action='store_true')
 a=p.parse_args()
 if a.demo:
  from .service import main as serve
  sys.argv=[sys.argv[0]];serve();return 0
 if a.verify:return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(['v135.test_core','v136.test_extension','v136.test_release'])).wasSuccessful()else 1
 if a.integrity:
  expected=json.loads((REPORT/'baseline_file_hashes.json').read_text());bad=[name for name,digest in expected.items()if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
  print(json.dumps({'files':len(expected),'mismatches':bad}));return bool(bad)
 if a.experiments:
  from .research import evaluate
  from .stability import run_stability
  from .mars import run_mars
  from .phase import phase_checks
  from .supplement import run_checks,run_unseen
  for fn in [evaluate,run_stability,run_mars,phase_checks,run_checks,run_unseen]:fn()
 if a.report or a.experiments:
  from .report import report
  report()
 return 0
if __name__=='__main__':raise SystemExit(main())
