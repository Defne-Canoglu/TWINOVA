"""Fresh virtual-suite evidence, kept separate from any physical certification."""
from .common import ROOT,REPORT,write,sha,now

def run_bench():
 from v133.bench import run
 rows=run(REPORT/'hardware_bench')
 write(REPORT/'hardware_bench/execution.json',dict(completed_at=now(),version='13.6.0',report_sha256=sha(REPORT/'hardware_bench/bench_results.csv'),source_hashes={str(p):sha(ROOT/p)for p in ['v133/bench.py','v133/hardware.py']},freshness_days=30,expiry_scope='Engineering artifact-review policy, not current device calibration',physical_run=False))
 from .operations import hardware_summary
 out=hardware_summary();write(REPORT/'hardware_validation_summary.json',out);return out
if __name__=='__main__':run_bench()
