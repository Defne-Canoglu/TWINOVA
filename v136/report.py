"""Render the recorded research results in English without rerunning experiments."""
import html,json
from .common import REPORT

def report():
 import pandas as pd
 data=pd.read_csv(REPORT/'ablation_results.csv')
 text='<h1>TWINOVA research results</h1><p>Retrospective comparison on a synthetic engineering cohort. This is not clinical or astronaut validation. The fixed long-acquisition comparison is separate from the frozen default short-acquisition model.</p>'
 cols=[c for c in ['model','stage','people','tasks','MAE_ms','sensitivity','false_positive_rate','task_coverage','person_all_tasks_coverage','width_mean_ms'] if c in data]
 text+=data[cols].to_html(index=False)
 for name in ['personal_development_selection.json','phase_consistency_results.json','mars_summary.json']:
  text+='<h2>'+html.escape(name)+'</h2><pre>'+html.escape(json.dumps(json.loads((REPORT/name).read_text()),indent=2))+'</pre>'
 text+='<p>The wake-rate candidate was not promoted. Numerical phase consistency and biological validation are separate. Hardware results refer to the virtual bench.</p>'
 (REPORT/'final_results.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TWINOVA research</title><style>body{font:16px/1.6 system-ui;max-width:1200px;padding:24px;margin:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'+text,encoding='utf-8')
if __name__=='__main__':report()
