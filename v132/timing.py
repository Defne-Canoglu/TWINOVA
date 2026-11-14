import numpy as np,pandas as pd
APP={'device','condition','trial','app_rt_ms','protocol','display_hz','browser','input_path'};REF={'device','condition','trial','edge','ref_us','clock_id','reference_kind','reference_uncertainty_us'}
def pair(app,reference):
 if not APP<=set(app) or not REF<=set(reference):raise ValueError('timing schema')
 if app[list(APP)].isna().any().any() or reference[list(REF)].isna().any().any():raise ValueError('missing timing fields')
 keys=['device','condition','trial']
 if app.duplicated(keys).any() or reference.duplicated(keys+['edge']).any():raise ValueError('duplicate event')
 if not reference.edge.isin(['S','R']).all() or not reference.reference_kind.isin(['physical_reference','engineering_simulation']).all():raise ValueError('edge/origin')
 if reference.clock_id.astype(str).str.strip().eq('').any():raise ValueError('independent clock required')
 if not np.isfinite(app.app_rt_ms).all() or not (app.app_rt_ms>0).all() or not np.isfinite(reference[['ref_us','reference_uncertainty_us']]).all().all() or (reference.reference_uncertainty_us<0).any():raise ValueError('timing domain')
 rows=[];rejected=[]
 for key,g in reference.groupby(keys):
  if len(g)!=2 or set(g.edge)!={'S','R'}:rejected.append(dict(zip(keys,key),reason='missing_edge'));continue
  if g.clock_id.nunique()!=1 or g.reference_kind.nunique()!=1:raise ValueError('clock/origin mismatch')
  s=g[g.edge=='S'].iloc[0];r=g[g.edge=='R'].iloc[0]
  if r.ref_us<s.ref_us:raise ValueError('unresolved clock wrap')
  rows.append(dict(zip(keys,key),reference_ms=(r.ref_us-s.ref_us)/1000,reference_kind=s.reference_kind,clock_id=s.clock_id,reference_uncertainty_ms=float(np.sqrt(np.sum(g.reference_uncertainty_us**2)))/1000))
 ref=pd.DataFrame(rows,columns=keys+['reference_ms','reference_kind','clock_id','reference_uncertainty_ms']);d=app.merge(ref,on=keys,how='outer',indicator=True,validate='one_to_one')
 for r in d[d['_merge']!='both'].to_dict('records'):rejected.append({**{k:r[k] for k in keys},'reason':'unmatched_app_or_reference'})
 d=d[d['_merge']=='both'].drop(columns='_merge');d['signed_error_ms']=d.app_rt_ms-d.reference_ms;summary=d.groupby(['device','condition','reference_kind']).signed_error_ms.agg(n='count',bias_ms='mean',mae_ms=lambda x:x.abs().mean(),sd_ms='std',p95_abs_ms=lambda x:x.abs().quantile(.95),max_abs_ms=lambda x:x.abs().max()).reset_index();rejected=pd.DataFrame(rejected,columns=keys+['reason'])
 if len(rejected):rejected=rejected.groupby(keys,as_index=False).reason.agg(lambda x:';'.join(sorted(set(x))))
 return d,summary,rejected
def calibrate(app,reference,meta):
 if not {'device','protocol','clock_id','clock_calibration_id','available_h','reference_kind'}<=set(meta) or not meta['clock_calibration_id']:raise ValueError('calibration metadata')
 if set(app.device)!={meta['device']} or set(app.protocol)!={meta['protocol']} or set(reference.clock_id)!={meta['clock_id']} or set(reference.reference_kind)!={meta['reference_kind']}:raise ValueError('calibration identity/origin')
 good,summary,rejected=pair(app,reference)
 if not len(good):raise ValueError('no paired events')
 return dict(kind=meta['reference_kind'],device=meta['device'],protocol=meta['protocol'],available_h=meta['available_h'],calibration_id=meta['clock_calibration_id'],passed=not len(rejected) and bool((summary.bias_ms.abs()<=10).all() and (summary.p95_abs_ms<=20).all()),matched=len(good),unmatched=len(rejected),limits='exploratory engineering; not clinical',physical_certification=False,summary=summary.to_dict('records'))
