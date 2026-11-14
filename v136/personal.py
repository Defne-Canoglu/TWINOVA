"""A single wake-time-constant offset, fitted only from available labels.

The grid contains exact exponential homeostat trajectories from the person's
available sleep history; no tau/phase parameter is estimated from PVT alone.
"""
import numpy as np,pandas as pd
from scipy.special import logsumexp
from .features import RATE_GRID,RATE_COLS

class WakeRateAdapter:
 def adjust(self,q,past,spec,estimate):
  recent=[z for z in past if z.get('stage')=='after'and 0<=q['now_h']-z['available_h']<=72 and z.get('dynamic_context')]
  default=dict(intercept=0.,slope=0.,accepted=len(recent),identified_slope=False,identified_wake_rate=False,wake_rate_offset=0.,grid_offsets=list(RATE_GRID),posterior_weights=None,reason='insufficient_history')
  if len(recent)<6 or np.std([z['S']for z in recent])<=.04:return estimate,default
  cols=spec['cols'];preds=[]
  for z in recent:
   batch=[]
   for c in RATE_COLS:
    row=dict(z['model_features']);row['S']=z['dynamic_context'][c];batch.append(row)
   preds.append(spec['model'].predict(pd.DataFrame(batch)[cols])[:,0])
  preds=np.asarray(preds);y=np.array([z['rt']for z in recent]);w=np.exp(-np.array([q['now_h']-z['available_h']for z in recent])/48.)
  contrasts=preds[:,[-1]]-preds[:,[0]];identified=bool(np.std(contrasts)>.05)
  if not identified:default['reason']='wake_rate_confounded_with_intercept';return estimate,default
  intercepts=(w[:,None]*(y[:,None]-preds)).sum(axis=0)/(w.sum()+144/9)
  loss=(w[:,None]*(y[:,None]-preds-intercepts)**2).sum(axis=0)/144+intercepts**2/9
  logp=-.5*loss-.5*(np.array(RATE_GRID)/.05)**2;weights=np.exp(logp-logsumexp(logp));index=int(np.argmax(weights));offset=float(RATE_GRID[index]);batch=[]
  for c in RATE_COLS:
   row={k:q[k]for k in cols};row['S']=q[c];batch.append(row)
  new=spec['model'].predict(pd.DataFrame(batch)[cols]);estimate=dict(estimate);estimate['rt']=float(new[index,0]+intercepts[index])
  return estimate,dict(default,intercept=float(intercepts[index]),identified_wake_rate=True,wake_rate_offset=offset,posterior_weights=weights.tolist(),posterior_sd=float(np.sqrt(np.sum(weights*(np.array(RATE_GRID)-np.sum(weights*np.array(RATE_GRID)))**2))),reason='past_label_grid_update',parameter='wake_rise_h = 18.2*(1+offset)',identified_slope=False)
