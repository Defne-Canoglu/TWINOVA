import math
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from v133.data import BASE,AFTER

class ResidualModel:
 def __init__(self,alpha=10):self.base=make_pipeline(StandardScaler(),Ridge(alpha=alpha));self.residual=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=20,learning_rate=.05,random_state=134)
 def fit(self,x,y,sample_weight=None):
  self.base.fit(x,y,ridge__sample_weight=sample_weight);self.residual.fit(x,np.asarray(y)-self.base.predict(x),sample_weight=sample_weight);return self
 def predict(self,x):return self.base.predict(x)+self.residual.predict(x)

def candidate(name):
 if name.startswith('ridge'):return make_pipeline(StandardScaler(),Ridge(alpha=float(name.split('_')[1])))
 if name=='nonlinear':return HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=20,learning_rate=.05,random_state=134)
 return ResidualModel()

def fit(name,d,cols,y=None):
 m=candidate(name);w=1/d.groupby('person').person.transform('size');w=w/w.mean();kwargs={'ridge__sample_weight':w} if name.startswith('ridge') else {'sample_weight':w};m.fit(d[cols],d.target_rt if y is None else y,**kwargs);return m

def personal(d,model,now,config):
 past=d[(d.target_available_h<=now)&(d.quality=='valid')&(d.target_available_h>=now-72)].sort_values('target_available_h')
 used=[dict(task=str(r.task),available_h=float(r.target_available_h)) for r in past.itertuples()]
 if len(past)<2:return dict(mean_ms=0.,variance_ms2=100.,accepted=len(past),outcomes=used,coefficients=[0.,0.],identified_slope=False)
 X=np.column_stack([np.ones(len(past)),past.S.to_numpy()-.5]);res=past.target_rt.to_numpy()-model.predict(past[AFTER]);w=np.exp(-(now-past.target_available_h.to_numpy())/config['decay_h']);precision=np.diag([1/config['intercept_var'],1/config['slope_var']]);cov=np.linalg.inv(precision+X.T@(w[:,None]*X)/config['noise_var']);theta=cov@(X.T@(w*res))/config['noise_var']
 identifiable=bool(len(past)>=6 and past.S.std()>.04)
 if not identifiable:theta[1]=0.
 return dict(mean_ms=float(theta[0]),variance_ms2=float(cov[0,0]),accepted=len(past),outcomes=used,coefficients=theta.tolist(),identified_slope=identifiable)

def correction(post,S):return post['coefficients'][0]+post['coefficients'][1]*(float(S)-.5)

def personal_predictions(d,model,config):
 pred=model.predict(d[AFTER]);out=np.array(pred)
 for _,g in d.groupby('person',sort=False):
  for idx,r in g.iterrows():
   past=g[(g.protocol==r.protocol)&(g.task!=r.task)];p=personal(past,model,r.issued_h,config);out[d.index.get_loc(idx)]+=correction(p,r.S)
 return out

def conformal(scores,nominal=.9):
 a=np.sort(np.asarray(scores));k=math.ceil((len(a)+1)*nominal)
 return float(a[k-1]) if k<=len(a) else math.inf

def scales(model,d,cols):return np.maximum(3.,model.predict(d[cols]))

def interval_radius(bundle,key,d):
 item=bundle.get('interval_models',{}).get(key)
 if item is None:return np.full(len(d),bundle['radii'][key])
 return item['q']*(scales(item['scale'],d,item['cols']) if item['method']=='normalized' else np.ones(len(d)))
