"""Conditional state/parameter sensitivity, distinct from held-out calibrated residuals."""
import numpy as np
from .data import BASE,AFTER

def propagate(row,model,features,seed,draws=30):
 rng=np.random.default_rng(seed);d=row.loc[row.index.repeat(draws)].reset_index(drop=True).copy();angle=np.arctan2(-d.xc.to_numpy(),d.x.to_numpy());amp=np.hypot(d.x.to_numpy(),d.xc.to_numpy());phase_sd=float(row.iloc[0].get('phase_sd_rad',.2));shift=rng.normal(0,phase_sd,draws);d['x']=amp*np.cos(angle+shift);d['xc']=-amp*np.sin(angle+shift);d['S']=np.clip(d.S+rng.normal(0,.03,draws),0,1);p=model.predict(d[features]);return dict(draws=draws,prediction_sd_ms=float(np.std(p,ddof=1)),phase_sd_rad_assumed=phase_sd,homeostat_sd_assumed=.03,scope='conditional Monte Carlo sensitivity; not calibrated state probability; no biological phase measurement implied')
