"""Forger99 dynamics + sleep homeostat; published population priors, not personal validation.
Equations: https://arcascope.github.io/circadian/api/models.html#forger99
State [x,xc,n,S], time hours, input photopic lux at EYE only.
"""
from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class Parameters:
 tau_h:float=24.2;mu:float=.23;G:float=33.75;alpha0:float=.05;beta:float=.0075;p:float=.5;I0:float=9500.;k:float=.55
 sleep_decay_h:float=4.2;wake_rise_h:float=18.2
 def __post_init__(self):
  if not all(np.isfinite(v) and v>0 for v in asdict(self).values()) or not 18<=self.tau_h<=30:raise ValueError('physiological parameter domain')
def rhs(y,eye_lux,asleep,p=Parameters()):
 y=np.asarray(y,float);x,xc,n,s=np.moveaxis(y,-1,0);alpha=p.alpha0*(np.asarray(eye_lux)/p.I0)**p.p;B=p.G*(1-n)*alpha*(1-.4*x)*(1-.4*xc)
 return np.stack([np.pi/12*(xc+B),np.pi/12*(p.mu*(xc-4*xc**3/3)-x*((24/(.99669*p.tau_h))**2+p.k*B)),60*(alpha*(1-n)-p.beta*n),np.where(asleep,-s/p.sleep_decay_h,(1-s)/p.wake_rise_h)],axis=-1)
def advance(state,eye_lux,asleep,hours,p=Parameters(),max_step_h=.025):
 y=np.asarray(state,float).copy();lux=np.asarray(eye_lux,float)
 if y.shape[-1]!=4 or not np.isfinite(y).all() or not np.isfinite(lux).all() or (lux<0).any() or not np.isfinite(hours) or hours<0 or not 0<max_step_h<=.05:raise ValueError('state/input/time domain')
 if ((y[...,2:]<0)|(y[...,2:]>1)).any() or np.max(abs(y[...,:2]))>3:raise ValueError('unphysical state')
 steps=max(1,int(np.ceil(hours/max_step_h)));dt=hours/steps
 for _ in range(steps):
  a=rhs(y,lux,asleep,p);b=rhs(y+dt*a/2,lux,asleep,p);c=rhs(y+dt*b/2,lux,asleep,p);d=rhs(y+dt*c,lux,asleep,p);y+=dt*(a+2*b+2*c+d)/6
 if not np.isfinite(y).all() or ((y[...,2:]<-.000001)|(y[...,2:]>1.000001)).any():raise ValueError('integration failure')
 return y

def exposure(previous,co2_ppm,dose_ugy_h,gravity_g,hours):
 if not all(np.isfinite(v) for v in [co2_ppm,dose_ugy_h,gravity_g,hours]) or min(co2_ppm,dose_ugy_h,gravity_g,hours)<0:raise ValueError('environment domain')
 return dict(co2_excess_ppm_h=previous.get('co2_excess_ppm_h',0)+max(0,co2_ppm-1000)*hours,radiation_ugy=previous.get('radiation_ugy',0)+dose_ugy_h*hours,gravity_deviation_g_h=previous.get('gravity_deviation_g_h',0)+abs(1-gravity_g)*hours)
