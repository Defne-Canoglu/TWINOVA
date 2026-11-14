"""Robust pre-decision calibration; no current target enters the correction."""
import numpy as np,pandas as pd
from v133.hardware import decode,VirtualRig,Device,Supervisor

def fit_correction(app,raw,metadata,now):
 verdict=Supervisor().inspect(app,raw,metadata,now)
 if not verdict['passed']:raise ValueError('Rejected raw calibration; correction cannot rescue it')
 ref=decode(raw,metadata);s=ref[ref.edge=='S'].set_index('trial').ref_us;r=ref[ref.edge=='R'].set_index('trial').ref_us;truth=(r-s)/1000;a=app.set_index('trial');ids=a.index.intersection(truth.index);errors=a.loc[ids,'app_rt_ms'].to_numpy()-truth.loc[ids].to_numpy();x=np.linspace(-1,1,len(errors));X=np.column_stack([np.ones(len(x)),x]);w=np.ones(len(x))
 for _ in range(8):
  theta=np.linalg.lstsq(X*w[:,None],errors*w,rcond=None)[0];res=errors-X@theta;scale=max(.05,1.4826*np.median(abs(res-np.median(res))));w=np.minimum(1.,1.5*scale/np.maximum(abs(res),1e-8))
 # Extrapolate no trial trend into future sessions: trend contributes uncertainty.
 return dict(bias_ms=float(theta[0]),trend_span_ms=float(2*theta[1]),residual_sd_ms=float(max(.05,np.std(res,ddof=2))),bias_se_ms=float(max(.05,np.std(res,ddof=2))/np.sqrt(len(res))),reference_sd_ms=float(metadata['reference_uncertainty_us']/1000),available_h=float(metadata['available_h']),max_age_h=24.,n=len(res),verdict=verdict)

def correct(value,cal,now):
 age=now-cal['available_h']
 if not 0<=age<=cal['max_age_h']:raise ValueError('Stale or future calibration')
 # This instrument component is recorded separately; intervals already calibrated
 # on corrected inputs must not have it added a second time.
 sd=np.sqrt(cal['residual_sd_ms']**2+cal['bias_se_ms']**2+cal['reference_sd_ms']**2+(age/24*cal['trend_span_ms'])**2)
 return float(value-cal['bias_ms']),float(sd)

def benchmark(out):
 rows=[];faultrows=[]
 for hz in [60,120]:
  for load in ['idle','stress']:
   device=Device(name=f'D{hz}',display_hz=hz,usb_ms=1 if load=='idle' else 2,hid_poll_ms=1 if load=='idle' else 2)
   rig=VirtualRig(device,134+hz);cal=rig.run(300,n=40,condition=load,now_h=0);fit=fit_correction(cal['app'],cal['raw'],cal['metadata'],0);previous=float(cal['verdict']['summary'][0]['bias_ms'])
   for reaction in [220,350,500]:
    z=rig.run(reaction,n=40,condition=load,now_h=2);ref=decode(z['raw'],z['metadata']);s=ref[ref.edge=='S'].set_index('trial').ref_us;r=ref[ref.edge=='R'].set_index('trial').ref_us;target=(r-s).to_numpy()/1000;app=z['app'].app_rt_ms.to_numpy()
    for name,p in [('uncorrected',app),('previous',app-previous),('robust',np.array([correct(x,fit,2)[0] for x in app]))]:rows.append(dict(device=device.name,load=load,reaction_ms=reaction,method=name,MAE_ms=float(np.mean(abs(p-target))),p95_ms=float(np.quantile(abs(p-target),.95))))
   for fault in ['none','hid_delay','drift','missing_optical','clock_reset','stale_calibration']:
    z=rig.run(300,n=30,fault=fault,condition=load,now_h=2);faultrows.append(dict(device=device.name,load=load,fault=fault,accepted=z['verdict']['passed'],expected=fault=='none'))
 pd.DataFrame(rows).to_csv(out/'device_correction.csv',index=False);pd.DataFrame(faultrows).to_csv(out/'device_acceptance.csv',index=False)
