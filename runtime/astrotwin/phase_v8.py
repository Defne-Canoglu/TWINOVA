"""Explicit assay protocol, not a substitute for biological validation."""
import numpy as np
def melatonin_crossing(times_h,concentration,*,threshold,unit,max_gap_h=1.5,minimum_followup=2):
 t=np.asarray(times_h,float);y=np.asarray(concentration,float)
 if unit!='pg/mL':raise ValueError('Documented assay units required')
 if t.ndim!=1 or y.shape!=t.shape or len(t)<3 or not np.isfinite(t).all() or not np.isfinite(y).all() or (np.diff(t)<=0).any() or (y<0).any() or not np.isfinite([threshold,max_gap_h]).all() or min(threshold,max_gap_h)<=0 or minimum_followup<1:raise ValueError('Invalid protocol/sequence')
 for i in range(1,len(t)):
  if y[i-1]<threshold<=y[i]:
   end=i+minimum_followup
   if end>len(t) or np.any(np.diff(t[i-1:end])>max_gap_h) or not np.all(y[i:end]>=threshold):continue
   onset=t[i-1]+(threshold-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]);return {'status':'THRESHOLD_CROSSING','time_h':float(onset),'bracket_h':[float(t[i-1]),float(t[i])],'interpretation':'assay threshold extraction, not oscillator validation'}
 return {'status':'UNIDENTIFIABLE_WITH_RELEASED_SAMPLES'}
def dlmo_crossing(times_h,concentration,*,threshold,unit,dim_light_confirmed,max_gap_h=1.5,minimum_followup=2):
 if not dim_light_confirmed:raise ValueError('Documented dim-light protocol required for DLMO interpretation')
 return melatonin_crossing(times_h,concentration,threshold=threshold,unit=unit,max_gap_h=max_gap_h,minimum_followup=minimum_followup)
def circular_error_h(predicted,reference):
 if not np.isfinite([predicted,reference]).all():raise ValueError('Finite phase required')
 return float((predicted-reference+12)%24-12)
