from dataclasses import dataclass,asdict
import hashlib,json,math
UNITS={'pvt_median':'ms','pvt_lapse_rate':'fraction','sleep_duration':'h','photopic_light':'lux','melanopic_light':'melanopic_EDI_lux','kss':'score','heart_rate':'bpm','rr_interval':'ms','feature':'model_native'}
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
@dataclass(frozen=True)
class Observation:
 source:str;version:str;study:str;person:str;session:str;protocol:str;device:str;variable:str;unit:str;start_h:float;end_h:float;available_h:float;value:float|None;quality:str;missing_reason:str|None;origin:str;root_id:str;file_sha256:str;transformations:tuple;light_position:str|None=None
 def validate(self,decision_h):
  if not all(isinstance(getattr(self,k),str) and getattr(self,k).strip() for k in ('source','version','study','person','session','protocol','device','root_id')):raise ValueError('missing provenance')
  if self.origin not in ('real_archive','real_archive_injected','synthetic_engineering'):raise ValueError('origin')
  if len(self.file_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.file_sha256) or not self.transformations:raise ValueError('lineage')
  if UNITS.get(self.variable)!=self.unit:raise ValueError('unit')
  if not all(math.isfinite(v) for v in (self.start_h,self.end_h,self.available_h,decision_h)) or not self.start_h<=self.end_h<=self.available_h<=decision_h:raise ValueError('availability/time')
  if 'light' in self.variable and self.light_position not in ('wrist','eye','room_reference'):raise ValueError('light position')
  if self.value is None:
   if self.quality!='missing' or not self.missing_reason:raise ValueError('missing reason')
  elif not math.isfinite(self.value) or self.quality not in ('valid','flagged') or self.missing_reason:raise ValueError('value/quality')
  if self.value is not None:
   if self.variable=='pvt_lapse_rate' and not 0<=self.value<=1:raise ValueError('fraction')
   if self.variable=='kss' and not 1<=self.value<=10:raise ValueError('KSS')
   if self.variable not in ('feature','kss','pvt_lapse_rate') and self.value<0:raise ValueError('negative measurement')
  return self
 def as_dict(self):return asdict(self)
def compatible(a,b,decision_h,independent=False):
 a.validate(decision_h);b.validate(max(decision_h,b.available_h))
 if any(getattr(a,k)!=getattr(b,k) for k in ('source','version','study','person','protocol','device','unit','origin','variable')):raise ValueError('incompatible identity/source')
 if independent and (a.root_id==b.root_id or a.session==b.session or not a.available_h<=decision_h<=b.start_h or not 0<=b.start_h-a.end_h<=48):raise ValueError('not independent')
 return True
def fuse(observations,decision_h):
 if not observations:raise ValueError('empty input')
 first=observations[0]
 for o in observations:
  o.validate(decision_h)
  if any(getattr(o,k)!=getattr(first,k) for k in ('source','version','study','person','protocol','device','origin')):raise ValueError('cross-source/person fusion')
 return True
def validate_split(rows):
 roots={};people={}
 for r in rows:
  if r['split'] not in ('train','probability','policy','calibration','test'):raise ValueError('split')
  for key,cache in [((r['source'],r['root_id']),roots),((r['source'],r['person']),people)]:
   if key in cache and cache[key]!=r['split']:raise ValueError('derivative/person leakage')
   cache[key]=r['split']
 return True
