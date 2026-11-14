import sqlite3,json,math
from pathlib import Path
from .contracts import canonical,digest
class Ledger:
 def __init__(self,path):
  self.path=str(path);Path(path).parent.mkdir(parents=True,exist_ok=True)
  with self.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,episode TEXT,stage TEXT,available REAL,payload TEXT,sha TEXT)')
 def connect(self):return sqlite3.connect(self.path,timeout=30)
 def put(self,key,episode,stage,at,payload):
  if not key or not episode or not math.isfinite(at) or stage not in ('observation','initial','decision','revision','outcome'):raise ValueError('event identity/stage/clock')
  row=(episode,stage,at,canonical(payload),digest(payload))
  with self.connect() as c:
   c.execute('BEGIN IMMEDIATE');old=c.execute('SELECT episode,stage,available,payload,sha FROM events WHERE id=?',(key,)).fetchone()
   if old and old!=row:raise ValueError('immutable event collision')
   if not old:c.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(key,*row))
 def read(self,at=None):
  with self.connect() as c:rows=c.execute('SELECT * FROM events ORDER BY available,id').fetchall()
  out=[]
  for key,episode,stage,available,raw,sha in rows:
   p=json.loads(raw)
   if digest(p)!=sha:raise ValueError('corrupt ledger')
   if at is None or available<=at:out.append(dict(id=key,episode=episode,stage=stage,available_h=available,payload=p))
  return out
