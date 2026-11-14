from pathlib import Path
import subprocess,argparse,json
R=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--toolchain',type=Path,required=True);a=p.parse_args();b=a.toolchain/'bin';out=R/'build';out.mkdir(exist_ok=True);commands=[[str(b/'avr-gcc'),'-std=gnu11','-Os','-Wall','-Wextra','-Werror','-mmcu=atmega328p','-DF_CPU=16000000UL',str(R/'reference_avr.c'),'-o',str(out/'reference.elf')],[str(b/'avr-objcopy'),'-O','ihex','-R','.eeprom',str(out/'reference.elf'),str(out/'reference.hex')],[str(b/'avr-size'),str(out/'reference.elf')]];logs=[]
 for cmd in commands:
  r=subprocess.run(cmd,capture_output=True,text=True);logs.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
  if r.returncode:break
 (out/'firmware_build.json').write_text(json.dumps(logs,indent=2));assert len(logs)==3 and all(r['returncode']==0 for r in logs);print(logs[-1]['stdout']);print('Actual AVR build; no physical flashing or capture')
