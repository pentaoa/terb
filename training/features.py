"""Reference implementation matching src/features.rs; also compares Rust dumps."""
import argparse, wave
import numpy as np
SR=22050; N=1024; HOP=441; MELS=128
def hzmel(x):
 x=np.asarray(x); return np.where(x<1000,x/(200/3),15+np.log(x/1000)/(np.log(6.4)/27))
def melhz(x):
 x=np.asarray(x); return np.where(x<15,x*(200/3),1000*np.exp((np.log(6.4)/27)*(x-15)))
def filters():
 p=melhz(np.linspace(hzmel(30),hzmel(SR/2),MELS+2))*N/SR; bank=np.zeros((MELS,N//2+1),np.float32)
 for m in range(MELS):
  for b in range(max(0,int(np.floor(p[m]))),min(N//2,int(np.ceil(p[m+2])))+1):
   bank[m,b]=max(0,(b-p[m])/(p[m+1]-p[m]) if b<=p[m+1] else (p[m+2]-b)/(p[m+2]-p[m+1]))
 return bank
def extract(x,input_rate):
 pos=np.arange(0,len(x)-1,input_rate/SR); x=np.interp(pos,np.arange(len(x)),x).astype(np.float32)
 x=np.pad(x,(N//2,N//2))
 out=[]; bank=filters(); win=.5-.5*np.cos(2*np.pi*np.arange(N)/N)
 for start in range(0,len(x)-N+1,HOP):
  magnitude=np.abs(np.fft.rfft(x[start:start+N]*win))
  z=np.log1p(1000*(bank@magnitude)/np.sqrt(N)); out.append(z)
 return np.asarray(out,np.float32)
def main():
 p=argparse.ArgumentParser(); p.add_argument("wav"); p.add_argument("--rust-json"); a=p.parse_args()
 with wave.open(a.wav) as w:
  raw=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").reshape(-1,w.getnchannels()).mean(1)/32768
  result=extract(raw,w.getframerate())
 if a.rust_json:
  import json
  other=np.asarray(json.load(open(a.rust_json)),np.float32); d=np.abs(result[:len(other)]-other); print({"max_abs":float(d.max()),"mean_abs":float(d.mean())})
 else: np.save(a.wav+".mel.npy",result)
if __name__=="__main__":main()
