"""Evaluate the fixed Rust-style decoder; tune parameters on validation only."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from train import TinyBeatTCN
from evaluate import peaks,f1,reference,category
FPS=50.
def ac(x,lag):
 k=int(lag);q=lag-k
 if k+1>=len(x):return 0.
 delayed=x[:len(x)-k-1]*(1-q)+x[1:len(x)-k]*q;now=x[k+1:]
 return float(np.dot(now,delayed)/np.sqrt(np.dot(now,now)*np.dot(delayed,delayed)+1e-8))
def tempo(a,seconds=None):
 if seconds:a=a[:int(seconds*FPS)]
 x=np.maximum(a-a.mean(),0)
 if len(x)<100 or np.dot(x,x)<1e-6:return None
 p=np.asarray([i for i in range(1,len(x)-1) if x[i]>=.18 and x[i]>x[i-1] and x[i]>=x[i+1]])
 intervals=np.diff(p);best=(-1.,None)
 for bpm in np.arange(60,210.001,.25):
  lag=FPS*60/bpm;iv=0.
  if len(intervals):
   direct=np.exp(-(intervals-lag)**2/(2*1.5**2));double=.65*np.exp(-(intervals-2*lag)**2/8);iv=float(np.maximum(direct,double).mean())
  prior=np.exp(-.10*abs(np.log2(bpm/120.)))
  score=(.62*ac(x,lag)+.28*iv+.06)*prior
  if score>best[0]:best=(score,float(bpm))
 return best[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('checkpoint');p.add_argument('--split',default='validation');p.add_argument('--out',required=True);a=p.parse_args()
 rows=[json.loads(line) for line in open(a.manifest) if json.loads(line)['split']==a.split]
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m=TinyBeatTCN(channels=ck.get('channels',64),downbeat=ck['downbeat'],dilations=tuple(ck.get('dilations',(1,2,4,8,16,32))),separable=ck.get('separable',False));m.load_state_dict(ck['state_dict']);m.eval();device='cuda' if torch.cuda.is_available() else 'cpu';m.to(device);out=[]
 with torch.no_grad():
  for r in rows:
   z=np.load(r['feature']);act=torch.sigmoid(m(torch.from_numpy(z['mel'].astype('float32')).T[None].to(device)))[0].cpu().numpy();ref=reference(z['beat']);item={'recording_id':r['recording_id'],'dataset':r.get('dataset','unknown'),'reference_bpm':ref,'beat_f1':f1(peaks(act[0],.5),peaks(z['beat'],.9))}
   for sec in (4,8,16,30):item[f'bpm_{sec}s']=tempo(act[0],sec)
   item['bpm_final']=tempo(act[0]);item['classification']=category(item['bpm_final'],ref);out.append(item)
 n=len(out);rate=lambda k:sum(x['classification']==k for x in out)/max(1,n);summary={'split':a.split,'recordings':n,'strict_accuracy':rate('correct'),'metrical_accuracy':sum(rate(k) for k in ('correct','half_time','double_time')),'half_time_rate':rate('half_time'),'double_time_rate':rate('double_time'),'wrong_rate':rate('wrong')}
 for sec in (4,8,16,30):summary[f'accuracy_{sec}s']=sum(category(x[f'bpm_{sec}s'],x['reference_bpm'])=='correct' for x in out)/max(1,n)
 per_dataset={}
 for dataset in sorted({x['dataset'] for x in out}):
  subset=[x for x in out if x['dataset']==dataset];dn=len(subset);dr=lambda k:sum(x['classification']==k for x in subset)/dn
  per_dataset[dataset]={'recordings':dn,'strict_accuracy':dr('correct'),'metrical_accuracy':sum(dr(k) for k in ('correct','half_time','double_time')),'half_time_rate':dr('half_time'),'double_time_rate':dr('double_time'),'wrong_rate':dr('wrong'),'accuracy_4s':sum(category(x['bpm_4s'],x['reference_bpm'])=='correct' for x in subset)/dn}
 summary['per_dataset']=per_dataset
 Path(a.out).write_text(json.dumps({'summary':summary,'recordings':out},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
