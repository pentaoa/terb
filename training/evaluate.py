"""Evaluate exported checkpoint activations and tempo on a recording-level split."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from train import TinyBeatTCN

FPS=50.
def peaks(x, threshold):
 return np.asarray([i for i in range(1,len(x)-1) if x[i]>=threshold and x[i]>x[i-1] and x[i]>=x[i+1]],int)
def f1(pred,true,tol=3):
 used=set(); hit=0
 for p in pred:
  candidates=[(abs(int(p)-int(t)),j) for j,t in enumerate(true) if j not in used and abs(int(p)-int(t))<=tol]
  if candidates:
   _,j=min(candidates);used.add(j);hit+=1
 return 2*hit/max(1,len(pred)+len(true))
def tempo(x,seconds=None):
 if seconds: x=x[:int(seconds*FPS)]
 x=np.maximum(x-x.mean(),0); n=len(x)
 if n<100 or np.dot(x,x)<1e-6:return None
 best=(-1,None)
 for bpm in np.arange(60,210.001,.25):
  lag=FPS*60/bpm; k=int(lag); frac=lag-k
  if k+1>=n:continue
  delayed=x[:n-k-1]*(1-frac)+x[1:n-k]*frac; now=x[k+1:]
  score=float(np.dot(now,delayed)/np.sqrt((np.dot(now,now)*np.dot(delayed,delayed)+1e-8)))
  if score>best[0]:best=(score,float(bpm))
 return best[1]
def reference(y):
 p=peaks(y,.9)
 return float(60*FPS/np.median(np.diff(p))) if len(p)>2 else None
def category(est,ref):
 if est is None:return "no_result"
 rel=lambda target:abs(est-target)/target
 if rel(ref)<=.04:return "correct"
 if rel(ref*.5)<=.04:return "half_time"
 if rel(ref*2)<=.04:return "double_time"
 return "wrong"
def main():
 p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("checkpoint");p.add_argument("--split",default="validation");p.add_argument("--out",default="metrics-eval.json");a=p.parse_args()
 rows=[json.loads(x) for x in open(a.manifest) if json.loads(x)["split"]==a.split]
 ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True);model=TinyBeatTCN(channels=ck.get("channels",64),downbeat=ck["downbeat"],dilations=tuple(ck.get("dilations",(1,2,4,8,16,32))),separable=ck.get("separable",False));model.load_state_dict(ck["state_dict"]);model.eval()
 results=[]
 with torch.no_grad():
  for r in rows:
   z=np.load(r["feature"]); mel=torch.from_numpy(z["mel"].astype("float32")).T[None]
   logits=model(mel); act=torch.sigmoid(logits)[0].numpy(); ref=reference(z["beat"])
   item=dict(recording_id=r["recording_id"],reference_bpm=ref,beat_f1=f1(peaks(act[0],.5),peaks(z["beat"],.9)))
   if "downbeat" in z and act.shape[0]>1:item["downbeat_f1"]=f1(peaks(act[1],.5),peaks(z["downbeat"],.9))
   for sec in (4,8,16,30):item[f"bpm_{sec}s"]=tempo(act[0],sec)
   item["bpm_final"]=tempo(act[0]);item["classification"]=category(item["bpm_final"],ref);results.append(item)
 n=len(results); rate=lambda k:sum(x["classification"]==k for x in results)/max(1,n)
 summary={"split":a.split,"recordings":n,"beat_f1":float(np.mean([x["beat_f1"] for x in results])), "downbeat_f1":float(np.mean([x.get("downbeat_f1",0) for x in results])), "strict_accuracy":rate("correct"),"metrical_accuracy":rate("correct")+rate("half_time")+rate("double_time"),"half_time_rate":rate("half_time"),"double_time_rate":rate("double_time"),"wrong_rate":rate("wrong")}
 for sec in (4,8,16,30):summary[f"accuracy_{sec}s"]=sum(category(x[f"bpm_{sec}s"],x["reference_bpm"])=="correct" for x in results)/max(1,n)
 Path(a.out).write_text(json.dumps({"summary":summary,"recordings":results},indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
