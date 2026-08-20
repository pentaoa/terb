"""Evaluate causal transformer exactly as a fixed rolling-window stream."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from causal_beat_this import load_causal_checkpoint
from evaluate import peaks,f1,reference,category
from evaluate_production_decoder import tempo
from evaluate_causal_reference import summarize


def rolling_predict(model,mel,device,window,hop,lookahead,batch_size):
    total=len(mel); activation=np.zeros(total,np.float32)
    specs=[]
    for start in range(0,total,hop):
        target_end=min(total,start+hop)
        observed_end=min(total,target_end+lookahead)
        source_start=max(0,observed_end-window)
        chunk=mel[source_start:observed_end]
        left=window-len(chunk)
        padded=np.pad(chunk,((left,0),(0,0)))
        positions=np.arange(start,target_end)-source_start+left
        specs.append((padded,positions,start,target_end))
    with torch.inference_mode():
        for offset in range(0,len(specs),batch_size):
            group=specs[offset:offset+batch_size]
            x=torch.from_numpy(np.stack([g[0] for g in group]).astype("float32")).to(device)
            pred=torch.sigmoid(model(x)["beat"]).cpu().numpy()
            for row,(_,positions,start,end) in zip(pred,group):
                activation[start:end]=row[positions]
    return activation


def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest");p.add_argument("checkpoint");p.add_argument("--out",required=True)
    p.add_argument("--reference-root",default="data/beat_this-reference")
    p.add_argument("--split",default="validation")
    p.add_argument("--window",type=int,default=256);p.add_argument("--hop",type=int,default=5)
    p.add_argument("--lookahead",type=int,default=4);p.add_argument("--attention-context",type=int,default=84)
    p.add_argument("--batch",type=int,default=32)
    a=p.parse_args();rows=[json.loads(x) for x in open(a.manifest) if json.loads(x)["split"]==a.split]
    if not rows: raise ValueError(f"split {a.split} has no recordings")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_causal_checkpoint(a.reference_root,a.checkpoint,device,a.attention_context).eval()
    items=[];started=time.perf_counter()
    for row in rows:
        z=np.load(row["feature"]);act=rolling_predict(model,z["mel"],device,a.window,a.hop,a.lookahead,a.batch)
        ref=reference(z["beat"]);item={"recording_id":row["recording_id"],"dataset":row.get("dataset","unknown"),"reference_bpm":ref,"beat_f1":f1(peaks(act,.5),peaks(z["beat"],.9))}
        for sec in (4,8,16,30):item[f"bpm_{sec}s"]=tempo(act,sec)
        item["bpm_final"]=tempo(act);item["classification"]=category(item["bpm_final"],ref);items.append(item)
    summary=summarize(items,time.perf_counter()-started)
    summary.update({"window":a.window,"hop":a.hop,"lookahead":a.lookahead,"max_algorithmic_delay_ms":20*(a.lookahead+a.hop-1)})
    Path(a.out).write_text(json.dumps({"summary":summary,"recordings":items},indent=2));print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
