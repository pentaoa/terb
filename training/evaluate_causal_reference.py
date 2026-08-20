"""Evaluate bounded-lookahead Beat This! checkpoint on a recording split."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from causal_beat_this import load_causal_checkpoint
from evaluate import peaks,f1,reference,category
from evaluate_production_decoder import tempo


def summarize(items,seconds):
    n=len(items)
    rate=lambda k:sum(x["classification"]==k for x in items)/max(1,n)
    out={
        "recordings":n,"seconds":seconds,
        "beat_f1":float(np.mean([x["beat_f1"] for x in items])),
        "downbeat_f1":float(np.mean([x.get("downbeat_f1",0) for x in items])),
        "strict_accuracy":rate("correct"),
        "metrical_accuracy":sum(rate(k) for k in ("correct","half_time","double_time")),
        "half_time_rate":rate("half_time"),
        "double_time_rate":rate("double_time"),
        "wrong_rate":rate("wrong"),
    }
    for sec in (4,8,16,30):
        out[f"accuracy_{sec}s"]=sum(category(x[f"bpm_{sec}s"],x["reference_bpm"])=="correct" for x in items)/max(1,n)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("checkpoint")
    p.add_argument("--reference-root",default="data/beat_this-reference")
    p.add_argument("--split",default="validation")
    p.add_argument("--out",required=True)
    p.add_argument("--context-frames",type=int,default=0)
    a=p.parse_args()
    rows=[json.loads(x) for x in open(a.manifest) if json.loads(x)["split"]==a.split]
    if not rows: raise ValueError(f"split {a.split} has no recordings in {a.manifest}")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_causal_checkpoint(a.reference_root,a.checkpoint,device,a.context_frames or None).eval()
    items=[]
    started=time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            z=np.load(row["feature"])
            x=torch.from_numpy(z["mel"].astype("float32"))[None].to(device)
            pred=model(x)
            beat=torch.sigmoid(pred["beat"])[0].cpu().numpy()
            down=torch.sigmoid(pred["downbeat"])[0].cpu().numpy()
            ref=reference(z["beat"])
            item={
                "recording_id":row["recording_id"],
                "reference_bpm":ref,
                "beat_f1":f1(peaks(beat,.5),peaks(z["beat"],.9)),
            }
            if "downbeat" in z:
                item["downbeat_f1"]=f1(peaks(down,.5),peaks(z["downbeat"],.9))
            for sec in (4,8,16,30): item[f"bpm_{sec}s"]=tempo(beat,sec)
            item["bpm_final"]=tempo(beat)
            item["classification"]=category(item["bpm_final"],ref)
            items.append(item)
    summary=summarize(items,time.perf_counter()-started)
    Path(a.out).write_text(json.dumps({"summary":summary,"recordings":items},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
