"""Convert extracted Beat This! spectrograms + v1.1 annotations to training NPZ/JSONL."""
import argparse, json
from pathlib import Path
import numpy as np

FPS=50.0
def target(times, frames, sigma=1.5):
    y=np.zeros(frames,np.float32); grid=np.arange(frames)
    for time in times:
        center=time*FPS; y=np.maximum(y,np.exp(-0.5*((grid-center)/sigma)**2))
    return y
def read_beats(path):
    rows=[]
    for line in path.read_text().splitlines():
        p=line.split()
        if p: rows.append((float(p[0]),int(p[1]) if len(p)>1 else None))
    return rows
def main():
    p=argparse.ArgumentParser(); p.add_argument("spectrogram_root"); p.add_argument("annotations_root"); p.add_argument("output")
    p.add_argument("--datasets",nargs="+",default=["candombe","smc","guitarset","ballroom","gtzan"]); a=p.parse_args()
    root=Path(a.spectrogram_root); ann=Path(a.annotations_root); out=Path(a.output); (out/"features").mkdir(parents=True,exist_ok=True)
    manifest=[]
    for dataset in a.datasets:
        before=len(manifest)
        archive_path=root/f"{dataset}.npz"
        archive=np.load(archive_path,allow_pickle=False) if archive_path.is_file() else None
        dataset_dir=root/dataset
        if archive is None and not dataset_dir.is_dir(): raise FileNotFoundError(f"missing requested dataset {dataset}: expected {archive_path} or {dataset_dir}")
        split_file=ann/dataset/"single.split"
        splits=dict(line.split()[:2] for line in split_file.read_text().splitlines() if line.strip()) if split_file.exists() else {}
        for beat_path in sorted((ann/dataset/"annotations"/"beats").glob("*.beats")):
            rid=beat_path.stem; split={"val":"validation"}.get(splits.get(rid,"test"),splits.get(rid,"test"))
            if dataset=="gtzan": split="test"
            if split not in {"train","validation","test"}: raise ValueError(f"{dataset}:{rid}: invalid split {split}")
            candidates=list((root/dataset/rid).glob("track.npy"))
            if not candidates: candidates=list((root/dataset).glob(f"**/{rid}/track.npy"))
            archive_key=f"{rid}/track"
            if candidates:
                mel=np.load(candidates[0]).astype("float32")
                source=str(candidates[0])
            elif archive is not None and archive_key in archive:
                mel=archive[archive_key].astype("float32")
                source=f"{archive_path}:{archive_key}"
            else: continue
            if mel.ndim!=2: raise ValueError(f"{source}: expected 2-D mel")
            if mel.shape[0]==128: mel=mel.T
            if mel.shape[1]!=128: raise ValueError(f"{source}: expected 128 Mel bands, got {mel.shape}")
            if not np.isfinite(mel).all(): raise ValueError(f"{source}: non-finite Mel value")
            beats=read_beats(beat_path); arrays={"mel":mel,"beat":target([x[0] for x in beats],len(mel))}
            if beats and all(x[1] is not None for x in beats):
                arrays["downbeat"]=target([t for t,n in beats if n==1],len(mel))
            feature=out/"features"/f"{dataset}__{rid}.npz"; np.savez(feature,**arrays)
            manifest.append({"feature":str(feature.resolve()),"recording_id":f"{dataset}:{rid}","split":split,"dataset":dataset})
        if len(manifest)==before: raise ValueError(f"requested dataset {dataset} produced no recordings")
    ids=[x["recording_id"] for x in manifest]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate recording_id in prepared manifest")
    (out/"manifest.jsonl").write_text("".join(json.dumps(x)+"\n" for x in manifest))
    counts={}; [counts.__setitem__((x["dataset"],x["split"]),counts.get((x["dataset"],x["split"]),0)+1) for x in manifest]
    print(json.dumps({"recordings":len(manifest),"counts":{f"{k[0]}/{k[1]}":v for k,v in counts.items()}},indent=2))
if __name__=="__main__": main()
