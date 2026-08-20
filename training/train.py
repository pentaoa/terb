"""Train the compact causal beat activation model.

Input manifest is JSONL. Each row has: feature (.npz), recording_id, split and
optional dataset/genre. NPZ keys: mel [T,128], beat [T], optional downbeat [T].
Splits are recording-level and are checked before training.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CausalConv(nn.Module):
    def __init__(self, cin, cout, kernel=5, dilation=1):
        super().__init__(); self.pad=(kernel-1)*dilation
        self.conv=nn.Conv1d(cin,cout,kernel,dilation=dilation,padding=self.pad)
    def forward(self,x): return self.conv(x)[...,:x.shape[-1]]

class DepthwiseCausalConv(nn.Module):
    def __init__(self,channels,kernel=5,dilation=1):
        super().__init__(); self.pad=(kernel-1)*dilation
        self.conv=nn.Conv1d(channels,channels,kernel,dilation=dilation,padding=self.pad,groups=channels)
    def forward(self,x): return self.conv(x)[...,:x.shape[-1]]

class TinyBeatTCN(nn.Module):
    def __init__(self, channels=64, downbeat=False, dilations=(1,2,4,8,16,32), separable=False):
        super().__init__(); self.front=CausalConv(128,channels,5)
        conv=lambda d: DepthwiseCausalConv(channels,5,d) if separable else CausalConv(channels,channels,5,d)
        self.blocks=nn.ModuleList([nn.Sequential(conv(d),nn.ReLU(),nn.Conv1d(channels,channels,1)) for d in dilations])
        self.head=nn.Conv1d(channels,2 if downbeat else 1,1)
    def forward(self,x):
        x=torch.relu(self.front(x))
        for block in self.blocks: x=x+block(x)
        return self.head(x)

def balance_by_dataset(rows,weights=None):
    groups={}
    for row in rows: groups.setdefault(row.get("dataset","unknown"),[]).append(row)
    if weights:
        missing=set(groups)-set(weights)
        if missing: raise ValueError(f"missing dataset weights for {sorted(missing)}")
        if any(weights[name]<=0 for name in groups): raise ValueError("dataset weights must be positive")
        scale=max(len(groups[name])/weights[name] for name in groups)
        counts={name:max(1,round(scale*weights[name])) for name in groups}
    else:
        target=max(map(len,groups.values())); counts={name:target for name in groups}
    balanced=[]
    for name in sorted(groups):
        group=groups[name]
        balanced.extend(group[i%len(group)] for i in range(counts[name]))
    return balanced

def shift_mel(mel, bins):
    if bins==0: return mel
    out=np.zeros_like(mel)
    if bins>0: out[:,bins:]=mel[:,:-bins]
    else: out[:,:bins]=mel[:,-bins:]
    return out

_FEATURE_CACHE={}

def load_feature(path,cache=False):
    if not cache: return np.load(path)
    if path not in _FEATURE_CACHE:
        with np.load(path) as z: _FEATURE_CACHE[path]={k:z[k] for k in z.files}
    return _FEATURE_CACHE[path]

class Clips(Dataset):
    def __init__(self, rows, frames, seed, repeats=16, cache_features=False):
        self.rows=rows; self.frames=frames; self.seed=seed; self.repeats=repeats; self.cache_features=cache_features
    def __len__(self): return len(self.rows)*self.repeats
    def __getitem__(self,index):
        rng=np.random.default_rng(self.seed+index); z=load_feature(self.rows[index%len(self.rows)]["feature"],self.cache_features)
        mel=z["mel"].astype("float32"); beat=z["beat"].astype("float32")
        down=z["downbeat"].astype("float32") if "downbeat" in z else np.zeros_like(beat)
        mask=np.ones_like(beat) if "downbeat" in z else np.zeros_like(beat)
        teacher_beat=z["teacher_beat"].astype("float32") if "teacher_beat" in z else beat.copy()
        teacher_down=z["teacher_downbeat"].astype("float32") if "teacher_downbeat" in z else down.copy()
        # Feature-domain augmentations preserve or synchronously transform labels.
        mel += rng.normal(0, rng.uniform(0,0.04), mel.shape).astype("float32")
        mel *= rng.uniform(0.85,1.15)
        mel=shift_mel(mel,int(rng.integers(-3,4))) # zero-filled spectral shift; no high/low wrap
        speed=float(rng.uniform(.94,1.06))
        old=np.arange(len(mel)); new=np.arange(0,len(mel)-1,speed)
        mel=np.stack([np.interp(new,old,mel[:,m]) for m in range(128)],1).astype("float32")
        beat=np.interp(new,old,beat).astype("float32"); down=np.interp(new,old,down).astype("float32"); mask=np.interp(new,old,mask).astype("float32")
        teacher_beat=np.interp(new,old,teacher_beat).astype("float32"); teacher_down=np.interp(new,old,teacher_down).astype("float32")
        if len(mel)<self.frames:
            pad=self.frames-len(mel); mel=np.pad(mel,((pad,0),(0,0))); beat=np.pad(beat,(pad,0)); down=np.pad(down,(pad,0)); mask=np.pad(mask,(pad,0)); teacher_beat=np.pad(teacher_beat,(pad,0)); teacher_down=np.pad(teacher_down,(pad,0))
        start=int(rng.integers(0,max(1,len(mel)-self.frames+1))); sl=slice(start,start+self.frames)
        return tuple(torch.from_numpy(x[sl].copy()) for x in (mel,beat,down,mask,teacher_beat,teacher_down))

def loss_fn(logits, beat, down, mask, teacher_beat, teacher_down, alpha):
    beat_gt=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],beat,pos_weight=torch.tensor(8.,device=logits.device))
    beat_soft=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],teacher_beat)
    beat_loss=(1-alpha)*beat_gt+alpha*beat_soft
    if logits.shape[1]==1 or mask.sum()==0: return beat_loss
    raw_gt=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1],down,reduction="none",pos_weight=torch.tensor(16.,device=logits.device))
    raw_soft=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1],teacher_down,reduction="none")
    down_loss=(((1-alpha)*raw_gt+alpha*raw_soft)*mask).sum()/mask.sum().clamp_min(1)
    return beat_loss+down_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("--out",default="runs/tiny")
    p.add_argument("--epochs",type=int,default=30); p.add_argument("--seed",type=int,default=20260818)
    p.add_argument("--batch",type=int,default=16); p.add_argument("--clips-per-recording",type=int,default=16); p.add_argument("--frames",type=int,default=800); p.add_argument("--downbeat",action="store_true"); p.add_argument("--distill-alpha",type=float,default=0.0); p.add_argument("--keep-epochs",action="store_true"); p.add_argument("--balance-datasets",action="store_true"); p.add_argument("--cache-features",action="store_true"); p.add_argument("--extended-dilations",action="store_true"); p.add_argument("--channels",type=int,default=64); p.add_argument("--learning-rate",type=float,default=2e-4); p.add_argument("--init-checkpoint"); p.add_argument("--dataset-weights"); p.add_argument("--separable",action="store_true")
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows=[json.loads(x) for x in open(a.manifest) if x.strip()]
    if not 0.0 <= a.distill_alpha <= 1.0: raise ValueError("--distill-alpha must be in [0,1]")
    owners={}
    for r in rows:
        old=owners.setdefault(r["recording_id"],r["split"])
        if old!=r["split"]: raise ValueError(f'data leakage: {r["recording_id"]} occurs in {old} and {r["split"]}')
    train=[r for r in rows if r["split"]=="train"]; valid=[r for r in rows if r["split"]=="validation"]
    unique_train_recordings=len(train)
    weights=None
    if a.dataset_weights:
        weights={part.split("=",1)[0]:float(part.split("=",1)[1]) for part in a.dataset_weights.split(",")}
    if a.balance_datasets or weights: train=balance_by_dataset(train,weights)
    if not train or not valid: raise ValueError("manifest needs non-empty recording-level train and validation sets")
    device="cuda" if torch.cuda.is_available() else "cpu"; dilations=(1,2,4,8,16,32,64,128) if a.extended_dilations else (1,2,4,8,16,32); model=TinyBeatTCN(channels=a.channels,downbeat=a.downbeat,dilations=dilations,separable=a.separable).to(device)
    if a.init_checkpoint:
        init=torch.load(a.init_checkpoint,map_location=device,weights_only=True); model.load_state_dict(init["state_dict"])
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(a)|{"device":device,"train_recordings":unique_train_recordings,"sampled_train_rows":len(train),"validation_recordings":len(valid)},indent=2))
    best=1e9; history=[]
    for epoch in range(a.epochs):
        model.train()
        for mel,beat,down,mask,teacher_beat,teacher_down in DataLoader(Clips(train,a.frames,a.seed+epoch*100000,a.clips_per_recording,a.cache_features),batch_size=a.batch,shuffle=True):
            logits=model(mel.transpose(1,2).to(device)); loss=loss_fn(logits,beat.to(device),down.to(device),mask.to(device),teacher_beat.to(device),teacher_down.to(device),a.distill_alpha)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),3); opt.step()
        model.eval(); losses=[]
        with torch.no_grad():
            for mel,beat,down,mask,teacher_beat,teacher_down in DataLoader(Clips(valid,a.frames,a.seed,a.clips_per_recording,a.cache_features),batch_size=a.batch):
                logits=model(mel.transpose(1,2).to(device)); losses.append(float(loss_fn(logits,beat.to(device),down.to(device),mask.to(device),teacher_beat.to(device),teacher_down.to(device),a.distill_alpha)))
        val=float(np.mean(losses)); event={"epoch":epoch+1,"validation_loss":val}; history.append(event); (out/"history.json").write_text(json.dumps(history,indent=2)); print(json.dumps(event),flush=True)
        if a.keep_epochs: torch.save({"state_dict":model.state_dict(),"downbeat":a.downbeat,"seed":a.seed,"epoch":epoch+1,"validation_loss":val,"dilations":dilations,"channels":a.channels,"separable":a.separable},out/f"epoch-{epoch+1:03d}.pt")
        if val<best: best=val; torch.save({"state_dict":model.state_dict(),"downbeat":a.downbeat,"seed":a.seed,"dilations":dilations,"channels":a.channels,"separable":a.separable},out/"best.pt")
    model.load_state_dict(torch.load(out/"best.pt",map_location=device,weights_only=True)["state_dict"]); model.eval()
    dummy=torch.zeros(1,128,256,device=device)
    torch.onnx.export(model,dummy,out/"beat_tracker.onnx",input_names=["mel"],output_names=["activation_logits"],opset_version=17,dynamic_axes=None)
    (out/"metrics.json").write_text(json.dumps({"best_validation_loss":best,"parameters":sum(p.numel() for p in model.parameters()),"model_bytes":(out/"beat_tracker.onnx").stat().st_size},indent=2))
if __name__=="__main__": main()
