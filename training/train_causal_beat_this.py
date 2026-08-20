"""Fine-tune a bounded-lookahead Beat This! small checkpoint.

This is an architecture feasibility experiment. A final run must document
pretraining provenance and use a genuinely held-out evaluation set.
"""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from causal_beat_this import load_causal_checkpoint
from train import Clips,loss_fn


def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--extra-manifest",action="append",default=[])
    p.add_argument("checkpoint")
    p.add_argument("--reference-root",default="data/beat_this-reference")
    p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--frames",type=int,default=800)
    p.add_argument("--context-frames",type=int,default=28)
    p.add_argument("--clips-per-recording",type=int,default=2)
    p.add_argument("--learning-rate",type=float,default=2e-5)
    p.add_argument("--distill-alpha",type=float,default=.5)
    p.add_argument("--shift-tolerant",action="store_true")
    p.add_argument("--seed",type=int,default=20260819)
    a=p.parse_args()
    torch.manual_seed(a.seed)
    rows=[json.loads(x) for x in open(a.manifest) if x.strip()]
    for manifest in a.extra_manifest:
        rows.extend(json.loads(x) for x in open(manifest) if x.strip())
    if any(r.get("distillation_only") for r in rows) and a.distill_alpha != 1.0:
        raise ValueError("distillation-only rows require --distill-alpha 1")
    owners={}
    for row in rows:
        old=owners.setdefault(row["recording_id"],row["split"])
        if old!=row["split"]:
            raise ValueError(f'data leakage: {row["recording_id"]} occurs in {old} and {row["split"]}')
    train=[r for r in rows if r["split"]=="train"]
    valid=[r for r in rows if r["split"]=="validation"]
    if not train or not valid: raise ValueError("manifest needs train and validation")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=load_causal_checkpoint(a.reference_root,a.checkpoint,device,a.context_frames)
    original=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if a.shift_tolerant:
        from beat_this.model.loss import ShiftTolerantBCELoss
        beat_shift=ShiftTolerantBCELoss(pos_weight=19).to(device)
        down_shift=ShiftTolerantBCELoss(pos_weight=86).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(a)|{
        "device":device,"train_recordings":len(train),
        "validation_recordings":len(valid),"lookahead_frames":4,"attention_context_frames":a.context_frames,"total_past_receptive_frames":248,
    },indent=2))
    history=[]
    for epoch in range(a.epochs):
        model.train()
        losses=[]
        loader=DataLoader(
            Clips(train,a.frames,a.seed+epoch*100000,a.clips_per_recording),
            batch_size=a.batch,shuffle=True,
        )
        for mel,beat,down,mask,tbeat,tdown in loader:
            pred=model(mel.to(device))
            logits=torch.stack((pred["beat"],pred["downbeat"]),1)
            if a.shift_tolerant:
                beat_d=beat.to(device); down_d=down.to(device); mask_d=mask.to(device); tbeat_d=tbeat.to(device); tdown_d=tdown.to(device)
                gt_beat=beat_shift(logits[:,0],(beat_d>.5).float())
                gt_down=down_shift(logits[:,1],(down_d>.5).float(),mask_d)
                soft_beat=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],tbeat_d)
                soft_down=(torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1],tdown_d,reduction="none")*mask_d).sum()/mask_d.sum().clamp_min(1)
                loss=(1-a.distill_alpha)*(gt_beat+gt_down)+a.distill_alpha*(soft_beat+soft_down)
            else:
                loss=loss_fn(logits,beat.to(device),down.to(device),mask.to(device),tbeat.to(device),tdown.to(device),a.distill_alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); losses.append(float(loss.detach()))
        event={"epoch":epoch+1,"train_loss":sum(losses)/len(losses)}
        history.append(event); (out/"history.json").write_text(json.dumps(history,indent=2))
        state={"model."+k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save({
            "state_dict":state,
            "hyper_parameters":original["hyper_parameters"],
            "causal_time_attention":True,
            "lookahead_frames":4,
            "epoch":epoch+1,
        },out/f"epoch-{epoch+1:03d}.ckpt")
        print(json.dumps(event),flush=True)


if __name__=="__main__":
    main()
