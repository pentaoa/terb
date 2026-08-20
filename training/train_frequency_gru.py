"""Distill Beat This!'s pretrained frequency frontend into a stateful GRU."""
import argparse, copy, json, random
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader

from causal_beat_this import load_causal_checkpoint
from train import Clips, balance_by_dataset, loss_fn


def center_time_conv(source):
    old = source
    conv = nn.Conv2d(old.in_channels, old.out_channels, (old.kernel_size[0], 1),
                     stride=old.stride, padding=(old.padding[0], 0), bias=old.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(old.weight[:, :, :, old.kernel_size[1] // 2:old.kernel_size[1] // 2 + 1])
        if old.bias is not None: conv.bias.copy_(old.bias)
    return conv


class FrequencyBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.attn = copy.deepcopy(source.partial.attnF)
        self.ff = copy.deepcopy(source.partial.ffF)
        self.conv = center_time_conv(source.conv2d)
        self.norm = copy.deepcopy(source.norm)
        self.activation = copy.deepcopy(source.activation)

    def forward(self, x):
        b, _, _, t = x.shape
        y = rearrange(x, "b c f t -> (b t) f c")
        y = y + self.attn(y); y = y + self.ff(y)
        x = rearrange(y, "(b t) f c -> b c f t", b=b, t=t)
        return self.activation(self.norm(self.conv(x)))


class FrequencyGRU(nn.Module):
    def __init__(self, teacher, hidden=128, layers=2):
        super().__init__()
        stem = teacher.frontend.stem
        self.input_norm = copy.deepcopy(stem.bn1d)
        self.stem_conv = center_time_conv(stem.conv2d)
        self.stem_norm = copy.deepcopy(stem.bn2d)
        self.stem_activation = copy.deepcopy(stem.activation)
        self.blocks = nn.ModuleList([FrequencyBlock(x) for x in teacher.frontend.blocks])
        self.projection = copy.deepcopy(teacher.frontend.linear)
        self.gru = nn.GRU(128, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        if hidden == 128:
            with torch.no_grad(): self.head.load_state_dict(teacher.task_heads.beat_downbeat_lin.state_dict())
        self.hidden = hidden; self.layers = layers

    def encode(self, mel):
        x = self.input_norm(mel.transpose(1, 2)).unsqueeze(1)
        x = self.stem_activation(self.stem_norm(self.stem_conv(x)))
        for block in self.blocks: x = block(x)
        return self.projection(rearrange(x, "b c f t -> b t (c f)"))

    def forward(self, mel, state=None):
        x, state = self.gru(self.encode(mel), state)
        raw = self.head(x); down = raw[..., 1]; beat = raw[..., 0] + down
        return torch.stack((beat, down), dim=1), state


def load_student(checkpoint, teacher_checkpoint, reference_root, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    teacher = load_causal_checkpoint(reference_root, teacher_checkpoint, device, 84)
    model = FrequencyGRU(teacher, ck["hidden"], ck["layers"]); model.load_state_dict(ck["state_dict"])
    return model.to(device)


class ExportStep(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, mel, state): return self.model(mel, state)


def main():
    p = argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("teacher_checkpoint")
    p.add_argument("--reference-root", default="data/beat_this-reference"); p.add_argument("--out", default="runs/frequency-gru")
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--frames", type=int, default=800); p.add_argument("--stream-frames", type=int, default=5)
    p.add_argument("--clips-per-recording", type=int, default=4); p.add_argument("--hidden", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--distill-alpha", type=float, default=.65)
    p.add_argument("--dataset-weights", default="guitarset=.5,candombe=.2,smc=.3"); p.add_argument("--keep-epochs", action="store_true")
    p.add_argument("--freeze-frontend", action="store_true")
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows = [json.loads(x) for x in open(a.manifest) if x.strip()]; owners = {}
    for row in rows:
        if owners.setdefault(row["recording_id"], row["split"]) != row["split"]: raise ValueError("recording leakage")
    train = [x for x in rows if x["split"] == "train"]; valid = [x for x in rows if x["split"] == "validation"]
    weights = {x.split("=")[0]: float(x.split("=")[1]) for x in a.dataset_weights.split(",")}
    train = balance_by_dataset(train, weights); device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_causal_checkpoint(a.reference_root, a.teacher_checkpoint, device, 84).eval()
    model = FrequencyGRU(teacher, a.hidden, a.layers).to(device); del teacher
    if a.freeze_frontend:
        for name, parameter in model.named_parameters():
            if not name.startswith(("gru.", "head.")): parameter.requires_grad = False
    opt = torch.optim.AdamW((x for x in model.parameters() if x.requires_grad), lr=a.learning_rate, weight_decay=1e-4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "config.json").write_text(json.dumps(vars(a) | {"device": device, "sampled_train_rows": len(train)}, indent=2))
    best = float("inf"); history = []
    for epoch in range(a.epochs):
        model.train(); tl = []
        for values in DataLoader(Clips(train, a.frames, a.seed + epoch * 100000, a.clips_per_recording), batch_size=a.batch, shuffle=True):
            mel, beat, down, mask, tb, td = values; logits, _ = model(mel.to(device)); loss = loss_fn(logits, beat.to(device), down.to(device), mask.to(device), tb.to(device), td.to(device), a.distill_alpha)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3); opt.step(); tl.append(float(loss))
        model.eval(); vl = []
        with torch.no_grad():
            for values in DataLoader(Clips(valid, a.frames, a.seed, max(1, a.clips_per_recording // 2)), batch_size=a.batch):
                mel, beat, down, mask, tb, td = values; logits, _ = model(mel.to(device)); vl.append(float(loss_fn(logits, beat.to(device), down.to(device), mask.to(device), tb.to(device), td.to(device), a.distill_alpha)))
        val = float(np.mean(vl)); event = {"epoch": epoch + 1, "train_loss": float(np.mean(tl)), "validation_loss": val}; history.append(event); (out / "history.json").write_text(json.dumps(history, indent=2)); print(json.dumps(event), flush=True)
        ck = {"state_dict": model.state_dict(), "hidden": a.hidden, "layers": a.layers, "epoch": epoch + 1, "validation_loss": val}
        if a.keep_epochs: torch.save(ck, out / f"epoch-{epoch+1:03d}.pt")
        if val < best: best = val; torch.save(ck, out / "best.pt")
    model = load_student(out / "best.pt", a.teacher_checkpoint, a.reference_root).eval(); dummy = torch.zeros(1, a.stream_frames, 128); state = torch.zeros(a.layers, 1, a.hidden)
    torch.onnx.export(ExportStep(model), (dummy, state), out / "beat_tracker.onnx", input_names=["mel", "state_in"], output_names=["activation_logits", "state_out"], opset_version=17)
    meta = {"best_validation_loss": best, "parameters": sum(x.numel() for x in model.parameters()), "model_bytes": (out / "beat_tracker.onnx").stat().st_size, "lookahead_frames": 0}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
