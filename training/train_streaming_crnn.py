"""Train a compact stateful causal CRNN beat/downbeat activation model."""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import Clips, balance_by_dataset, loss_fn


class CausalConv2d(nn.Module):
    def __init__(self, cin, cout, kernel=(5, 1), stride=(2, 1)):
        super().__init__()
        self.left = kernel[1] - 1
        self.freq = kernel[0] // 2
        self.conv = nn.Conv2d(cin, cout, kernel, stride=stride, bias=False)

    def forward(self, x):
        # F.pad order: time-left, time-right, freq-top, freq-bottom.
        return self.conv(torch.nn.functional.pad(x, (self.left, 0, self.freq, self.freq)))


class StreamingBeatCRNN(nn.Module):
    """Frequency CNN + unidirectional GRU. No future-frame dependency."""
    def __init__(self, hidden=128, layers=2, downbeat=True):
        super().__init__()
        channels = (24, 48, 64)
        blocks = []
        cin = 1
        for cout in channels:
            blocks += [CausalConv2d(cin, cout), nn.BatchNorm2d(cout), nn.ReLU()]
            cin = cout
        self.cnn = nn.Sequential(*blocks)
        self.projection = nn.Sequential(nn.Linear(64 * 16, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 if downbeat else 1)
        self.hidden = hidden
        self.layers = layers
        self.downbeat = downbeat

    def encode(self, mel):
        # mel [B,T,128] -> CNN [B,64,16,T] -> tokens [B,T,1024].
        x = self.cnn(mel.transpose(1, 2).unsqueeze(1))
        return self.projection(x.permute(0, 3, 1, 2).flatten(2))

    def forward(self, mel, state=None):
        x, state = self.gru(self.encode(mel), state)
        return self.head(x).transpose(1, 2), state


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location=device, weights_only=True)
    model = StreamingBeatCRNN(ck["hidden"], ck["layers"], ck["downbeat"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device)


class ExportStep(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, mel, state):
        return self.model(mel, state)


def export(model, out, frames):
    model = model.cpu().eval(); wrapper = ExportStep(model)
    mel = torch.zeros(1, frames, 128)
    state = torch.zeros(model.layers, 1, model.hidden)
    torch.onnx.export(
        wrapper, (mel, state), out,
        input_names=["mel", "state_in"], output_names=["activation_logits", "state_out"],
        opset_version=17, dynamic_axes=None, do_constant_folding=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest"); p.add_argument("--out", default="runs/streaming-crnn")
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--batch", type=int, default=12); p.add_argument("--frames", type=int, default=800)
    p.add_argument("--stream-frames", type=int, default=5); p.add_argument("--clips-per-recording", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--distill-alpha", type=float, default=.5)
    p.add_argument("--init-checkpoint")
    p.add_argument("--dataset-weights", default="guitarset=.5,candombe=.2,smc=.3")
    p.add_argument("--cache-features", action="store_true"); p.add_argument("--keep-epochs", action="store_true")
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if not 0 <= a.distill_alpha <= 1: raise ValueError("distill alpha must be in [0,1]")
    rows = [json.loads(x) for x in open(a.manifest) if x.strip()]
    owners = {}
    for row in rows:
        previous = owners.setdefault(row["recording_id"], row["split"])
        if previous != row["split"]: raise ValueError(f"data leakage: {row['recording_id']}")
    train = [x for x in rows if x["split"] == "train"]
    valid = [x for x in rows if x["split"] == "validation"]
    weights = {x.split("=", 1)[0]: float(x.split("=", 1)[1]) for x in a.dataset_weights.split(",")}
    train = balance_by_dataset(train, weights)
    if not train or not valid: raise ValueError("manifest needs train and validation recordings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StreamingBeatCRNN(a.hidden, a.layers, True).to(device)
    if a.init_checkpoint:
        initial = torch.load(a.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial["state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    config = vars(a) | {"device": device, "sampled_train_rows": len(train), "validation_recordings": len(valid)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    best = float("inf"); history = []
    for epoch in range(a.epochs):
        model.train(); train_losses = []
        loader = DataLoader(Clips(train, a.frames, a.seed + epoch * 100000, a.clips_per_recording, a.cache_features), batch_size=a.batch, shuffle=True, num_workers=0)
        for mel, beat, down, mask, teacher_beat, teacher_down in loader:
            logits, _ = model(mel.to(device))
            loss = loss_fn(logits, beat.to(device), down.to(device), mask.to(device), teacher_beat.to(device), teacher_down.to(device), a.distill_alpha)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 3); opt.step(); train_losses.append(float(loss))
        model.eval(); losses = []
        with torch.no_grad():
            loader = DataLoader(Clips(valid, a.frames, a.seed, max(1, a.clips_per_recording // 2), a.cache_features), batch_size=a.batch)
            for mel, beat, down, mask, teacher_beat, teacher_down in loader:
                logits, _ = model(mel.to(device))
                losses.append(float(loss_fn(logits, beat.to(device), down.to(device), mask.to(device), teacher_beat.to(device), teacher_down.to(device), a.distill_alpha)))
        val = float(np.mean(losses)); event = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": val}
        history.append(event); (out / "history.json").write_text(json.dumps(history, indent=2)); print(json.dumps(event), flush=True)
        ck = {"state_dict": model.state_dict(), "hidden": a.hidden, "layers": a.layers, "downbeat": True, "epoch": epoch + 1, "validation_loss": val}
        if a.keep_epochs: torch.save(ck, out / f"epoch-{epoch+1:03d}.pt")
        if val < best: best = val; torch.save(ck, out / "best.pt")
    model = load_model(out / "best.pt").eval(); export(model, out / "beat_tracker.onnx", a.stream_frames)
    meta = {"best_validation_loss": best, "parameters": sum(x.numel() for x in model.parameters()), "model_bytes": (out / "beat_tracker.onnx").stat().st_size, "stream_frames": a.stream_frames, "lookahead_frames": 0}
    (out / "metrics.json").write_text(json.dumps(meta, indent=2)); print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
