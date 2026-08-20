"""Evaluate FrequencyGRU with true state-carrying streaming inference."""
import argparse, json
from pathlib import Path
import numpy as np
import torch

from evaluate import category, f1, peaks, reference
from evaluate_production_decoder import tempo
from train_frequency_gru import load_student


def predict(model, mel, chunk, device):
    state = None; outputs = []
    with torch.no_grad():
        for start in range(0, len(mel), chunk):
            x = torch.from_numpy(mel[start:start + chunk].astype("float32"))[None].to(device)
            logits, state = model(x, state); outputs.append(torch.sigmoid(logits)[0].cpu().numpy())
    return np.concatenate(outputs, axis=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("manifest"); p.add_argument("checkpoint"); p.add_argument("teacher_checkpoint")
    p.add_argument("--reference-root", default="data/beat_this-reference"); p.add_argument("--split", default="validation"); p.add_argument("--chunk", type=int, default=5); p.add_argument("--out", required=True)
    a = p.parse_args(); device = "cuda" if torch.cuda.is_available() else "cpu"; model = load_student(a.checkpoint, a.teacher_checkpoint, a.reference_root, device).eval()
    rows = [json.loads(x) for x in open(a.manifest) if x.strip() and json.loads(x)["split"] == a.split]; result = []
    for row in rows:
        with np.load(row["feature"]) as z:
            act = predict(model, z["mel"], a.chunk, device); ref = reference(z["beat"])
            item = {"recording_id": row["recording_id"], "dataset": row.get("dataset", "unknown"), "reference_bpm": ref, "beat_f1": f1(peaks(act[0], .5), peaks(z["beat"], .9))}
            for sec in (4, 8, 16, 30): item[f"bpm_{sec}s"] = tempo(act[0], sec)
            item["bpm_final"] = tempo(act[0]); item["classification"] = category(item["bpm_final"], ref); result.append(item)
    n = len(result); rate = lambda k: sum(x["classification"] == k for x in result) / max(1, n)
    summary = {"split": a.split, "recordings": n, "chunk": a.chunk, "beat_f1": float(np.mean([x["beat_f1"] for x in result])), "strict_accuracy": rate("correct"), "metrical_accuracy": sum(rate(k) for k in ("correct", "half_time", "double_time")), "half_time_rate": rate("half_time"), "double_time_rate": rate("double_time"), "wrong_rate": rate("wrong")}
    for sec in (4, 8, 16, 30): summary[f"accuracy_{sec}s"] = sum(category(x[f"bpm_{sec}s"], x["reference_bpm"]) == "correct" for x in result) / max(1, n)
    Path(a.out).write_text(json.dumps({"summary": summary, "recordings": result}, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
