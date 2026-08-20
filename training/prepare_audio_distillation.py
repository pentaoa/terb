#!/usr/bin/env python3
"""Prepare PCM WAV recordings for teacher-only activation distillation.

The input is the benchmark JSON manifest. Tempo labels are intentionally not
converted into synthetic beat locations. Beat/downbeat arrays only provide
shape; training must use --distill-alpha 1 so loss comes exclusively from the
teacher probabilities added by generate_teacher.py.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from features import extract


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected PCM16, got {wav.getsampwidth() * 8}-bit")
        channels = wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        samples = samples.reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0
        return samples, wav.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    feature_dir = args.output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for row in rows:
        path = Path(row["path"])
        audio, rate = read_pcm16(path)
        mel = extract(audio, rate)
        shape_only = np.zeros(len(mel), dtype=np.float32)
        feature = feature_dir / f"giantsteps__{path.stem}.npz"
        np.savez(feature, mel=mel, beat=shape_only, downbeat=shape_only)
        prepared.append(
            {
                "feature": str(feature.resolve()),
                "recording_id": f"giantsteps:{path.stem}",
                "split": "train",
                "dataset": "giantsteps-tempo-distillation",
                "distillation_only": True,
            }
        )
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in prepared), encoding="utf-8")
    print(json.dumps({"recordings": len(prepared), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
