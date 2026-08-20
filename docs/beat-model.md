# Beat model data and reproducibility

No copyrighted audio or third-party spectrogram is committed here. The generated
model must be reviewed against the licenses of every chosen source before release.

## Sources

- GiantSteps Tempo: https://github.com/GiantSteps/giantsteps-tempo-dataset
  contains tempo/genre annotations, predefined recording splits and a downloader
  for 664 Beatport previews (~1 GB). The repository does not state a blanket audio
  redistribution license; keep audio under `data/` and never commit it.
- Beat This! annotations v1.1:
  https://github.com/CPJKU/beat_this_annotations (MIT for the annotation
  repository; individual source datasets retain their own terms).
- Beat This! spectrogram release:
  https://doi.org/10.5281/zenodo.13922116. Files have dataset-specific rights.
  The collection is 141.2 GB; do not fetch all by default. The published feature
  definition is mono 22050 Hz, FFT 1024, hop 441, 128 Slaney Mel bands, magnitude
  `ln(1 + 1000*x)`.
- Beat This! code and weights: https://github.com/CPJKU/beat_this (MIT). The
  ~78 MB model is reference/teacher only. Its ~8.1 MB `small*` model is also a
  useful reference but is not treated as this project's independently trained model.

GTZAN is test-only. No GTZAN recording or augmentation may occur in train or
validation. Augmentations inherit the original `recording_id`; the training
loader rejects a recording ID present in more than one split.

## Deliberately bounded experiment

Start with Candombe (2.1 GB) + SMC (2.1 GB) + GuitarSet (1.4 GB) for training,
and use the official recording-level split to reserve validation. Add Ballroom
(4.8 GB) only after the first ablation. Download GTZAN (306.9 MB) only for the
held-out final test. This is 5.6 GB initially, 10.4 GB with Ballroom, rather than
blindly downloading 141.2 GB.

Clone annotations:

```sh
mkdir -p data
git clone --branch v1.1 --depth 1 https://github.com/CPJKU/beat_this_annotations data/beat_this_annotations
git clone --depth 1 https://github.com/GiantSteps/giantsteps-tempo-dataset data/giantsteps-tempo
```

Download selected spectrogram archives from the Zenodo record into
`data/downloads/`, verify the published MD5 values, and extract outside git.
A preparation adapter must write JSONL rows consumed by `training/train.py`:
`feature`, stable `recording_id`, `split`, `dataset`, and optional `genre`.
Each feature NPZ contains `mel [frames,128]`, `beat [frames]`, and only when
annotated, `downbeat [frames]`. Missing downbeats are masked, never negatives.

## Train and export the production model

```sh
python3 -m venv .venv
.venv/bin/pip install -r training/requirements.txt
.venv/bin/python training/train_causal_beat_this.py \
  data/prepared/manifest.jsonl data/beat_this-models/small0.ckpt \
  --reference-root data/beat_this-reference \
  --out runs/local84-small0-guitar-pilot \
  --epochs 4 --frames 800 --context-frames 84
.venv/bin/python training/prepare_audio_distillation.py \
  data/giantsteps-tempo/production-subset/manifest.json \
  data/giantsteps-distillation
.venv/bin/python training/generate_teacher.py \
  data/giantsteps-distillation/manifest.jsonl \
  data/beat_this-models/small0.ckpt
.venv/bin/python training/train_causal_beat_this.py \
  data/prepared/manifest.jsonl \
  runs/local84-small0-giantsteps-distilled/epoch-001.ckpt \
  --extra-manifest data/giantsteps-distillation/manifest.jsonl \
  --extra-manifest data/giantsteps-distillation-round2/manifest.jsonl \
  --reference-root data/beat_this-reference \
  --out runs/local84-small0-giantsteps-distilled-r2 \
  --epochs 2 --frames 800 --context-frames 84 --clips-per-recording 2 \
  --learning-rate 5e-6 --distill-alpha 1 --seed 20260819
.venv/bin/python training/export_causal_beat_this.py \
  runs/local84-small0-giantsteps-distilled-r2/epoch-001.ckpt \
  assets/beat_tracker.onnx \
  --reference-root data/beat_this-reference \
  --frames 112 --attention-context 84
cargo test
cargo run --release --bin terb-bpm -- analyze song.wav
```

The seed, exact manifest, configuration, best validation loss, parameter count
and ONNX byte size are stored in the run directory. ONNX export is fixed opset
17 with static `[1,112,128]` (`batch,time,mel`) input. Time attention is causal
with 84-frame per-layer context. Four symmetric frontend convolutions retain
four frames of future context. Production inference advances by 10 frames,
giving a measured worst scheduling delay of 260ms. No user-facing latency option
is added.

The default ONNX is embedded in the Rust binary. `--model` and
`TERB_BPM_MODEL` are explicit development/override paths only. Runtime inference
uses pure-Rust RTen and does not require Python or ONNX Runtime.

## Benchmark manifest

`terb-bpm-benchmark` accepts a JSON array:

```json
[{"path":"/data/test/song.wav","bpm":128.0,"dataset":"gtzan","split":"test","genre":"disco"}]
```

It streams each WAV in the selected chunk size, automatically averages channels,
and writes `results.json`, `results.csv`, and `report.md`. Run each of
`--chunk 128`, `512`, and `2048` to verify chunk invariance. For a fair
Beat This! reference, use its `final*`/small model only on GTZAN as its
documentation warns that other published datasets overlap its training set.
