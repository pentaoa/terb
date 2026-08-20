# Beat model experiment report (2026-08-19)

## Reproducibility

- Beat This! annotations v1.1: commit 890407d158078527ab396b49fea3c8a83e5734ee.
- GuitarSet spectrograms: MD5 2bd210bf3e994065641410f2c0bb00fe.
- Official recording split: 153 train, 27 validation.
- GTZAN test-only: 999 recordings, MD5 39a7dfe6a6b0a5279a94d770506db879.
- Seed 20260818; every augmentation inherits the original recording ID.
- GTZAN was not used for model training, teacher generation or threshold fitting.
- Production decoder parameters were selected on GuitarSet validation, then frozen for GTZAN.
- Exact run configurations and generated metrics live under ignored runs/ directories.

## Historical tiny-TCN baseline

The current asset is a 189,378-parameter causal TCN trained for 10 epochs on GuitarSet. ONNX size is 767,780 bytes.

Two decoder reports exist because the first evaluator predated the fixed Rust-style decoder:

| Evaluation | Guitar strict | Guitar metrical | GTZAN strict | GTZAN metrical | GTZAN half | GTZAN double | GTZAN wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial simple autocorrelation | 74.1% | 85.2% | 34.3% | 62.4% | 24.5% | 3.5% | 37.6% |
| Frozen production decoder | 85.2% | 88.9% | 40.64% | 64.46% | 13.71% | 10.11% | 35.54% |

With the production decoder, GTZAN strict accuracy at 4 / 8 / 16 / 30 seconds is 27.3 / 32.2 / 37.1 / 40.7%.

The decoder improvement is real, but the cross-domain model remains far below the production gate. It is not connected to the TUI.

## Official Beat This! small0 reference

The official small0 checkpoint was evaluated on the same 999 GTZAN recordings, without DBN, using the same frozen production decoder:

| beat F1 | downbeat F1 | strict | metrical | half | double | wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 84.86% | 71.76% | 75.58% | 85.19% | 6.41% | 3.20% | 14.81% |

Strict accuracy at 4 / 8 / 16 / 30 seconds is 72.67 / 73.57 / 74.47 / 75.58%.

This proves that activation plus the production tempo decoder can meet the accuracy target. The reference is not a deployment candidate: it is noncausal and its runtime/architecture do not satisfy the small streaming student constraints. It is used only as an offline teacher and quality reference.

## Distillation ablations

Teacher beat/downbeat probabilities were generated for all 180 GuitarSet recordings with official small0. Missing downbeat annotations remain masked.

| Candidate | epochs | distill alpha | checkpoint selection | strict | metrical | half | double | wrong | 4s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| GuitarSet distilled a07 | 30 | 0.7 | lowest validation BCE | 81.48% | 85.19% | 3.70% | 0.0% | 14.81% | 70.37% |
| GuitarSet distilled a03 | 15 | 0.3 | best of all epochs by full-record BPM | 81.48% | 88.89% | 7.41% | 0.0% | 11.11% | 55.56% |

Neither candidate exceeds the current 85.2% GuitarSet strict baseline, so neither was run on GTZAN and neither replaced the asset. The a03 sweep also proves the rejection was not merely caused by selecting checkpoints with BCE instead of BPM.

The a07 candidate improved 4-second accuracy but worsened strict/wrong metrics. This may be useful later as a latency-oriented signal, but it does not pass the current quality gate.

## Historical gate status before the final Transformer

| Metric | Required | Tiny TCN | Official reference |
|---|---:|---:|---:|
| GTZAN strict | >=70% | 40.64% | 75.58% |
| GTZAN metrical | >=85% | 64.46% | 85.19% |
| half-time error | <=10% | 13.71% | 6.41% |
| wrong | <=15% | 35.54% | 14.81% |
| 4-second strict | >=55% | 27.3% | 72.67% |
| CPU RTF | <=0.25 | 0.172 | not a deployment candidate |
| model size | <=15MiB | 0.73MiB | not a deployment candidate |

The historical tiny TCN passes only runtime and size. It was rejected and is no
longer the deployed asset. Final passing results are recorded below.

## Deployment checks

- ONNX opset 17 loads and runs with tract-onnx 0.21.12.
- 30-second stereo 48kHz synthetic stream, 512-sample chunks: RTF 0.172.
- Feature extraction 18.4ms; inference 3.721s; decoder 1.411s; peak RSS 20.7MB.
- Feature lookahead 23.2ms plus up to 200ms batched inference cadence.
- Python/Rust feature parity: max absolute error 0.0001151, mean 0.00001339.
- Chunk sizes 128, 512 and 2048 produce the same final synthetic tempo.
- 75 Rust tests passed at the time of the baseline measurement.

## Historical decision after tiny-TCN experiments

Single-domain distillation is rejected. The primary failure is activation generalization, not ONNX parity, model size or inference speed.

The next experiment is recording-level multi-domain training with GuitarSet plus Candombe, then SMC if needed. Training now supports dataset-balanced sampling, zero-filled spectral shifts, synchronized tempo augmentation, per-epoch checkpoints and complete-record BPM selection. GTZAN remains test-only and will only be evaluated after a candidate passes validation.

## Final bounded-causal Transformer deployment

After the first causal Transformer passed the original gate, a same-audio GiantSteps benchmark exposed an electronic-music domain gap. Two teacher-only continuation rounds added 40 MD5-verified GiantSteps recordings without inventing beat locations from tempo labels. Checkpoints were selected on GuitarSet validation and gated on GTZAN before the final 20-recording GiantSteps holdout was opened. The final deployment is round-2 epoch 1, exported at 112 frames and scheduled every 10 frames.

| Final GTZAN test (999 recordings) | Result |
|---|---:|
| beat F1 | 53.00% |
| strict BPM accuracy | 76.68% |
| metrical accuracy | 92.49% |
| half-time / double-time / wrong | 6.31 / 9.51 / 7.51% |
| 4 / 8 / 16 / 30 second strict | 64.46 / 70.47 / 74.17 / 76.68% |
| rolling window / hop | 112 / 10 frames |
| worst activation scheduling delay | 260ms |

This exact deployment shape passes all predeclared accuracy gates. The previous 128-frame production model reached 70.97% strict; the new 112-frame distilled model reaches 76.68% while reducing CPU work. A 96-frame historical model missed strict accuracy by 0.43 percentage points (69.57%) and was rejected. Dynamic MatMul Int8 reduced the model to 4.30MB but slowed RTen, so FP32 remains the production artifact.

### Rust production measurements

On a 155.69-second real music file converted to 48kHz WAV, the release `terb-bpm` path produced identical BPM/confidence for 128, 512 and 2048-sample input chunks.

| Stage | 512-sample chunks |
|---|---:|
| features | 0.108s |
| model inference | 34.533s |
| decoder | 0.515s |
| total RTF | 0.226 |
| first estimate / stable lock | 4.09 / 4.49s |
| model size | 9,333,229 bytes |
| observed peak RSS | 132MiB in this run |

RTen is pure Rust and the default model is embedded in the executable. TUI integration uses a bounded `try_send` worker queue, so inference does not block the audio or UI path. Explicit missing external models exit with code 2 and a concrete path error.

### Rejected streaming students

- Three-domain 351k-parameter CNN+GRU: validation reached 87.5% strict, but GTZAN reached only 54.65% strict / 79.68% metrical; rejected.
- Pretrained frequency-attention frontend + GRU: streaming equivalence passed, but validation topped out below the existing gate; rejected before GTZAN.
- 128-frame dynamic Int8: output correlation >0.996 with FP32 but slower in pure Rust RTen on this CPU; rejected.

### Same-audio three-way sealed test

The sealed comparison uses a fourth, non-overlapping set of 20 GiantSteps Beatport previews selected only after the model, 112-frame window and decoder were frozen: two recordings from each of ten common genres, excluding all 40 distillation recordings and all prior development/diagnostic subsets. Every MP3 matches the dataset MD5; raw and derived audio remain ignored and are not redistributed.

| Algorithm | strict | metrical | half | double | wrong | 4s | 8s | 16s | 30s | lock | jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy `src/bpm.rs` | 25% | 55% | 30% | 0% | 20% (+25% unstable) | 25% | 25% | 25% | 25% | 4.92s | 0.04 BPM* |
| final causal student | 55% | 55% | 0% | 0% | 45% | 60% | 60% | 55% | 55% | 5.30s | 2.67 BPM |
| Beat This! small reference | 70% | 75% | 5% | 0% | 25% | 65% | 60% | 65% | 70% | n/a | n/a |

The final student improves strict accuracy by 30 points and eliminates the legacy method's half-time and unstable categories, but metrical accuracy only ties it and its 45% fully-wrong rate on this difficult subset remains a real limitation. The offline noncausal teacher is 15 points stronger in strict accuracy. `*` Legacy jitter only covers the 75% of recordings classified as locked. Machine-readable JSON/CSV and per-genre results are generated by `training/merge_reference_benchmark.py`.
